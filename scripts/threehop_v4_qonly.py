#!/usr/bin/env python3
"""3-hop v4: Fix Step1 to use ONLY question tokens, not entire user message.

Also produces per-step visualization showing the 3-hop chain.
"""

import torch
import random
import math
import re
import gc
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half
from kvpress.attention_patch import patch_attention_functions

patch_attention_functions()

ADJECTIVES = ["silent", "golden", "dusty", "hollow", "bitter", "crimson",
              "gentle", "fierce", "mystic", "frozen", "swift", "ancient",
              "bright", "dark", "wild", "calm", "bold", "shy", "proud", "humble"]
NOUNS = ["rabbit", "hawk", "maple", "pine", "stream", "tide", "wolf",
         "eagle", "river", "stone", "flame", "shadow", "storm", "leaf",
         "crystal", "thunder", "ocean", "moon", "star", "wind"]


def build_niah_prompt(num_distractors, target_key, target_value, needle_pos_frac, seed=50):
    random.seed(seed)
    needles = []
    used_keys = {target_key}
    for _ in range(num_distractors):
        while True:
            key = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
            if key not in used_keys:
                used_keys.add(key)
                break
        value = str(random.randint(1000000, 9999999))
        needles.append(f"One of the special magic numbers for {key} is: {value}.")
    target_needle = f"One of the special magic numbers for {target_key} is: {target_value}."
    target_pos = max(0, min(num_distractors, int(num_distractors * needle_pos_frac)))
    needles.insert(target_pos, target_needle)
    context = "\n".join(needles)
    return (
        f"A special magic number is hidden within the following text. "
        f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {target_key} mentioned in the provided text?"
    )


def find_question_positions(tokenizer, ctx_ids, ctx_len, target_key):
    """Find ONLY the question part: 'What is the special magic number for mystic-thunder ...'"""
    full_text = tokenizer.decode(ctx_ids[0])
    question_str = f"What is the special magic number for {target_key} mentioned in the provided text?"
    char_pos = full_text.find(question_str)
    if char_pos < 0:
        # Fallback: last 20 tokens
        return list(range(max(0, ctx_len - 20), ctx_len))
    positions = []
    cum = ""
    for i in range(ctx_len):
        prev_len = len(cum)
        cum += tokenizer.decode(ctx_ids[0, i])
        if len(cum) > char_pos and prev_len < char_pos + len(question_str):
            positions.append(i)
    return positions


def find_token_positions(tokenizer, ctx_ids, search_str):
    ctx_len = ctx_ids.shape[1]
    full_text = tokenizer.decode(ctx_ids[0])
    char_pos = full_text.find(search_str)
    if char_pos < 0:
        return []
    positions = []
    cum = ""
    for i in range(ctx_len):
        prev_len = len(cum)
        cum += tokenizer.decode(ctx_ids[0, i])
        if len(cum) > char_pos and prev_len < char_pos + len(search_str):
            positions.append(i)
    return positions


def compute_threehop_v4(model, tokenizer, ctx_ids, target_key,
                        chunk_size=1024, vis_layers=None):
    """3-hop v4: Step1 uses ONLY question tokens.

    Returns:
        scores: [n_layers, 1, n_kv_heads, ctx_len]
        step_scores: dict with per-step scores for visualization layers
    """
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = ctx_ids.shape[1]

    # FIXED: Only question tokens, not entire user message
    q_pos = find_question_positions(tokenizer, ctx_ids, ctx_len, target_key)
    q_pos_t = torch.tensor(q_pos, dtype=torch.long, device=model.device)
    n_q = len(q_pos)
    print(f"  Step1 source: {n_q} QUESTION tokens (pos {min(q_pos)}-{max(q_pos)})")
    for p in q_pos:
        print(f"    pos {p}: '{tokenizer.decode(ctx_ids[0, p])}'")

    final_scores = torch.zeros(n_layers, 1, n_kv_heads, ctx_len, dtype=torch.float32)

    if vis_layers is None:
        vis_layers = set()
    else:
        vis_layers = set(vis_layers)

    # Store per-step scores for visualization
    step_scores = {}  # layer_idx -> {step1, step2, step3, combined} each [n_kv_heads, ctx_len]

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return
            bsz, seq_len, _ = hidden_states.shape
            if seq_len != ctx_len:
                return
            position_embeddings = kwargs.get("position_embeddings", None)

            with torch.no_grad():
                q = module.q_proj(hidden_states).view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                k = module.k_proj(hidden_states).view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                    k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
                q_grouped = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                scale = math.sqrt(head_dim)

                save_steps = layer_idx in vis_layers
                if save_steps:
                    step_scores[layer_idx] = {
                        'step1': torch.zeros(n_kv_heads, ctx_len),
                        'step2': torch.zeros(n_kv_heads, ctx_len),
                        'step3': torch.zeros(n_kv_heads, ctx_len),
                        'combined': torch.zeros(n_kv_heads, ctx_len),
                    }

                for hi in range(n_kv_heads):
                    k_h = k[0, hi]
                    q_h = q_grouped[0, hi]

                    # ═══ Step 1: ONLY question tokens outgoing (FIXED!) ═══
                    question_q = q_h[:, q_pos_t, :]  # [n_groups, n_q, d]
                    q_logits = torch.matmul(question_q, k_h.T) / scale
                    for qi in range(n_q):
                        q_logits[:, qi, q_pos[qi]+1:] = float('-inf')
                    q_attn = torch.softmax(q_logits.float(), dim=-1)
                    step1 = q_attn.amax(dim=(0, 1))  # [L]
                    del q_logits, q_attn, question_q

                    # ═══ Pass 1: Exact fan-in ═══
                    fan_in = torch.zeros(ctx_len, dtype=torch.float32, device=k_h.device)
                    for start in range(0, ctx_len, chunk_size):
                        end = min(start + chunk_size, ctx_len)
                        chunk_q = q_h[:, start:end, :]
                        logits = torch.matmul(chunk_q, k_h.T) / scale
                        for ci in range(end - start):
                            logits[:, ci, start + ci + 1:] = float('-inf')
                        chunk_attn = torch.softmax(logits.float(), dim=-1).amax(dim=0)
                        fan_in += chunk_attn.sum(dim=0)
                        del chunk_q, logits, chunk_attn

                    inv_fanin = 1.0 / (fan_in + 1e-6)
                    inv_fanin = inv_fanin / inv_fanin.sum()
                    step1_weighted = step1 * inv_fanin

                    # ═══ Pass 2: Step 2+3 ═══
                    step2 = torch.zeros(ctx_len, dtype=torch.float32, device=k_h.device)
                    step3 = torch.zeros(ctx_len, dtype=torch.float32, device=k_h.device)
                    for start in range(0, ctx_len, chunk_size):
                        end = min(start + chunk_size, ctx_len)
                        chunk_q = q_h[:, start:end, :]
                        logits = torch.matmul(chunk_q, k_h.T) / scale
                        for ci in range(end - start):
                            logits[:, ci, start + ci + 1:] = float('-inf')
                        chunk_attn = torch.softmax(logits.float(), dim=-1).amax(dim=0)
                        chunk_step2 = torch.matmul(chunk_attn, step1_weighted)
                        step2[start:end] = chunk_step2
                        step3 += torch.matmul(chunk_attn.T, chunk_step2)
                        del chunk_q, logits, chunk_attn, chunk_step2

                    def norm01(x):
                        mn, mx = x.min(), x.max()
                        return (x - mn) / (mx - mn + 1e-10)

                    combined = norm01(step1) + norm01(step2) + norm01(step3)
                    final_scores[layer_idx, 0, hi, :] = combined.cpu()

                    if save_steps:
                        step_scores[layer_idx]['step1'][hi] = step1.cpu()
                        step_scores[layer_idx]['step2'][hi] = step2.cpu()
                        step_scores[layer_idx]['step3'][hi] = step3.cpu()
                        step_scores[layer_idx]['combined'][hi] = combined.cpu()

                    del step1, step2, step3, fan_in, inv_fanin, step1_weighted
                del q, k, q_grouped

        return hook_fn

    hooks = []
    for layer in model.model.layers:
        h = layer.self_attn.register_forward_hook(
            make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True)
        hooks.append(h)

    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=cache)
    for h in hooks:
        h.remove()
    del cache
    torch.cuda.empty_cache()

    return final_scores, step_scores


# ─── Compression & Generation (same as before) ───

def apply_global_compression(model, score_val, cr, ctx_len):
    n_layers, bsz, n_kv_heads, seq_len = score_val.shape
    total = n_layers * bsz * n_kv_heads * seq_len
    n_pruned = int(total * cr)
    if n_pruned <= 0:
        for layer in model.model.layers:
            layer.self_attn.masked_key_indices = None
        return
    flat_scores = score_val.reshape(-1)
    _, prune_idx = torch.topk(-flat_scores, min(n_pruned, flat_scores.numel()))
    layer_size = bsz * n_kv_heads * seq_len
    for layer in model.model.layers:
        li = int(layer.self_attn.layer_idx)
        layer_start = li * layer_size
        layer_end = layer_start + layer_size
        layer_mask = (prune_idx >= layer_start) & (prune_idx < layer_end)
        layer_indices = prune_idx[layer_mask] - layer_start
        if len(layer_indices) == 0:
            layer.self_attn.masked_key_indices = None
            continue
        bi = layer_indices // (n_kv_heads * seq_len)
        remainder = layer_indices % (n_kv_heads * seq_len)
        hi = remainder // seq_len
        si = remainder % seq_len
        layer.self_attn.masked_key_indices = (bi, hi, si)


def clear_compression(model):
    for layer in model.model.layers:
        layer.self_attn.masked_key_indices = None


def generate_with_scores(model, tokenizer, ctx_ids, q_ids, score_val, cr, max_new=60):
    ctx_len = ctx_ids.shape[1]
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=cache)
    if cr > 0:
        apply_global_compression(model, score_val, cr, ctx_len)
    q_len = q_ids.shape[1]
    pos = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=q_ids, past_key_values=cache, position_ids=pos, num_logits_to_keep=1)
    gen = [out.logits[0, -1].argmax()]
    eos = model.generation_config.eos_token_id
    if not isinstance(eos, list):
        eos = [eos]
    cp = ctx_len + q_len
    for i in range(max_new - 1):
        with torch.no_grad():
            out = model(input_ids=gen[-1].unsqueeze(0).unsqueeze(0),
                       past_key_values=cache,
                       position_ids=torch.tensor([[cp + i]], device=model.device))
        nxt = out.logits[0, -1].argmax()
        gen.append(nxt)
        if nxt.item() in eos:
            break
    text = tokenizer.decode(torch.stack(gen), skip_special_tokens=True)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    clear_compression(model)
    del cache
    return text


# ─── Visualization ───

def visualize_3hop(tokenizer, ctx_ids, step_scores, layer_idx, head_idx,
                   target_key, target_value, out_dir, n_dist):
    """Visualize the 3-hop chain for a specific layer and head."""
    ctx_len = ctx_ids.shape[1]
    ss = step_scores[layer_idx]
    s1 = ss['step1'][head_idx].numpy()
    s2 = ss['step2'][head_idx].numpy()
    s3 = ss['step3'][head_idx].numpy()
    combined = ss['combined'][head_idx].numpy()

    # Find key positions
    needle_str = f"One of the special magic numbers for {target_key} is: {target_value}."
    needle_pos = find_token_positions(tokenizer, ctx_ids, needle_str)
    key_pos = set(find_token_positions(tokenizer, ctx_ids, target_key))
    value_pos = set(find_token_positions(tokenizer, ctx_ids, target_value))
    question_str = f"What is the special magic number for {target_key}"
    question_pos = set(find_question_positions(tokenizer, ctx_ids, ctx_len, target_key))

    # Token colors
    colors = []
    for i in range(ctx_len):
        if i in key_pos and (needle_pos and needle_pos[0] <= i <= needle_pos[-1]):
            colors.append('#FF4444')  # needle key = red
        elif i in value_pos:
            colors.append('#4444FF')  # needle value = blue
        elif needle_pos and needle_pos[0] <= i <= needle_pos[-1]:
            colors.append('#FF88FF')  # needle other = pink
        elif i in question_pos:
            colors.append('#44AA44')  # question = green
        elif i < 4:
            colors.append('#FFaa00')  # sink = orange
        else:
            colors.append('#CCCCCC')  # other = gray

    fig, axes = plt.subplots(4, 1, figsize=(24, 16), sharex=True)
    titles = [
        f'Step 1: Question → What it attends to (OUTGOING)',
        f'Step 2: Who attends to Step1 targets (INCOMING, inv fan-in weighted)',
        f'Step 3: Step2 tokens → What they attend to (OUTGOING)',
        f'Combined: norm(S1) + norm(S2) + norm(S3)',
    ]
    data = [s1, s2, s3, combined]

    for ax_idx, (ax, title, d) in enumerate(zip(axes, titles, data)):
        ax.bar(range(ctx_len), d, color=colors, width=1.0, edgecolor='none')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(title, fontsize=12)

        # Highlight needle region
        if needle_pos:
            ax.axvspan(needle_pos[0]-0.5, needle_pos[-1]+0.5, alpha=0.1, color='red')

        # Mark top-10 tokens with arrows
        topk_idx = np.argsort(d)[-10:][::-1]
        for rank, idx in enumerate(topk_idx):
            tok = tokenizer.decode(ctx_ids[0, idx]).strip()[:10]
            if d[idx] > 0:
                label = ""
                if idx in key_pos:
                    label = "[K]"
                elif idx in value_pos:
                    label = "[V]"
                elif idx in question_pos:
                    label = "[Q]"
                ax.annotate(f'#{rank+1}:{tok}{label}',
                           xy=(idx, d[idx]), fontsize=6,
                           ha='center', va='bottom',
                           rotation=45)

    axes[-1].set_xlabel('Token Position', fontsize=12)

    patches = [
        mpatches.Patch(color='#FF4444', label='Needle Key'),
        mpatches.Patch(color='#4444FF', label='Needle Value'),
        mpatches.Patch(color='#FF88FF', label='Needle Other'),
        mpatches.Patch(color='#44AA44', label='Question'),
        mpatches.Patch(color='#FFaa00', label='Sink'),
        mpatches.Patch(color='#CCCCCC', label='Other/Distractor'),
    ]
    fig.legend(handles=patches, loc='upper right', fontsize=9, ncol=2)
    plt.suptitle(f'3-Hop Chain: Layer {layer_idx}, Head {head_idx} ({n_dist}d)',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f'3hop_chain_L{layer_idx}_H{head_idx}_{n_dist}d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def visualize_3hop_zoom(tokenizer, ctx_ids, step_scores, layer_idx, head_idx,
                        target_key, target_value, out_dir, n_dist):
    """Zoomed visualization around needle region."""
    ctx_len = ctx_ids.shape[1]
    ss = step_scores[layer_idx]
    s1 = ss['step1'][head_idx].numpy()
    s2 = ss['step2'][head_idx].numpy()
    s3 = ss['step3'][head_idx].numpy()
    combined = ss['combined'][head_idx].numpy()

    needle_str = f"One of the special magic numbers for {target_key} is: {target_value}."
    needle_pos = find_token_positions(tokenizer, ctx_ids, needle_str)
    key_pos = set(find_token_positions(tokenizer, ctx_ids, target_key))
    value_pos = set(find_token_positions(tokenizer, ctx_ids, target_value))

    if not needle_pos:
        return

    # Zoom: needle ± 40 tokens
    zoom_start = max(0, needle_pos[0] - 40)
    zoom_end = min(ctx_len, needle_pos[-1] + 41)
    zoom_range = list(range(zoom_start, zoom_end))

    colors = []
    labels = []
    for i in zoom_range:
        tok = tokenizer.decode(ctx_ids[0, i]).strip()[:8]
        if i in key_pos and needle_pos[0] <= i <= needle_pos[-1]:
            colors.append('#FF4444')
            labels.append(f'{tok}[K]')
        elif i in value_pos:
            colors.append('#4444FF')
            labels.append(f'{tok}[V]')
        elif needle_pos[0] <= i <= needle_pos[-1]:
            colors.append('#FF88FF')
            labels.append(f'{tok}[N]')
        else:
            colors.append('#CCCCCC')
            labels.append(tok)

    fig, axes = plt.subplots(4, 1, figsize=(20, 14), sharex=True)
    titles = ['Step 1: Question OUTGOING', 'Step 2: INCOMING (inv fan-in)',
              'Step 3: Step2 OUTGOING', 'Combined']
    data = [s1, s2, s3, combined]

    for ax, title, d in zip(axes, titles, data):
        zoom_d = [d[i] for i in zoom_range]
        ax.bar(range(len(zoom_range)), zoom_d, color=colors, width=0.8)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(title, fontsize=12)

        # Highlight needle
        needle_local = [zoom_range.index(p) for p in needle_pos if p in zoom_range]
        if needle_local:
            ax.axvspan(min(needle_local)-0.5, max(needle_local)+0.5, alpha=0.1, color='red')

        # Value labels on key/value bars
        for j, pos in enumerate(zoom_range):
            if pos in key_pos or pos in value_pos:
                if zoom_d[j] > 0:
                    ax.text(j, zoom_d[j], f'{zoom_d[j]:.3f}', ha='center', va='bottom',
                           fontsize=6, fontweight='bold')

    # X labels
    axes[-1].set_xticks(range(len(zoom_range)))
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=6, ha='center')
    axes[-1].set_xlabel('Token', fontsize=12)

    patches = [mpatches.Patch(color='#FF4444', label='Key'),
               mpatches.Patch(color='#4444FF', label='Value'),
               mpatches.Patch(color='#FF88FF', label='Needle other')]
    fig.legend(handles=patches, loc='upper right', fontsize=10)
    plt.suptitle(f'3-Hop Chain (Zoomed): L{layer_idx} H{head_idx} ({n_dist}d)', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f'3hop_zoom_L{layer_idx}_H{head_idx}_{n_dist}d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    model_path = "Qwen/Qwen3-8B"
    target_key = "mystic-thunder"
    target_value = "7156842"
    out_dir = "/home/zichuanfu2/SparseKV/logs/threehop_vis"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    print(f"Model loaded. {n_layers} layers, {n_kv_heads} KV heads")

    # Layers to visualize
    vis_layers = [4, 8, 12, 16, 20, 24, 28, 32]

    for n_dist in [30, 100]:
        print(f"\n{'=' * 80}")
        print(f"  {n_dist} distractors, needle at 50%")
        print(f"{'=' * 80}")

        prompt = build_niah_prompt(n_dist, target_key, target_value, 0.5, seed=50)
        separator = "#" * (len(prompt) + 10)
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + separator}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        ctx_text, q_suffix = full_text.split(separator)
        ctx_ids = tokenizer.encode(ctx_text, return_tensors="pt",
                                   add_special_tokens=False).to(model.device)
        q_ids = tokenizer.encode(q_suffix, return_tensors="pt",
                                 add_special_tokens=False).to(model.device)
        ctx_len = ctx_ids.shape[1]
        print(f"  Context: {ctx_len} tokens, Question: {q_ids.shape[1]} tokens")

        # Compute v4 scores (question-only step1)
        print(f"\n  Computing v4 (question-only) 3-hop scores...")
        scores_v4, step_scores = compute_threehop_v4(
            model, tokenizer, ctx_ids, target_key,
            vis_layers=vis_layers
        )
        torch.cuda.empty_cache()

        # Score analysis
        needle_str = f"for {target_key} is: {target_value}."
        needle_pos = find_token_positions(tokenizer, ctx_ids, needle_str)
        key_pos = find_token_positions(tokenizer, ctx_ids, target_key)
        value_pos = find_token_positions(tokenizer, ctx_ids, target_value)

        if needle_pos:
            print(f"\n  Score comparison (needle at pos {needle_pos[0]}-{needle_pos[-1]}):")
            print(f"  {'Layer':>6} {'Key':>10} {'Value':>10} {'Other':>10} {'Key/Oth':>8} {'Val/Oth':>8}")
            for li in range(0, n_layers, 4):
                layer_scores = scores_v4[li, 0].mean(dim=0)
                key_score = layer_scores[torch.tensor(key_pos)].mean().item() if key_pos else 0
                val_score = layer_scores[torch.tensor(value_pos)].mean().item() if value_pos else 0
                mask = torch.ones(ctx_len, dtype=torch.bool)
                mask[:4] = False
                for p in needle_pos:
                    mask[p] = False
                other_score = layer_scores[mask].mean().item()
                k_ratio = key_score / max(other_score, 1e-10)
                v_ratio = val_score / max(other_score, 1e-10)
                print(f"  L{li:>4}: {key_score:>10.4f} {val_score:>10.4f} {other_score:>10.4f} {k_ratio:>7.2f}x {v_ratio:>7.2f}x")

        # Generation test
        print(f"\n  Generation results (v4 question-only):")
        for cr in [0.5, 0.7, 0.9]:
            if cr == 0.5:
                full = generate_with_scores(model, tokenizer, ctx_ids, q_ids,
                                           torch.ones_like(scores_v4), 0.0)
                ok_full = target_value in full
                print(f"    Full KV:            {'OK' if ok_full else 'FAIL'}  {full[:80]}")
                torch.cuda.empty_cache()

            gen = generate_with_scores(model, tokenizer, ctx_ids, q_ids,
                                      scores_v4, cr)
            ok = target_value in gen
            print(f"    v4-global cr={cr}:   {'OK' if ok else 'FAIL'}  {gen[:80]}")
            torch.cuda.empty_cache()

        # Visualize 3-hop chain for key layers/heads
        print(f"\n  Generating visualizations...")
        for li in vis_layers:
            if li >= n_layers or li not in step_scores:
                continue
            # Find head with highest value token score
            val_t = torch.tensor(value_pos) if value_pos else torch.tensor([0])
            val_scores = step_scores[li]['combined'][:, val_t].mean(dim=1)  # [n_kv_heads]
            best_head = val_scores.argmax().item()

            visualize_3hop(tokenizer, ctx_ids, step_scores, li, best_head,
                          target_key, target_value, out_dir, n_dist)
            visualize_3hop_zoom(tokenizer, ctx_ids, step_scores, li, best_head,
                               target_key, target_value, out_dir, n_dist)

        # Also visualize head 0 at L20 and L32 for comparison
        for li in [20, 32]:
            if li < n_layers and li in step_scores:
                for hi in range(min(4, n_kv_heads)):
                    visualize_3hop_zoom(tokenizer, ctx_ids, step_scores, li, hi,
                                       target_key, target_value, out_dir, n_dist)

        del scores_v4, step_scores
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nAll visualizations saved to: {out_dir}")
    print("DONE")


if __name__ == "__main__":
    main()
