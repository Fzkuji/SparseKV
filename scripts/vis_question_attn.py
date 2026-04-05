#!/usr/bin/env python3
"""Clear visualization: what does each question token attend to?

Produces:
1. Heatmap: question tokens (y) × all positions (x) — raw Step1 attention
2. Zoomed heatmap around needle region
3. Top-K attended positions per question token (text table)
"""

import torch
import random
import math
import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
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


def find_question_positions(tokenizer, ctx_ids, ctx_len, target_key):
    full_text = tokenizer.decode(ctx_ids[0])
    question_str = f"What is the special magic number for {target_key} mentioned in the provided text?"
    char_pos = full_text.find(question_str)
    if char_pos < 0:
        return list(range(max(0, ctx_len - 20), ctx_len))
    positions = []
    cum = ""
    for i in range(ctx_len):
        prev_len = len(cum)
        cum += tokenizer.decode(ctx_ids[0, i])
        if len(cum) > char_pos and prev_len < char_pos + len(question_str):
            positions.append(i)
    return positions


def main():
    model_path = "Qwen/Qwen3-8B"
    target_key = "mystic-thunder"
    target_value = "7156842"
    out_dir = "/home/zichuanfu2/SparseKV/logs/question_attn_vis"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    print(f"Model loaded. {n_layers} layers, {n_kv_heads} KV heads, {n_groups} groups")

    n_dist = 30
    prompt = build_niah_prompt(n_dist, target_key, target_value, 0.5, seed=50)
    separator = "#" * (len(prompt) + 10)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt + separator}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    ctx_text, q_suffix = full_text.split(separator)
    ctx_ids = tokenizer.encode(ctx_text, return_tensors="pt",
                               add_special_tokens=False).to(model.device)
    ctx_len = ctx_ids.shape[1]

    # Find positions
    q_pos = find_question_positions(tokenizer, ctx_ids, ctx_len, target_key)
    key_pos = find_token_positions(tokenizer, ctx_ids, target_key)
    value_pos = find_token_positions(tokenizer, ctx_ids, target_value)
    needle_str = f"One of the special magic numbers for {target_key} is: {target_value}."
    needle_pos = find_token_positions(tokenizer, ctx_ids, needle_str)

    q_pos_t = torch.tensor(q_pos, dtype=torch.long, device=model.device)
    n_q = len(q_pos)

    print(f"Context: {ctx_len} tokens")
    print(f"Question: {n_q} tokens (pos {q_pos[0]}-{q_pos[-1]})")
    print(f"Needle: pos {needle_pos[0]}-{needle_pos[-1]}")
    print(f"Key: {key_pos}, Value: {value_pos}")

    q_labels = []
    for p in q_pos:
        tok = tokenizer.decode(ctx_ids[0, p]).replace('\n', '\\n')
        q_labels.append(f"'{tok}' (p{p})")

    # Capture raw attention: question → all, per layer per KV head (max over groups)
    # Shape: [n_layers, n_kv_heads, n_q, ctx_len]
    vis_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
    vis_layers = [l for l in vis_layers if l < n_layers]

    raw_attn = {}  # layer -> [n_kv_heads, n_q, ctx_len]

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            if layer_idx not in vis_layers:
                return
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

                layer_attn = torch.zeros(n_kv_heads, n_q, ctx_len, dtype=torch.float32)

                for hi in range(n_kv_heads):
                    k_h = k[0, hi]  # [L, d]
                    q_h = q_grouped[0, hi]  # [n_groups, L, d]

                    # Question tokens' attention
                    qq = q_h[:, q_pos_t, :]  # [n_groups, n_q, d]
                    logits = torch.matmul(qq, k_h.T) / scale  # [n_groups, n_q, L]
                    for qi in range(n_q):
                        logits[:, qi, q_pos[qi]+1:] = float('-inf')
                    attn = torch.softmax(logits.float(), dim=-1)  # [n_groups, n_q, L]
                    # Max over groups
                    attn_max = attn.amax(dim=0)  # [n_q, L]
                    layer_attn[hi] = attn_max.cpu()
                    del qq, logits, attn

                raw_attn[layer_idx] = layer_attn
                del q, k, q_grouped

        return hook_fn

    hooks = []
    for layer in model.model.layers:
        h = layer.self_attn.register_forward_hook(
            make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True)
        hooks.append(h)

    print("Running prefill to capture attention...")
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=cache)
    for h in hooks:
        h.remove()
    del cache
    torch.cuda.empty_cache()
    print("Done capturing attention.")

    # Token type coloring for x-axis
    token_type = np.zeros(ctx_len, dtype=int)
    key_pos_set = set(key_pos)
    value_pos_set = set(value_pos)
    needle_pos_set = set(needle_pos)
    question_pos_set = set(q_pos)
    for p in range(min(4, ctx_len)):
        token_type[p] = 4  # sink
    for p in needle_pos:
        token_type[p] = 5  # needle other
    for p in key_pos:
        token_type[p] = 1  # key
    for p in value_pos:
        token_type[p] = 2  # value
    for p in q_pos:
        token_type[p] = 3  # question

    type_colors = {0: '#CCCCCC', 1: '#FF4444', 2: '#4444FF', 3: '#44AA44', 4: '#FFaa00', 5: '#FF88FF'}

    # ═══════════════════════════════════════════════════════
    # For each layer: heatmap of question attention (max over heads)
    # ═══════════════════════════════════════════════════════
    for li in vis_layers:
        if li not in raw_attn:
            continue
        attn = raw_attn[li]  # [n_kv_heads, n_q, ctx_len]
        # Max over heads
        attn_max_h = attn.amax(dim=0).numpy()  # [n_q, ctx_len]

        # ── Figure 1: Full heatmap ──
        fig, ax = plt.subplots(figsize=(24, 6))
        im = ax.imshow(attn_max_h, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_yticks(range(n_q))
        ax.set_yticklabels(q_labels, fontsize=7)
        ax.set_xlabel('Context Position', fontsize=12)
        ax.set_ylabel('Question Token', fontsize=12)
        ax.set_title(f'Layer {li}: Question Token Attention (max over {n_kv_heads} KV heads)', fontsize=14)

        # Mark needle region
        if needle_pos:
            ax.axvline(x=needle_pos[0]-0.5, color='lime', linewidth=1.5, linestyle='--')
            ax.axvline(x=needle_pos[-1]+0.5, color='lime', linewidth=1.5, linestyle='--')
            ax.text(needle_pos[0], -0.8, 'needle', color='lime', fontsize=8, ha='left')

        # Color bar for x-axis
        for p in range(ctx_len):
            if token_type[p] == 1:
                ax.axvline(x=p, color='red', alpha=0.3, linewidth=0.5)
            elif token_type[p] == 2:
                ax.axvline(x=p, color='blue', alpha=0.3, linewidth=0.5)

        plt.colorbar(im, ax=ax, label='Attention weight', shrink=0.8)
        plt.tight_layout()
        path = os.path.join(out_dir, f'question_attn_full_L{li}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

        # ── Figure 2: Zoomed around needle ──
        if needle_pos:
            zoom_start = max(0, needle_pos[0] - 30)
            zoom_end = min(ctx_len, needle_pos[-1] + 31)
            zoom_data = attn_max_h[:, zoom_start:zoom_end]

            fig, ax = plt.subplots(figsize=(16, 6))
            im = ax.imshow(zoom_data, aspect='auto', cmap='hot', interpolation='nearest')
            ax.set_yticks(range(n_q))
            ax.set_yticklabels(q_labels, fontsize=7)

            # X-axis: token labels
            zoom_labels = []
            for p in range(zoom_start, zoom_end):
                tok = tokenizer.decode(ctx_ids[0, p]).strip()[:8]
                marker = ""
                if p in key_pos_set:
                    marker = "[K]"
                elif p in value_pos_set:
                    marker = "[V]"
                elif p in needle_pos_set:
                    marker = "[N]"
                zoom_labels.append(f"{tok}{marker}")
            ax.set_xticks(range(len(zoom_labels)))
            ax.set_xticklabels(zoom_labels, rotation=90, fontsize=6)
            ax.set_title(f'Layer {li}: Question → Needle Region (max over heads)', fontsize=13)

            # Highlight key and value columns
            for i, p in enumerate(range(zoom_start, zoom_end)):
                if p in key_pos_set:
                    ax.axvline(x=i, color='red', alpha=0.4, linewidth=2)
                elif p in value_pos_set:
                    ax.axvline(x=i, color='blue', alpha=0.4, linewidth=2)

            plt.colorbar(im, ax=ax, label='Attention weight', shrink=0.8)
            plt.tight_layout()
            path = os.path.join(out_dir, f'question_attn_zoom_L{li}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {path}")

        # ── Figure 3: Per-head heatmap for key layers ──
        if li in [20, 24, 28, 32]:
            fig, axes = plt.subplots(2, 4, figsize=(28, 8))
            for hi in range(n_kv_heads):
                ax = axes[hi // 4][hi % 4]
                head_attn = attn[hi].numpy()  # [n_q, ctx_len]
                if needle_pos:
                    zoom_data = head_attn[:, zoom_start:zoom_end]
                    im = ax.imshow(zoom_data, aspect='auto', cmap='hot', interpolation='nearest')
                    # Highlight
                    for i, p in enumerate(range(zoom_start, zoom_end)):
                        if p in key_pos_set:
                            ax.axvline(x=i, color='red', alpha=0.4, linewidth=1)
                        elif p in value_pos_set:
                            ax.axvline(x=i, color='blue', alpha=0.4, linewidth=1)
                    ax.set_title(f'H{hi}', fontsize=10)
                    if hi % 4 == 0:
                        ax.set_yticks(range(n_q))
                        ax.set_yticklabels(q_labels, fontsize=5)
                    else:
                        ax.set_yticks([])
                    if hi >= 4:
                        ax.set_xticks(range(0, len(zoom_labels), 3))
                        ax.set_xticklabels(zoom_labels[::3], rotation=90, fontsize=5)
                    else:
                        ax.set_xticks([])

            plt.suptitle(f'Layer {li}: Per-Head Question → Needle (red=key, blue=value)', fontsize=14)
            plt.tight_layout()
            path = os.path.join(out_dir, f'question_attn_perhead_L{li}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {path}")

    # ═══════════════════════════════════════════════════════
    # Text summary: top-10 attended positions per question token
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("Top-10 attended positions per question token (L20, L24, L28, L32)")
    print(f"{'=' * 80}")

    for li in [20, 24, 28, 32]:
        if li not in raw_attn:
            continue
        attn = raw_attn[li]
        attn_max_h = attn.amax(dim=0).numpy()  # [n_q, ctx_len]

        print(f"\n  Layer {li}:")
        for qi in range(n_q):
            tok = tokenizer.decode(ctx_ids[0, q_pos[qi]]).strip()
            topk_idx = np.argsort(attn_max_h[qi])[-10:][::-1]
            entries = []
            for idx in topk_idx:
                target_tok = tokenizer.decode(ctx_ids[0, idx]).strip()[:12]
                score = attn_max_h[qi, idx]
                marker = ""
                if idx in key_pos_set:
                    marker = "[KEY]"
                elif idx in value_pos_set:
                    marker = "[VAL]"
                elif idx in needle_pos_set:
                    marker = "[NDL]"
                elif idx in question_pos_set:
                    marker = "[Q]"
                entries.append(f"p{idx}:'{target_tok}'{marker}={score:.4f}")
            print(f"    Q['{tok}']: {', '.join(entries[:5])}")
            print(f"               {', '.join(entries[5:])}")

    # ═══════════════════════════════════════════════════════
    # Summary: total attention to each category per layer
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("Total question attention to each category (sum over question tokens, max over heads)")
    print(f"{'=' * 80}")
    print(f"  {'Layer':>6} {'→Key':>10} {'→Value':>10} {'→NdlOther':>10} {'→Sink':>10} {'→Question':>10} {'→Distract':>10}")

    for li in vis_layers:
        if li not in raw_attn:
            continue
        attn = raw_attn[li]
        attn_max_h = attn.amax(dim=0).numpy()  # [n_q, ctx_len]
        # Sum over question tokens
        total = attn_max_h.sum(axis=0)  # [ctx_len]

        to_key = sum(total[p] for p in key_pos)
        to_val = sum(total[p] for p in value_pos)
        to_ndl = sum(total[p] for p in needle_pos if p not in key_pos_set and p not in value_pos_set)
        to_sink = sum(total[p] for p in range(min(4, ctx_len)))
        to_q = sum(total[p] for p in q_pos)
        to_dist = total.sum() - to_key - to_val - to_ndl - to_sink - to_q
        print(f"  L{li:>4}: {to_key:>10.4f} {to_val:>10.4f} {to_ndl:>10.4f} {to_sink:>10.4f} {to_q:>10.4f} {to_dist:>10.4f}")

    print(f"\nAll figures saved to: {out_dir}")
    print("DONE")


if __name__ == "__main__":
    main()
