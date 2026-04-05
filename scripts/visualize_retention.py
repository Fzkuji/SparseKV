#!/usr/bin/env python3
"""Visualize token retention at different compression ratios.

Produces a visual map showing which tokens are kept/evicted at cr=0.7 and cr=0.9,
highlighting needle components. Also generates per-layer heatmaps.

Output: saves figures to /home/zichuanfu2/SparseKV/logs/retention_vis/
"""

import torch
import random
import math
import re
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
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


def find_user_input_positions(tokenizer, context_ids, ctx_len):
    full_text = tokenizer.decode(context_ids[0])
    user_marker = "user\n"
    user_start_char = full_text.find(user_marker)
    if user_start_char < 0:
        return list(range(max(0, int(ctx_len * 0.8)), ctx_len))
    user_start_char += len(user_marker)
    im_end = "<|im_end|>"
    user_end_char = full_text.find(im_end, user_start_char)
    if user_end_char < 0:
        user_end_char = len(full_text)
    positions = []
    cum_text = ""
    for i in range(ctx_len):
        tok_text = tokenizer.decode(context_ids[0, i])
        start_char = len(cum_text)
        cum_text += tok_text
        end_char = len(cum_text)
        if end_char > user_start_char and start_char < user_end_char:
            positions.append(i)
    return positions if positions else list(range(max(0, ctx_len - 50), ctx_len))


def compute_scores(model, tokenizer, ctx_ids, n_layers, n_kv_heads, n_q_heads, head_dim, chunk_size=1024):
    """Compute 3-hop scores. Returns [n_layers, 1, n_kv_heads, ctx_len]."""
    n_groups = n_q_heads // n_kv_heads
    ctx_len = ctx_ids.shape[1]
    user_pos = find_user_input_positions(tokenizer, ctx_ids, ctx_len)
    user_pos_t = torch.tensor(user_pos, dtype=torch.long, device=model.device)

    final_scores = torch.zeros(n_layers, 1, n_kv_heads, ctx_len, dtype=torch.float32)

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

                for hi in range(n_kv_heads):
                    k_h = k[0, hi]
                    q_h = q_grouped[0, hi]
                    user_q = q_h[:, user_pos_t, :]
                    user_logits = torch.matmul(user_q, k_h.T) / scale
                    for ui in range(len(user_pos)):
                        user_logits[:, ui, user_pos[ui]+1:] = float('-inf')
                    user_attn = torch.softmax(user_logits.float(), dim=-1)
                    step1 = user_attn.amax(dim=(0, 1))
                    del user_logits, user_attn, user_q

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
    return final_scores


def main():
    model_path = "Qwen/Qwen3-8B"
    target_key = "mystic-thunder"
    target_value = "7156842"
    out_dir = "/home/zichuanfu2/SparseKV/logs/retention_vis"
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
    head_dim = model.config.hidden_size // n_q_heads
    print(f"Model loaded. {n_layers} layers, {n_kv_heads} KV heads")

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
    key_pos = find_token_positions(tokenizer, ctx_ids, target_key)
    value_pos = find_token_positions(tokenizer, ctx_ids, target_value)
    needle_str = f"One of the special magic numbers for {target_key} is: {target_value}."
    needle_pos = find_token_positions(tokenizer, ctx_ids, needle_str)
    question_str = f"What is the special magic number for {target_key}"
    question_pos = find_token_positions(tokenizer, ctx_ids, question_str)

    # Build token labels
    token_labels = []
    for i in range(ctx_len):
        tok = tokenizer.decode(ctx_ids[0, i]).replace('\n', '\\n')
        token_labels.append(tok)

    # Token type for coloring
    token_type = np.zeros(ctx_len, dtype=int)  # 0=normal, 1=key, 2=value, 3=question, 4=sink, 5=needle_other
    for p in range(min(4, ctx_len)):
        token_type[p] = 4  # sink
    for p in needle_pos:
        token_type[p] = 5  # needle other (prefix, sep, period)
    for p in key_pos:
        token_type[p] = 1
    for p in value_pos:
        token_type[p] = 2
    for p in question_pos:
        token_type[p] = 3

    print(f"Context: {ctx_len} tokens")
    print(f"Computing 3-hop scores...")
    scores = compute_scores(model, tokenizer, ctx_ids, n_layers, n_kv_heads, n_q_heads, head_dim)
    print("Scores computed.")

    # ═══════════════════════════════════════════════════════
    # Figure 1: Token-level retention map at cr=0.7 vs cr=0.9
    # Show which tokens are kept (averaged over layers and heads)
    # ═══════════════════════════════════════════════════════
    print("Generating Figure 1: Token retention overview...")

    # For global allocation: compute per-token retention fraction across all (layer, head) pairs
    total_slots = n_layers * n_kv_heads
    for cr in [0.7, 0.9]:
        n_pruned = int(n_layers * 1 * n_kv_heads * ctx_len * cr)
        flat = scores.reshape(-1)
        sorted_s, _ = flat.sort()
        threshold = sorted_s[min(n_pruned, len(sorted_s)-1)].item()

        # Per-token: fraction of (layer, head) pairs where it's kept
        kept_mask = (scores[:, 0, :, :] >= threshold)  # [n_layers, n_kv_heads, ctx_len]
        retention_frac = kept_mask.float().mean(dim=(0, 1)).numpy()  # [ctx_len]

        # Create figure: horizontal bar for each token position
        fig, axes = plt.subplots(2, 1, figsize=(24, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Top: retention fraction as bar chart
        ax = axes[0]
        colors = []
        type_colors = {0: '#CCCCCC', 1: '#FF4444', 2: '#4444FF', 3: '#44AA44', 4: '#FFaa00', 5: '#FF88FF'}
        type_names = {0: 'Other', 1: 'Key (mystic-thunder)', 2: 'Value (7156842)',
                      3: 'Question', 4: 'Sink', 5: 'Needle (prefix/sep/period)'}
        for i in range(ctx_len):
            colors.append(type_colors[token_type[i]])

        ax.bar(range(ctx_len), retention_frac, color=colors, width=1.0, edgecolor='none')
        ax.set_ylabel('Retention fraction\n(across layers×heads)', fontsize=12)
        ax.set_title(f'Token Retention at CR={cr} (Global Allocation, 30 distractors)\n'
                     f'Total {ctx_len} tokens, keep {100*(1-cr):.0f}%', fontsize=14)
        ax.set_xlim(-1, ctx_len)
        ax.axhline(y=1-cr, color='red', linestyle='--', alpha=0.5, label=f'Expected {100*(1-cr):.0f}%')

        # Add needle region highlight
        if needle_pos:
            ax.axvspan(needle_pos[0]-0.5, needle_pos[-1]+0.5, alpha=0.1, color='red', label='Needle region')
        if question_pos:
            ax.axvspan(question_pos[0]-0.5, question_pos[-1]+0.5, alpha=0.1, color='green', label='Question region')

        # Legend
        patches = [mpatches.Patch(color=type_colors[t], label=type_names[t]) for t in sorted(type_colors.keys())]
        ax.legend(handles=patches, loc='upper right', fontsize=9, ncol=2)
        ax.set_ylim(0, 1.05)

        # Bottom: zoomed view of needle region (±30 tokens)
        ax2 = axes[1]
        zoom_start = max(0, needle_pos[0] - 30)
        zoom_end = min(ctx_len, needle_pos[-1] + 31)
        zoom_range = range(zoom_start, zoom_end)
        zoom_colors = [type_colors[token_type[i]] for i in zoom_range]
        zoom_retention = [retention_frac[i] for i in zoom_range]

        bars = ax2.bar(range(len(zoom_range)), zoom_retention, color=zoom_colors, width=1.0, edgecolor='none')
        ax2.set_ylabel('Retention', fontsize=11)
        ax2.set_title(f'Zoomed: Needle region (pos {zoom_start}-{zoom_end})', fontsize=12)
        ax2.axhline(y=1-cr, color='red', linestyle='--', alpha=0.5)

        # Add token text labels
        xtick_pos = []
        xtick_labels = []
        for i, pos in enumerate(zoom_range):
            if token_type[pos] in [1, 2, 5]:  # Only label needle tokens
                xtick_pos.append(i)
                xtick_labels.append(f"{pos}:{token_labels[pos][:8]}")
        ax2.set_xticks(xtick_pos)
        ax2.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_ylim(0, 1.05)

        plt.tight_layout()
        path = os.path.join(out_dir, f'token_retention_cr{int(cr*100)}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # ═══════════════════════════════════════════════════════
    # Figure 2: Side-by-side CR=0.7 vs CR=0.9, needle region only
    # ═══════════════════════════════════════════════════════
    print("Generating Figure 2: CR=0.7 vs CR=0.9 comparison...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    zoom_start = max(0, needle_pos[0] - 15)
    zoom_end = min(ctx_len, needle_pos[-1] + 16)
    zoom_range = list(range(zoom_start, zoom_end))

    for ax_idx, cr in enumerate([0.7, 0.9]):
        ax = axes[ax_idx]
        n_pruned = int(n_layers * 1 * n_kv_heads * ctx_len * cr)
        flat = scores.reshape(-1)
        sorted_s, _ = flat.sort()
        threshold = sorted_s[min(n_pruned, len(sorted_s)-1)].item()
        kept_mask = (scores[:, 0, :, :] >= threshold)
        retention_frac = kept_mask.float().mean(dim=(0, 1)).numpy()

        zoom_colors = [type_colors[token_type[i]] for i in zoom_range]
        zoom_retention = [retention_frac[i] for i in zoom_range]

        bars = ax.bar(range(len(zoom_range)), zoom_retention, color=zoom_colors, width=0.8)
        ax.axhline(y=1-cr, color='red', linestyle='--', alpha=0.5, label=f'Expected {100*(1-cr):.0f}%')
        ax.set_ylabel('Retention fraction', fontsize=11)
        ax.set_title(f'CR={cr} — keep {100*(1-cr):.0f}% (threshold={threshold:.3f})', fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

        # Add value labels on bars for needle tokens
        for i, pos in enumerate(zoom_range):
            if token_type[pos] in [1, 2]:
                ax.text(i, zoom_retention[i] + 0.02, f'{zoom_retention[i]:.0%}',
                       ha='center', va='bottom', fontsize=7, fontweight='bold')

    # X-axis labels
    xtick_pos = list(range(len(zoom_range)))
    xtick_labels = []
    for pos in zoom_range:
        tok = token_labels[pos][:10].strip()
        typ = ""
        if pos in set(key_pos): typ = "[K]"
        elif pos in set(value_pos): typ = "[V]"
        xtick_labels.append(f"{tok}\n{typ}")
    axes[1].set_xticks(xtick_pos)
    axes[1].set_xticklabels(xtick_labels, fontsize=7, ha='center')

    patches = [mpatches.Patch(color=type_colors[1], label='Key'),
               mpatches.Patch(color=type_colors[2], label='Value'),
               mpatches.Patch(color=type_colors[5], label='Needle other')]
    fig.legend(handles=patches, loc='upper right', fontsize=10)
    plt.suptitle('Needle Token Retention: CR=0.7 vs CR=0.9', fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'needle_cr07_vs_cr09.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ═══════════════════════════════════════════════════════
    # Figure 3: Per-layer heatmap of needle token retention
    # ═══════════════════════════════════════════════════════
    print("Generating Figure 3: Per-layer retention heatmap...")

    for cr in [0.7, 0.9]:
        n_pruned = int(n_layers * 1 * n_kv_heads * ctx_len * cr)
        flat = scores.reshape(-1)
        sorted_s, _ = flat.sort()
        threshold = sorted_s[min(n_pruned, len(sorted_s)-1)].item()
        kept_mask = (scores[:, 0, :, :] >= threshold)

        # For each needle token, retention per layer (avg over heads)
        needle_tokens = []
        needle_labels_list = []
        for p in needle_pos:
            tok = token_labels[p].strip()[:8]
            typ = "K" if p in set(key_pos) else ("V" if p in set(value_pos) else "O")
            needle_labels_list.append(f"{p}:{tok}[{typ}]")
            # [n_layers] retention per layer (avg over heads)
            per_layer = kept_mask[:, :, p].float().mean(dim=1).numpy()  # [n_layers]
            needle_tokens.append(per_layer)

        heatmap = np.array(needle_tokens)  # [n_needle, n_layers]

        fig, ax = plt.subplots(figsize=(18, 6))
        im = ax.imshow(heatmap, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_yticks(range(len(needle_labels_list)))
        ax.set_yticklabels(needle_labels_list, fontsize=9)
        ax.set_xticks(range(0, n_layers, 2))
        ax.set_xticklabels([f'L{i}' for i in range(0, n_layers, 2)], fontsize=8)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Needle Token', fontsize=12)
        ax.set_title(f'Per-Layer Retention of Needle Tokens at CR={cr}\n'
                     f'(fraction of heads retaining each token)', fontsize=13)
        plt.colorbar(im, ax=ax, label='Retention fraction', shrink=0.8)

        # Add text annotations
        for i in range(len(needle_tokens)):
            for j in range(n_layers):
                val = heatmap[i, j]
                if val > 0:
                    ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=5,
                           color='black' if val > 0.5 else 'white')

        plt.tight_layout()
        path = os.path.join(out_dir, f'perlayer_heatmap_cr{int(cr*100)}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # ═══════════════════════════════════════════════════════
    # Figure 4: Score distribution with needle token positions marked
    # ═══════════════════════════════════════════════════════
    print("Generating Figure 4: Score distribution...")

    avg_scores = scores[:, 0, :, :].mean(dim=(0, 1)).numpy()  # [ctx_len]

    fig, ax = plt.subplots(figsize=(14, 5))
    # Histogram of all scores
    ax.hist(avg_scores, bins=100, alpha=0.5, color='gray', label='All tokens', density=True)

    # Mark needle components
    key_scores = [avg_scores[p] for p in key_pos]
    value_scores = [avg_scores[p] for p in value_pos]
    for s in key_scores:
        ax.axvline(x=s, color='red', alpha=0.7, linewidth=2)
    for s in value_scores:
        ax.axvline(x=s, color='blue', alpha=0.7, linewidth=2)

    # Thresholds
    for cr, ls in [(0.7, '--'), (0.9, '-')]:
        n_pruned = int(n_layers * 1 * n_kv_heads * ctx_len * cr)
        flat = scores.reshape(-1)
        sorted_s, _ = flat.sort()
        threshold = sorted_s[min(n_pruned, len(sorted_s)-1)].item()
        # Convert to avg score threshold (approximate)
        avg_threshold = np.percentile(avg_scores, cr * 100)
        ax.axvline(x=avg_threshold, color='green', linestyle=ls, linewidth=2,
                  label=f'CR={cr} threshold (P{int(cr*100)})')

    ax.set_xlabel('Average Score (across layers×heads)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Score Distribution with Needle Token Positions', fontsize=14)

    patches = [mpatches.Patch(color='red', label='Key tokens'),
               mpatches.Patch(color='blue', label='Value tokens')]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0][:2], fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, 'score_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ═══════════════════════════════════════════════════════
    # Figure 5: Diff — what cr=0.7 keeps but cr=0.9 loses
    # ═══════════════════════════════════════════════════════
    print("Generating Figure 5: Diff between CR=0.7 and CR=0.9...")

    retention_07 = {}
    retention_09 = {}
    for cr, store in [(0.7, retention_07), (0.9, retention_09)]:
        n_pruned = int(n_layers * 1 * n_kv_heads * ctx_len * cr)
        flat = scores.reshape(-1)
        sorted_s, _ = flat.sort()
        threshold = sorted_s[min(n_pruned, len(sorted_s)-1)].item()
        kept_mask = (scores[:, 0, :, :] >= threshold)
        store['frac'] = kept_mask.float().mean(dim=(0, 1)).numpy()
        store['threshold'] = threshold

    diff = retention_07['frac'] - retention_09['frac']  # positive = lost when going to cr=0.9

    fig, ax = plt.subplots(figsize=(24, 6))
    colors_diff = []
    for i in range(ctx_len):
        colors_diff.append(type_colors[token_type[i]])
    ax.bar(range(ctx_len), diff, color=colors_diff, width=1.0, edgecolor='none')
    ax.set_ylabel('Retention lost (CR=0.7 → CR=0.9)', fontsize=12)
    ax.set_title('What CR=0.7 keeps but CR=0.9 loses\n(higher = more impact from tighter compression)', fontsize=14)
    ax.set_xlim(-1, ctx_len)

    if needle_pos:
        ax.axvspan(needle_pos[0]-0.5, needle_pos[-1]+0.5, alpha=0.15, color='red', label='Needle')
    if question_pos:
        ax.axvspan(question_pos[0]-0.5, question_pos[-1]+0.5, alpha=0.15, color='green', label='Question')
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, 'diff_cr07_vs_cr09.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    print(f"\nAll figures saved to: {out_dir}")
    print("DONE")


if __name__ == "__main__":
    main()
