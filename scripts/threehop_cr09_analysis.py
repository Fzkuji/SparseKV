#!/usr/bin/env python3
"""Analyze WHY 3-hop tracing fails at cr=0.9.

Detailed analysis:
1. Score distribution at cr=0.9 vs 0.7
2. Which tokens get kept/evicted at cr=0.9?
3. Per-layer budget allocation (how many tokens per head)
4. Are needle tokens being evicted? Which ones?
5. Step-by-step: which step's score is causing the problem?
"""

import torch
import random
import math
import re
import gc
import time
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


def compute_threehop_scores_with_steps(model, tokenizer, context_ids,
                                        fanin_temp=1.0, chunk_size=1024):
    """Same as compute_threehop_scores but also returns per-step scores."""
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = context_ids.shape[1]

    user_pos = find_user_input_positions(tokenizer, context_ids, ctx_len)
    user_pos_t = torch.tensor(user_pos, dtype=torch.long, device=model.device)
    n_user = len(user_pos)

    final_scores = torch.zeros(n_layers, 1, n_kv_heads, ctx_len, dtype=torch.float32)
    step1_all = torch.zeros(n_layers, n_kv_heads, ctx_len, dtype=torch.float32)
    step2_all = torch.zeros(n_layers, n_kv_heads, ctx_len, dtype=torch.float32)
    step3_all = torch.zeros(n_layers, n_kv_heads, ctx_len, dtype=torch.float32)
    fanin_all = torch.zeros(n_layers, n_kv_heads, ctx_len, dtype=torch.float32)

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
                q = module.q_proj(hidden_states)
                q = q.view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                k = module.k_proj(hidden_states)
                k = k.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                    k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))

                q_grouped = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                scale = math.sqrt(head_dim)

                for hi in range(n_kv_heads):
                    k_h = k[0, hi]
                    q_h = q_grouped[0, hi]

                    # Step 1
                    user_q = q_h[:, user_pos_t, :]
                    user_logits = torch.matmul(user_q, k_h.T) / scale
                    for ui in range(n_user):
                        user_logits[:, ui, user_pos[ui]+1:] = float('-inf')
                    user_attn = torch.softmax(user_logits.float(), dim=-1)
                    step1 = user_attn.amax(dim=(0, 1))
                    del user_logits, user_attn, user_q

                    # Pass 1: exact fan-in
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

                    inv_fanin = 1.0 / (fan_in + 1e-6).pow(fanin_temp)
                    inv_fanin = inv_fanin / inv_fanin.sum()
                    step1_weighted = step1 * inv_fanin

                    # Pass 2: step2+step3
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

                    s1n = norm01(step1)
                    s2n = norm01(step2)
                    s3n = norm01(step3)
                    combined = s1n + s2n + s3n

                    final_scores[layer_idx, 0, hi, :] = combined.cpu()
                    step1_all[layer_idx, hi, :] = s1n.cpu()
                    step2_all[layer_idx, hi, :] = s2n.cpu()
                    step3_all[layer_idx, hi, :] = s3n.cpu()
                    fanin_all[layer_idx, hi, :] = fan_in.cpu()

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
        model.model(input_ids=context_ids, past_key_values=cache)
    for h in hooks:
        h.remove()

    return final_scores, step1_all, step2_all, step3_all, fanin_all, cache


def find_needle_positions(tokenizer, ctx_ids, ctx_len, target_key, target_value):
    """Find exact token positions for target needle components."""
    full_text = tokenizer.decode(ctx_ids[0])

    # Build char→token mapping
    char_positions = []
    cum = ""
    for i in range(ctx_len):
        start = len(cum)
        cum += tokenizer.decode(ctx_ids[0, i])
        char_positions.append((start, len(cum)))

    def chars_to_tokens(char_start, char_end):
        tokens = []
        for i, (cs, ce) in enumerate(char_positions):
            if ce > char_start and cs < char_end:
                tokens.append(i)
        return tokens

    # Find full needle sentence
    needle_text = f"One of the special magic numbers for {target_key} is: {target_value}."
    needle_char = full_text.find(needle_text)
    if needle_char < 0:
        return None

    sentence_toks = chars_to_tokens(needle_char, needle_char + len(needle_text))

    # Key tokens
    key_char = full_text.find(target_key, needle_char, needle_char + len(needle_text))
    key_toks = chars_to_tokens(key_char, key_char + len(target_key))

    # Value tokens
    val_char = full_text.find(target_value, needle_char, needle_char + len(needle_text))
    val_toks = chars_to_tokens(val_char, val_char + len(target_value))

    # Period
    period_char = needle_char + len(needle_text) - 1
    period_toks = chars_to_tokens(period_char, period_char + 1)

    # Question text
    q_text = f"What is the special magic number for {target_key}"
    q_char = full_text.find(q_text)
    q_toks = chars_to_tokens(q_char, q_char + len(q_text)) if q_char >= 0 else []

    return {
        'sentence': sentence_toks,
        'key': key_toks,
        'value': val_toks,
        'period': period_toks,
        'question': q_toks,
    }


def main():
    model_path = "Qwen/Qwen3-8B"
    target_key = "mystic-thunder"
    target_value = "7156842"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    print("Model loaded.")

    # Use 30 distractors for detailed analysis
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
    q_ids = tokenizer.encode(q_suffix, return_tensors="pt",
                             add_special_tokens=False).to(model.device)
    ctx_len = ctx_ids.shape[1]

    print(f"\n30 distractors, {ctx_len} tokens")

    # Find needle positions
    positions = find_needle_positions(tokenizer, ctx_ids, ctx_len, target_key, target_value)
    if positions is None:
        print("Cannot find needle!")
        return

    print(f"  Needle sentence: pos {positions['sentence'][0]}-{positions['sentence'][-1]}")
    print(f"  Key tokens ({target_key}): {positions['key']}")
    print(f"  Value tokens ({target_value}): {positions['value']}")
    print(f"  Period: {positions['period']}")
    print(f"  Question text: pos {positions['question'][0]}-{positions['question'][-1]}")

    # Show actual tokens
    for name, toks in positions.items():
        tok_strs = [tokenizer.decode(ctx_ids[0, t]) for t in toks[:8]]
        print(f"  {name}: {tok_strs}")

    # Compute scores
    print(f"\nComputing 3-hop scores...")
    scores, s1, s2, s3, fanin, cache = compute_threehop_scores_with_steps(
        model, tokenizer, ctx_ids)
    del cache
    torch.cuda.empty_cache()

    n_layers = scores.shape[0]
    n_kv_heads = scores.shape[2]

    # ═══ Analysis 1: Per-step scores for needle vs others ═══
    print(f"\n{'='*80}")
    print("ANALYSIS 1: Per-step scores for needle components (averaged over heads)")
    print(f"{'='*80}")

    for step_name, step_scores in [("Step1", s1), ("Step2", s2), ("Step3", s3), ("Combined", scores[:, 0])]:
        print(f"\n  {step_name}:")
        print(f"  {'Layer':>6} {'Key':>8} {'Value':>8} {'Period':>8} {'Question':>8} {'Other':>8} {'Key/Other':>10}")
        for li in range(0, n_layers, 4):
            if step_name == "Combined":
                ls = step_scores[li].mean(dim=0)
            else:
                ls = step_scores[li].mean(dim=0)
            key_s = ls[positions['key']].mean().item()
            val_s = ls[positions['value']].mean().item()
            per_s = ls[positions['period']].mean().item() if positions['period'] else 0
            q_s = ls[positions['question']].mean().item()
            mask = torch.ones(ctx_len, dtype=torch.bool)
            mask[:4] = False
            for t in positions['sentence']:
                mask[t] = False
            for t in positions['question']:
                mask[t] = False
            other_s = ls[mask].mean().item()
            ratio = key_s / max(other_s, 1e-10)
            print(f"  L{li:>4}: {key_s:>8.4f} {val_s:>8.4f} {per_s:>8.4f} {q_s:>8.4f} {other_s:>8.4f} {ratio:>9.3f}x")

    # ═══ Analysis 2: Token retention at cr=0.7 vs cr=0.9 ═══
    print(f"\n{'='*80}")
    print("ANALYSIS 2: Token retention at cr=0.7 vs cr=0.9 (global bottom-k)")
    print(f"{'='*80}")

    flat_scores = scores.reshape(-1)
    total = flat_scores.numel()

    for cr in [0.7, 0.9, 0.95]:
        n_pruned = int(total * cr)
        threshold = torch.topk(flat_scores, total - n_pruned).values[-1].item()

        # For each needle component, count how many (layer, head) pairs retain it
        print(f"\n  cr = {cr} (threshold = {threshold:.4f}):")
        print(f"  Total budget: {total - n_pruned}/{total} ({(1-cr)*100:.0f}%)")

        for name, toks in [('key', positions['key']), ('value', positions['value']),
                           ('period', positions['period']), ('question', positions['question'][:5])]:
            n_retained = 0
            n_total = 0
            for t in toks:
                for li in range(n_layers):
                    for hi in range(n_kv_heads):
                        n_total += 1
                        if scores[li, 0, hi, t].item() >= threshold:
                            n_retained += 1
            print(f"    {name:10s}: {n_retained}/{n_total} retained ({n_retained/max(n_total,1)*100:.1f}%)")

        # Per-layer budget distribution
        print(f"\n  Per-layer budget distribution (tokens kept per head, avg):")
        for li in range(0, n_layers, 4):
            layer_scores = scores[li, 0]  # [n_kv_heads, ctx_len]
            kept_per_head = (layer_scores >= threshold).sum(dim=1).float()
            print(f"    L{li:>2}: min={kept_per_head.min().item():.0f} "
                  f"avg={kept_per_head.mean().item():.0f} "
                  f"max={kept_per_head.max().item():.0f} / {ctx_len}")

    # ═══ Analysis 3: Fan-in values for key tokens ═══
    print(f"\n{'='*80}")
    print("ANALYSIS 3: Fan-in values (who is 'universal' vs 'specific')")
    print(f"{'='*80}")

    # Average over layers and heads
    avg_fanin = fanin.mean(dim=(0, 1))  # [ctx_len]

    # Top fan-in tokens (most universally attended)
    top_fanin_idx = avg_fanin.topk(10).indices
    print(f"\n  Top 10 highest fan-in tokens:")
    for i, idx in enumerate(top_fanin_idx):
        tok = tokenizer.decode(ctx_ids[0, idx])
        print(f"    #{i+1}: pos {idx.item()} '{tok}' fan_in={avg_fanin[idx].item():.2f}")

    # Fan-in of needle components
    print(f"\n  Needle component fan-in:")
    for name, toks in positions.items():
        if toks:
            fi = avg_fanin[toks].mean().item()
            print(f"    {name:10s}: {fi:.2f}")

    # ═══ Analysis 4: Score distribution ═══
    print(f"\n{'='*80}")
    print("ANALYSIS 4: Score distribution")
    print(f"{'='*80}")

    flat = scores.reshape(-1)
    print(f"\n  Overall: min={flat.min():.4f} mean={flat.mean():.4f} "
          f"median={flat.median():.4f} max={flat.max():.4f}")

    # Percentiles
    for p in [10, 30, 50, 70, 90, 95, 99]:
        val = torch.quantile(flat.float(), p / 100).item()
        print(f"    P{p:>2}: {val:.4f}")

    # Needle scores vs percentile
    print(f"\n  Needle token score percentiles:")
    for name, toks in [('key', positions['key']), ('value', positions['value']),
                       ('period', positions['period'])]:
        for t in toks:
            tok_scores = scores[:, 0, :, t].reshape(-1)
            avg_score = tok_scores.mean().item()
            # What percentile is this?
            pct = (flat < avg_score).float().mean().item() * 100
            tok_str = tokenizer.decode(ctx_ids[0, t])
            print(f"    {name} pos {t} '{tok_str}': avg_score={avg_score:.4f} (percentile={pct:.1f}%)")

    print("\n\nDONE")


if __name__ == "__main__":
    main()
