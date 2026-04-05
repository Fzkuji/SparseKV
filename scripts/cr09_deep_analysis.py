#!/usr/bin/env python3
"""Deep analysis of cr=0.9 failure:
1. Which needle tokens are evicted at cr=0.9?
2. Do question tokens directly attend to value tokens?
3. What is the attention pattern from question → needle components?
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


def find_question_positions(tokenizer, ctx_ids, ctx_len, target_key):
    """Find the question part: 'What is the special magic number for mystic-thunder ...'"""
    question_str = f"What is the special magic number for {target_key}"
    return find_token_positions(tokenizer, ctx_ids, question_str)


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
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    print(f"Model loaded. {n_layers} layers, {n_kv_heads} KV heads, {n_groups} groups/head")

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
    print(f"Context: {ctx_len} tokens")

    # Find all positions
    key_pos = find_token_positions(tokenizer, ctx_ids, target_key)
    value_pos = find_token_positions(tokenizer, ctx_ids, target_value)
    needle_str = f"One of the special magic numbers for {target_key} is: {target_value}."
    needle_pos = find_token_positions(tokenizer, ctx_ids, needle_str)
    question_pos = find_question_positions(tokenizer, ctx_ids, ctx_len, target_key)
    user_pos = find_user_input_positions(tokenizer, ctx_ids, ctx_len)

    print(f"\nNeedle: pos {needle_pos[0]}-{needle_pos[-1]} ({len(needle_pos)} tokens)")
    print(f"  Key '{target_key}': {key_pos}")
    print(f"  Value '{target_value}': {value_pos}")
    print(f"  Question: {question_pos}")
    print(f"  User input: {user_pos[0]}-{user_pos[-1]} ({len(user_pos)} tokens)")

    # Print token-by-token for needle
    print(f"\nNeedle tokens:")
    for p in needle_pos:
        tok = tokenizer.decode(ctx_ids[0, p])
        label = ""
        if p in key_pos:
            label = " [KEY]"
        elif p in value_pos:
            label = " [VALUE]"
        print(f"  pos {p}: '{tok}'{label}")

    print(f"\nQuestion tokens:")
    for p in question_pos:
        tok = tokenizer.decode(ctx_ids[0, p])
        print(f"  pos {p}: '{tok}'")

    # ═══════════════════════════════════════════════════════
    # PART 1: Compute 3-hop scores AND capture raw attention from question → all positions
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("PART 1: Computing scores + raw attention from question tokens")
    print(f"{'=' * 80}")

    user_pos_t = torch.tensor(user_pos, dtype=torch.long, device=model.device)
    question_pos_t = torch.tensor(question_pos, dtype=torch.long, device=model.device)
    n_user = len(user_pos)

    final_scores = torch.zeros(n_layers, 1, n_kv_heads, ctx_len, dtype=torch.float32)

    # Store raw attention: question tokens → all positions, per layer per head
    # Shape: [n_layers, n_kv_heads, len(question_pos), ctx_len]
    n_q = len(question_pos)
    question_attn_all = torch.zeros(n_layers, n_kv_heads, n_q, ctx_len, dtype=torch.float32)

    # Also store: ALL user input tokens → needle positions (to see who attends to value)
    # For each layer/head, store attention from every user-input token to each needle pos
    # Shape: [n_layers, n_kv_heads, n_user, len(needle_pos)]
    n_needle = len(needle_pos)
    user_to_needle_attn = torch.zeros(n_layers, n_kv_heads, n_user, n_needle, dtype=torch.float32)

    chunk_size = 1024

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
                    k_h = k[0, hi]  # [L, d]
                    q_h = q_grouped[0, hi]  # [n_groups, L, d]

                    # ── Raw attention: question tokens → all positions ──
                    q_question = q_h[:, question_pos_t, :]  # [n_groups, n_q, d]
                    q_logits = torch.matmul(q_question, k_h.T) / scale  # [n_groups, n_q, L]
                    for qi in range(n_q):
                        q_logits[:, qi, question_pos[qi]+1:] = float('-inf')
                    q_attn = torch.softmax(q_logits.float(), dim=-1)  # [n_groups, n_q, L]
                    # Max over groups
                    q_attn_max = q_attn.amax(dim=0)  # [n_q, L]
                    question_attn_all[layer_idx, hi] = q_attn_max.cpu()
                    del q_logits

                    # ── Raw attention: ALL user tokens → needle positions ──
                    user_q = q_h[:, user_pos_t, :]  # [n_groups, n_user, d]
                    needle_pos_t = torch.tensor(needle_pos, dtype=torch.long, device=k_h.device)
                    k_needle = k_h[needle_pos_t]  # [n_needle, d]
                    u_logits = torch.matmul(user_q, k_needle.T) / scale  # [n_groups, n_user, n_needle]
                    # For proper softmax we need full attention, but this gives unnormalized scores
                    # Let's compute full attention for user tokens instead
                    u_full_logits = torch.matmul(user_q, k_h.T) / scale  # [n_groups, n_user, L]
                    for ui in range(n_user):
                        u_full_logits[:, ui, user_pos[ui]+1:] = float('-inf')
                    u_full_attn = torch.softmax(u_full_logits.float(), dim=-1)  # [n_groups, n_user, L]
                    u_full_attn_max = u_full_attn.amax(dim=0)  # [n_user, L]
                    # Extract needle columns
                    user_to_needle_attn[layer_idx, hi] = u_full_attn_max[:, needle_pos_t].cpu()
                    del u_logits, u_full_logits, u_full_attn, user_q

                    # ── 3-hop scoring (same as before) ──
                    user_q2 = q_h[:, user_pos_t, :]
                    user_logits = torch.matmul(user_q2, k_h.T) / scale
                    for ui in range(n_user):
                        user_logits[:, ui, user_pos[ui]+1:] = float('-inf')
                    user_attn = torch.softmax(user_logits.float(), dim=-1)
                    step1 = user_attn.amax(dim=(0, 1))
                    del user_logits, user_attn, user_q2

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

    # ═══════════════════════════════════════════════════════
    # PART 2: Analyze direct attention from question → needle components
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("PART 2: Direct attention from QUESTION tokens → needle components")
    print(f"{'=' * 80}")

    # For each layer, show how much attention question tokens give to key vs value vs other
    needle_pos_set = set(needle_pos)
    key_pos_set = set(key_pos)
    value_pos_set = set(value_pos)

    # Map needle positions to indices in needle_pos list
    key_needle_idx = [needle_pos.index(p) for p in key_pos if p in needle_pos_set]
    value_needle_idx = [needle_pos.index(p) for p in value_pos if p in needle_pos_set]

    print(f"\n  Question attention to different targets (max over heads):")
    print(f"  {'Layer':>6} {'→Key':>10} {'→Value':>10} {'→Needle':>10} {'→Sink(0:4)':>12} {'→Other':>10} {'Val/Other':>10}")

    for li in range(0, n_layers, 4):
        # question_attn_all: [n_layers, n_kv_heads, n_q, ctx_len]
        # Average over question tokens, max over heads
        attn = question_attn_all[li]  # [n_kv_heads, n_q, ctx_len]
        attn_avg_q = attn.mean(dim=1)  # [n_kv_heads, ctx_len] - avg over question tokens
        attn_max_h = attn_avg_q.max(dim=0).values  # [ctx_len] - max over heads

        to_key = attn_max_h[torch.tensor(key_pos)].sum().item()
        to_value = attn_max_h[torch.tensor(value_pos)].sum().item()
        to_needle = attn_max_h[torch.tensor(needle_pos)].sum().item()
        to_sink = attn_max_h[:4].sum().item()

        mask = torch.ones(ctx_len, dtype=torch.bool)
        mask[:4] = False
        for p in needle_pos:
            mask[p] = False
        to_other = attn_max_h[mask].mean().item()

        val_ratio = to_value / max(to_other * len(value_pos), 1e-10)
        print(f"  L{li:>4}: {to_key:>10.5f} {to_value:>10.5f} {to_needle:>10.5f} {to_sink:>12.5f} {to_other:>10.5f} {val_ratio:>9.2f}x")

    # ═══ Per-head breakdown for top layers ═══
    print(f"\n  Per-head attention from question → value tokens (top layers):")
    print(f"  {'Layer':>6} " + " ".join([f"{'H'+str(h):>8}" for h in range(n_kv_heads)]))
    for li in [20, 24, 28, 32, 34, 35]:
        if li >= n_layers:
            continue
        attn = question_attn_all[li]  # [n_kv_heads, n_q, ctx_len]
        attn_avg_q = attn.mean(dim=1)  # [n_kv_heads, ctx_len]
        per_head_to_value = attn_avg_q[:, torch.tensor(value_pos)].sum(dim=1)  # [n_kv_heads]
        vals = " ".join([f"{v.item():>8.5f}" for v in per_head_to_value])
        print(f"  L{li:>4}: {vals}")

    # ═══ Which specific question tokens attend most to value? ═══
    print(f"\n  Per-question-token attention to VALUE (averaged over layers 24-35, max head):")
    for qi, qp in enumerate(question_pos):
        tok = tokenizer.decode(ctx_ids[0, qp])
        attn_to_val = 0
        count = 0
        for li in range(24, min(36, n_layers)):
            attn = question_attn_all[li]  # [n_kv_heads, n_q, ctx_len]
            # For this question token, max over heads
            attn_qi = attn[:, qi, :]  # [n_kv_heads, ctx_len]
            attn_qi_max = attn_qi.max(dim=0).values  # [ctx_len]
            attn_to_val += attn_qi_max[torch.tensor(value_pos)].sum().item()
            count += 1
        avg = attn_to_val / max(count, 1)
        print(f"    Q[{qi}] pos {qp} '{tok}': avg attn to value = {avg:.6f}")

    # ═══════════════════════════════════════════════════════
    # PART 3: Who in the ENTIRE context attends to value tokens?
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("PART 3: Who attends to VALUE tokens? (from all user-input tokens)")
    print(f"{'=' * 80}")

    # user_to_needle_attn: [n_layers, n_kv_heads, n_user, n_needle]
    # Sum attention to value tokens
    value_needle_idx_t = torch.tensor(value_needle_idx)

    print(f"\n  Top-20 user-input tokens that attend MOST to value (avg over L24-35, max head):")
    user_to_value = torch.zeros(n_user)
    count = 0
    for li in range(24, min(36, n_layers)):
        attn = user_to_needle_attn[li]  # [n_kv_heads, n_user, n_needle]
        attn_to_val = attn[:, :, value_needle_idx_t].sum(dim=2)  # [n_kv_heads, n_user]
        attn_max = attn_to_val.max(dim=0).values  # [n_user]
        user_to_value += attn_max
        count += 1
    user_to_value /= max(count, 1)

    topk = user_to_value.topk(20)
    for rank, (score, idx) in enumerate(zip(topk.values, topk.indices)):
        pos = user_pos[idx.item()]
        tok = tokenizer.decode(ctx_ids[0, pos])
        label = ""
        if pos in key_pos_set:
            label = " ← KEY TOKEN"
        elif pos in value_pos_set:
            label = " ← VALUE TOKEN (self)"
        elif pos in set(question_pos):
            label = " ← QUESTION TOKEN"
        print(f"    #{rank+1}: pos {pos} '{tok}' attn={score.item():.6f}{label}")

    # ═══ Same but for KEY tokens ═══
    key_needle_idx_t = torch.tensor(key_needle_idx)
    print(f"\n  Top-20 user-input tokens that attend MOST to KEY tokens (avg over L24-35, max head):")
    user_to_key = torch.zeros(n_user)
    count = 0
    for li in range(24, min(36, n_layers)):
        attn = user_to_needle_attn[li]
        attn_to_key = attn[:, :, key_needle_idx_t].sum(dim=2)
        attn_max = attn_to_key.max(dim=0).values
        user_to_key += attn_max
        count += 1
    user_to_key /= max(count, 1)

    topk = user_to_key.topk(20)
    for rank, (score, idx) in enumerate(zip(topk.values, topk.indices)):
        pos = user_pos[idx.item()]
        tok = tokenizer.decode(ctx_ids[0, pos])
        label = ""
        if pos in key_pos_set:
            label = " ← KEY TOKEN (self)"
        elif pos in value_pos_set:
            label = " ← VALUE TOKEN"
        elif pos in set(question_pos):
            label = " ← QUESTION TOKEN"
        print(f"    #{rank+1}: pos {pos} '{tok}' attn={score.item():.6f}{label}")

    # ═══════════════════════════════════════════════════════
    # PART 4: cr=0.9 eviction analysis — which needle tokens survive?
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("PART 4: cr=0.9 eviction — which needle tokens are kept/evicted?")
    print(f"{'=' * 80}")

    # Global allocation
    cr = 0.9
    total = n_layers * 1 * n_kv_heads * ctx_len
    n_pruned = int(total * cr)
    flat_scores = final_scores.reshape(-1)
    threshold_idx = n_pruned
    sorted_scores, _ = flat_scores.sort()
    threshold = sorted_scores[min(threshold_idx, len(sorted_scores)-1)].item()
    print(f"\n  Global threshold at cr={cr}: {threshold:.4f}")
    print(f"  Total slots: {total}, pruned: {n_pruned}, kept: {total - n_pruned}")

    # Per needle token: in how many (layer, head) pairs is it kept?
    print(f"\n  Needle token retention (out of {n_layers} layers × {n_kv_heads} heads = {n_layers * n_kv_heads} slots):")
    print(f"  {'Pos':>5} {'Token':>15} {'Type':>8} {'Kept':>6} {'%':>6} {'AvgScore':>10} {'MaxScore':>10} {'MinScore':>10}")

    for p in needle_pos:
        tok = tokenizer.decode(ctx_ids[0, p]).strip()
        if p in key_pos_set:
            typ = "KEY"
        elif p in value_pos_set:
            typ = "VALUE"
        else:
            typ = "other"

        scores_at_p = final_scores[:, 0, :, p]  # [n_layers, n_kv_heads]
        kept = (scores_at_p >= threshold).sum().item()
        total_slots = n_layers * n_kv_heads
        pct = kept / total_slots * 100
        avg_s = scores_at_p.mean().item()
        max_s = scores_at_p.max().item()
        min_s = scores_at_p.min().item()
        print(f"  {p:>5} {tok:>15} {typ:>8} {kept:>6} {pct:>5.1f}% {avg_s:>10.4f} {max_s:>10.4f} {min_s:>10.4f}")

    # ═══ Per-layer retention for value tokens ═══
    print(f"\n  Per-layer retention of VALUE tokens at cr={cr}:")
    print(f"  {'Layer':>6} " + " ".join([f"{'pos'+str(p):>8}" for p in value_pos]) + "  avg_kept")
    for li in range(0, n_layers, 2):
        parts = []
        kept_count = 0
        for vp in value_pos:
            scores_at = final_scores[li, 0, :, vp]  # [n_kv_heads]
            n_kept = (scores_at >= threshold).sum().item()
            kept_count += n_kept
            parts.append(f"{n_kept:>8}")
        avg_k = kept_count / len(value_pos)
        print(f"  L{li:>4}: " + " ".join(parts) + f"  {avg_k:>8.1f}/{n_kv_heads}")

    # ═══ Which layers/heads have the highest scores for value tokens? ═══
    print(f"\n  Top-20 (layer, head) pairs with highest scores for VALUE tokens:")
    val_scores = final_scores[:, 0, :, torch.tensor(value_pos)]  # [n_layers, n_kv_heads, n_val]
    val_mean = val_scores.mean(dim=2)  # [n_layers, n_kv_heads]
    flat_val = val_mean.reshape(-1)
    topk = flat_val.topk(20)
    for rank, (score, idx) in enumerate(zip(topk.values, topk.indices)):
        li = idx.item() // n_kv_heads
        hi = idx.item() % n_kv_heads
        print(f"    #{rank+1}: L{li} H{hi} avg_value_score={score.item():.4f}")

    print("\n\nDONE")


if __name__ == "__main__":
    main()
