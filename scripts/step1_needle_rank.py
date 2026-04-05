#!/usr/bin/env python3
"""Simple: what's the rank of needle's leading tokens in Step1 scores?
At what CR are they retained?
"""

import torch
import random
import math
import numpy as np
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

    print(f"Context: {ctx_len} tokens")

    # Find needle token positions and decode them
    full_text_dec = tokenizer.decode(ctx_ids[0])
    needle_str = f"One of the special magic numbers for {target_key} is: {target_value}."
    needle_char = full_text_dec.find(needle_str)
    needle_pos = []
    if needle_char >= 0:
        cum = ""
        for i in range(ctx_len):
            prev_len = len(cum)
            cum += tokenizer.decode(ctx_ids[0, i])
            if len(cum) > needle_char and prev_len < needle_char + len(needle_str):
                needle_pos.append(i)

    print(f"\nNeedle tokens ({len(needle_pos)} total):")
    for i, p in enumerate(needle_pos):
        tok = tokenizer.decode(ctx_ids[0, p]).replace('\n', '\\n')
        print(f"  [{i:>2}] pos={p:>3}: '{tok}'")

    # Also find "mystic-thunder" key tokens specifically (the unique part)
    key_str = "mystic-thunder"
    key_char = full_text_dec.find(f"numbers for {key_str} is:")
    key_pos = []
    if key_char >= 0:
        # Find just the "mystic-thunder" part
        key_start = key_char + len("numbers for ")
        cum = ""
        for i in range(ctx_len):
            prev_len = len(cum)
            cum += tokenizer.decode(ctx_ids[0, i])
            if len(cum) > key_start and prev_len < key_start + len(key_str):
                key_pos.append(i)
    print(f"\nKey tokens ('{key_str}'): positions {key_pos}")
    for p in key_pos:
        tok = tokenizer.decode(ctx_ids[0, p]).replace('\n', '\\n')
        print(f"  pos={p}: '{tok}'")

    # ═══════════════════════════════════════════
    # Compute Step1 scores: question outgoing attention, max over groups
    # This is what 3-hop actually uses
    # ═══════════════════════════════════════════
    print("\nComputing Step1 scores (question → KV attention)...")

    q_pos = []
    question_str = f"What is the special magic number for {target_key} mentioned in the provided text?"
    q_char = full_text_dec.find(question_str)
    if q_char >= 0:
        cum = ""
        for i in range(ctx_len):
            prev_len = len(cum)
            cum += tokenizer.decode(ctx_ids[0, i])
            if len(cum) > q_char and prev_len < q_char + len(question_str):
                q_pos.append(i)
    q_pos_t = torch.tensor(q_pos, dtype=torch.long, device=model.device)
    n_q = len(q_pos)

    # step1_scores[layer, kv_head, ctx_len] = max over groups, max over q tokens
    step1_scores = torch.zeros(n_layers, n_kv_heads, ctx_len, dtype=torch.float32)

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
                    k_h = k[0, hi]  # [L, d]
                    head_scores = []
                    for gi in range(n_groups):
                        q_g = q_grouped[0, hi, gi, q_pos_t, :]  # [n_q, d]
                        logits = torch.matmul(q_g, k_h.T) / scale  # [n_q, L]
                        for qi in range(n_q):
                            logits[qi, q_pos[qi]+1:] = float('-inf')
                        attn = torch.softmax(logits.float(), dim=-1)  # [n_q, L]
                        head_scores.append(attn)
                    # Stack: [n_groups, n_q, L] → max over groups and q tokens → [L]
                    stacked = torch.stack(head_scores, dim=0)  # [G, Q, L]
                    score = stacked.amax(dim=(0, 1))  # [L]
                    step1_scores[layer_idx, hi] = score.cpu()
                del q, k, q_grouped

        return hook_fn

    hooks = []
    for layer in model.model.layers:
        h = layer.self_attn.register_forward_hook(
            make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True)
        hooks.append(h)

    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=DynamicCache())
    for h in hooks:
        h.remove()
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════
    # Rank of each needle token, per layer per KV head
    # ═══════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"Rank of needle's FIRST token (pos {needle_pos[0]}) per layer per KV head")
    print(f"(rank 1 = highest score, rank {ctx_len} = lowest)")
    print(f"{'='*100}")

    first_token = needle_pos[0]

    print(f"\n  {'Layer':>6}", end="")
    for hi in range(n_kv_heads):
        print(f"  {'KV'+str(hi):>6}", end="")
    print(f"  {'Best':>6}  {'BestHead':>8}")

    for li in range(n_layers):
        print(f"  {li:>6}", end="")
        ranks = []
        for hi in range(n_kv_heads):
            scores = step1_scores[li, hi].numpy()
            # Rank: how many tokens have score >= this token's score
            rank = (scores >= scores[first_token]).sum()
            ranks.append(int(rank))
            print(f"  {rank:>6}", end="")
        best_rank = min(ranks)
        best_head = ranks.index(best_rank)
        print(f"  {best_rank:>6}  KV{best_head:>6}")

    # ═══════════════════════════════════════════
    # Same for key tokens specifically
    # ═══════════════════════════════════════════
    if key_pos:
        print(f"\n{'='*100}")
        print(f"Rank of KEY tokens ({key_pos}) per layer — best head only")
        print(f"{'='*100}")

        print(f"\n  {'Layer':>6}", end="")
        for ki, kp in enumerate(key_pos):
            tok = tokenizer.decode(ctx_ids[0, kp]).strip()[:8]
            print(f"  {f'p{kp}({tok})':>14}", end="")
        print()

        for li in range(n_layers):
            print(f"  {li:>6}", end="")
            for kp in key_pos:
                best = ctx_len
                for hi in range(n_kv_heads):
                    scores = step1_scores[li, hi].numpy()
                    rank = int((scores >= scores[kp]).sum())
                    best = min(best, rank)
                print(f"  {best:>14}", end="")
            print()

    # ═══════════════════════════════════════════
    # For each needle token: what CR would evict it?
    # ═══════════════════════════════════════════
    print(f"\n{'='*100}")
    print("Per needle token: max CR where it survives (across all heads, best head)")
    print("CR = fraction evicted. If rank=R out of N, survives up to CR = 1 - R/N")
    print(f"{'='*100}")

    # Focus on key layers
    for li in [20, 24, 28, 32]:
        if li >= n_layers:
            continue
        print(f"\n  Layer {li}:")
        print(f"  {'Pos':>5} {'Token':>12} {'BestRank':>9} {'MaxCR':>7} {'BestHead':>9}  All ranks")

        for i, p in enumerate(needle_pos):
            tok = tokenizer.decode(ctx_ids[0, p]).replace('\n', '\\n').strip()[:10]
            ranks = []
            for hi in range(n_kv_heads):
                scores = step1_scores[li, hi].numpy()
                rank = int((scores >= scores[p]).sum())
                ranks.append(rank)
            best_rank = min(ranks)
            best_head = ranks.index(best_rank)
            max_cr = 1.0 - best_rank / ctx_len
            marker = " ←KEY" if p in key_pos else ""
            print(f"  {p:>5} {tok:>12} {best_rank:>9} {max_cr:>6.1%} {'KV'+str(best_head):>9}  {ranks}{marker}")

    # ═══════════════════════════════════════════
    # Summary: what's the max CR to keep ALL key tokens?
    # ═══════════════════════════════════════════
    print(f"\n{'='*100}")
    print("Summary: max CR to keep specific needle parts (best across all KV heads)")
    print(f"{'='*100}")

    target_sets = {
        "first_token": [needle_pos[0]],
        "key_tokens": key_pos if key_pos else [needle_pos[0]],
        "all_needle": needle_pos,
    }

    for li in [20, 24, 28, 32]:
        if li >= n_layers:
            continue
        print(f"\n  Layer {li}:")
        for name, positions in target_sets.items():
            # For each position, find its best rank across all KV heads
            # The bottleneck is the position with the worst best-rank
            worst_best_rank = 0
            for p in positions:
                best_rank = ctx_len
                for hi in range(n_kv_heads):
                    scores = step1_scores[li, hi].numpy()
                    rank = int((scores >= scores[p]).sum())
                    best_rank = min(best_rank, rank)
                worst_best_rank = max(worst_best_rank, best_rank)
            max_cr = 1.0 - worst_best_rank / ctx_len
            print(f"    {name:>15}: worst_rank={worst_best_rank:>4}, max_CR={max_cr:.1%}")

    # ═══════════════════════════════════════════
    # But: we need ALL heads to keep it (not just best head)
    # Because eviction is per-head
    # ═══════════════════════════════════════════
    print(f"\n{'='*100}")
    print("Reality check: for per-head eviction, EACH head decides independently")
    print("If any head evicts the needle, that head's KV is lost")
    print(f"{'='*100}")

    for li in [20, 24, 28, 32]:
        if li >= n_layers:
            continue
        print(f"\n  Layer {li}:")
        for name, positions in target_sets.items():
            # For each head: worst rank among target positions
            # = the rank that determines if ALL target tokens survive in this head
            head_worst_ranks = []
            for hi in range(n_kv_heads):
                scores = step1_scores[li, hi].numpy()
                worst_rank = 0
                for p in positions:
                    rank = int((scores >= scores[p]).sum())
                    worst_rank = max(worst_rank, rank)
                head_worst_ranks.append(worst_rank)

            max_crs = [1.0 - r / ctx_len for r in head_worst_ranks]
            print(f"    {name:>15}: per-head max_CR = {['%.0f%%' % (c*100) for c in max_crs]}")
            print(f"    {'':>15}  per-head rank   = {head_worst_ranks}")

    print("\nDONE")


if __name__ == "__main__":
    main()
