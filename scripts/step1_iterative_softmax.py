#!/usr/bin/env python3
"""Iterative softmax sharpening for adaptive KV eviction.

Idea: apply softmax repeatedly to attention weights → distribution becomes
more peaked → important tokens naturally stand out → keep tokens with
weight > 1/N (uniform baseline).

No manual threshold or CR needed.
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


def compute_per_qtoken_attention(model, tokenizer, ctx_ids, target_key):
    """Compute per-question-token, per-query-group attention.
    Returns: [n_layers, n_kv_heads, n_groups, n_q, ctx_len] attention weights
    Also returns the DynamicCache.
    """
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = ctx_ids.shape[1]

    q_pos = find_question_positions(tokenizer, ctx_ids, ctx_len, target_key)
    q_pos_t = torch.tensor(q_pos, dtype=torch.long, device=model.device)
    n_q = len(q_pos)

    all_attn = torch.zeros(n_layers, n_kv_heads, n_groups, n_q, ctx_len, dtype=torch.float32)

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
                    for gi in range(n_groups):
                        q_g = q_grouped[0, hi, gi, q_pos_t, :]
                        logits = torch.matmul(q_g, k_h.T) / scale
                        for qi in range(n_q):
                            logits[qi, q_pos[qi]+1:] = float('-inf')
                        attn = torch.softmax(logits.float(), dim=-1)
                        all_attn[layer_idx, hi, gi] = attn.cpu()
                        del q_g, logits, attn
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

    return all_attn, cache, q_pos


def iterative_softmax(weights, n_iter):
    """Apply softmax n_iter times to sharpen the distribution.
    Input: numpy array [ctx_len], already a probability distribution (from first softmax).
    Output: sharpened distribution.
    """
    w = weights.copy()
    for _ in range(n_iter):
        # Treat current distribution as logits, re-apply softmax
        w = w - w.max()  # numerical stability
        w = np.exp(w)
        w = w / w.sum()
    return w


def select_tokens_iterative_softmax(attn, n_iter, n_kv_heads, n_groups, n_q, ctx_len,
                                     protect_sink=4, protect_recent=4):
    """For each KV head: for each (group, question_token):
    1. Apply iterative softmax to sharpen the attention distribution
    2. Keep tokens with weight > 1/ctx_len (above uniform)
    Union across all groups and question tokens.
    """
    uniform_threshold = 1.0 / ctx_len
    keep_sets = []

    for hi in range(n_kv_heads):
        keep = set(range(protect_sink))
        keep.update(range(ctx_len - protect_recent, ctx_len))

        for gi in range(n_groups):
            for qi in range(n_q):
                weights = attn[hi, gi, qi].numpy()
                sharpened = iterative_softmax(weights, n_iter)
                # Keep positions above uniform threshold
                above = np.where(sharpened > uniform_threshold)[0]
                keep.update(above.tolist())

        keep_sets.append(keep)

    return keep_sets


def generate_with_cache(model, tokenizer, cache, suffix_ids, max_new=30):
    with torch.no_grad():
        out = model(input_ids=suffix_ids, past_key_values=cache)

    generated = []
    next_token = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    for _ in range(max_new):
        tok_id = next_token[0, 0].item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated.append(tok_id)
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=cache)
        next_token = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    return tokenizer.decode(generated)


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
    suffix_ids = tokenizer.encode(q_suffix, return_tensors="pt",
                                  add_special_tokens=False).to(model.device)
    ctx_len = ctx_ids.shape[1]

    print(f"Context: {ctx_len} tokens, uniform threshold = {1.0/ctx_len:.6f}")
    print(f"Target: {target_key} → {target_value}")

    # Find needle positions
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
    needle_set = set(needle_pos)
    print(f"Needle: {len(needle_pos)} tokens (pos {needle_pos[0]}-{needle_pos[-1]})")

    # ═══════════════════════════════════════════
    # Compute attention
    # ═══════════════════════════════════════════
    print("\nComputing per-question-token attention...")
    all_attn, _, q_pos = compute_per_qtoken_attention(model, tokenizer, ctx_ids, target_key)
    n_q = len(q_pos)
    print(f"Question: {n_q} tokens")
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════
    # Visualize sharpening effect on a single distribution
    # ═══════════════════════════════════════════
    print(f"\n{'='*100}")
    print("Demo: iterative softmax sharpening on KV5, group0, q_token0, Layer 24")
    print(f"{'='*100}")

    demo_w = all_attn[24, 5, 0, 0].numpy()  # KV5 has retrieval head
    print(f"  Original: top-5 weights = {np.sort(demo_w)[::-1][:5]}")
    print(f"  Original: #tokens > uniform = {(demo_w > 1.0/ctx_len).sum()}")

    for n_iter in [1, 2, 3, 5, 10, 20]:
        sharp = iterative_softmax(demo_w, n_iter)
        above = (sharp > 1.0/ctx_len).sum()
        top5 = np.sort(sharp)[::-1][:5]
        # Check if needle key is in the "above uniform" set
        ndl_above = sum(1 for p in needle_pos if sharp[p] > 1.0/ctx_len)
        print(f"  iter={n_iter:>2}: top-5 = {top5}, #above_uniform = {above}, needle_above = {ndl_above}/{len(needle_pos)}")

    # ═══════════════════════════════════════════
    # Analysis: different iterations, different layers
    # ═══════════════════════════════════════════
    print(f"\n{'='*100}")
    print("Token count analysis: how many tokens does iterative softmax keep?")
    print(f"{'='*100}")

    for li in [20, 24, 28, 32]:
        if li >= n_layers:
            continue
        layer_attn = all_attn[li]

        print(f"\n  Layer {li}:")
        print(f"  {'Iters':>6} ", end="")
        for hi in range(n_kv_heads):
            print(f"{'KV'+str(hi):>8}", end="")
        print(f" {'Mean':>8} {'CR':>8} {'Needle%':>8}")

        for n_iter in [0, 1, 2, 3, 5, 10, 20]:
            keep_sets = select_tokens_iterative_softmax(
                layer_attn, n_iter, n_kv_heads, n_groups, n_q, ctx_len)

            sizes = [len(s) for s in keep_sets]
            mean_size = np.mean(sizes)
            cr = 1.0 - mean_size / ctx_len

            needle_covered = []
            for hi in range(n_kv_heads):
                covered = len(needle_set & keep_sets[hi]) / len(needle_set) * 100
                needle_covered.append(covered)
            mean_needle = np.mean(needle_covered)

            print(f"  {n_iter:>6} ", end="")
            for s in sizes:
                print(f"{s:>8}", end="")
            print(f" {mean_size:>7.0f} {cr:>7.1%} {mean_needle:>7.1f}%")

    # ═══════════════════════════════════════════
    # Detail at iteration=5, Layer 24
    # ═══════════════════════════════════════════
    print(f"\n{'='*100}")
    print("Detail: tokens selected at iter=5, Layer 24")
    print(f"{'='*100}")

    layer_attn = all_attn[24]
    keep_sets_5 = select_tokens_iterative_softmax(
        layer_attn, 5, n_kv_heads, n_groups, n_q, ctx_len)

    for hi in range(n_kv_heads):
        keep = sorted(keep_sets_5[hi])
        n_keep = len(keep)
        n_sink = len([p for p in keep if p < 4])
        n_question = len([p for p in keep if p in set(q_pos)])
        n_needle = len([p for p in keep if p in needle_set])
        n_other = n_keep - n_sink - n_question - n_needle

        interesting = [p for p in keep if p >= 4 and p not in set(q_pos)]
        interesting_str = []
        for p in interesting[:30]:
            tok = tokenizer.decode(ctx_ids[0, p]).replace('\n', '\\n').strip()[:10]
            marker = "[NDL]" if p in needle_set else ""
            interesting_str.append(f"{p}:'{tok}'{marker}")

        print(f"\n  KV Head {hi}: keep {n_keep}/{ctx_len} (CR={1-n_keep/ctx_len:.1%})")
        print(f"    sink={n_sink}, question={n_question}, needle={n_needle}/{len(needle_pos)}, other={n_other}")
        print(f"    Interesting: {', '.join(interesting_str[:15])}")
        if len(interesting_str) > 15:
            print(f"                 {', '.join(interesting_str[15:30])}")

    # ═══════════════════════════════════════════
    # Eviction test: per-layer iterative softmax
    # ═══════════════════════════════════════════
    print(f"\n{'='*100}")
    print("Eviction test: per-layer iterative softmax (each layer uses its own attention)")
    print(f"{'='*100}")

    # Baseline
    _, cache_base, _ = compute_per_qtoken_attention(model, tokenizer, ctx_ids, target_key)
    answer = generate_with_cache(model, tokenizer, cache_base, suffix_ids)
    print(f"\n  Baseline: {answer[:60]}... Correct={'✓' if target_value in answer else '✗'}")
    del cache_base
    torch.cuda.empty_cache()

    for n_iter in [1, 2, 3, 5, 10, 20]:
        _, cache, _ = compute_per_qtoken_attention(model, tokenizer, ctx_ids, target_key)

        total_evicted = 0
        total_entries = 0

        for li in range(n_layers):
            layer_attn = all_attn[li]
            keep_sets = select_tokens_iterative_softmax(
                layer_attn, n_iter, n_kv_heads, n_groups, n_q, ctx_len)

            layer_obj = cache.layers[li]
            k_cache = layer_obj.keys
            v_cache = layer_obj.values

            for hi in range(n_kv_heads):
                evict_pos = [p for p in range(ctx_len) if p not in keep_sets[hi]]
                if evict_pos:
                    evict_t = torch.tensor(evict_pos, dtype=torch.long, device=k_cache.device)
                    k_cache[0, hi, evict_t, :] = 0
                    v_cache[0, hi, evict_t, :] = 0
                total_evicted += len(evict_pos)
                total_entries += ctx_len

        cr = total_evicted / total_entries
        answer = generate_with_cache(model, tokenizer, cache, suffix_ids)
        correct = target_value in answer
        print(f"  iter={n_iter:>2}: CR={cr:.1%}, Answer={answer[:60]}... Correct={'✓' if correct else '✗'}")
        del cache
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════
    # Eviction test: global attention from retrieval layers + iterative softmax
    # ═══════════════════════════════════════════
    print(f"\n{'='*100}")
    print("Eviction test: retrieval layers (L24,28,32) attention + iterative softmax for ALL layers")
    print(f"{'='*100}")

    retrieval_layers = [l for l in [24, 28, 32] if l < n_layers]
    agg_attn = all_attn[retrieval_layers].mean(dim=0)

    for n_iter in [1, 2, 3, 5, 10, 20]:
        keep_sets = select_tokens_iterative_softmax(
            agg_attn, n_iter, n_kv_heads, n_groups, n_q, ctx_len)

        mean_keep = np.mean([len(s) for s in keep_sets])
        cr = 1.0 - mean_keep / ctx_len

        _, cache, _ = compute_per_qtoken_attention(model, tokenizer, ctx_ids, target_key)
        for li in range(n_layers):
            layer_obj = cache.layers[li]
            k_cache = layer_obj.keys
            v_cache = layer_obj.values
            for hi in range(n_kv_heads):
                evict_pos = [p for p in range(ctx_len) if p not in keep_sets[hi]]
                if evict_pos:
                    evict_t = torch.tensor(evict_pos, dtype=torch.long, device=k_cache.device)
                    k_cache[0, hi, evict_t, :] = 0
                    v_cache[0, hi, evict_t, :] = 0

        answer = generate_with_cache(model, tokenizer, cache, suffix_ids)
        correct = target_value in answer
        print(f"  iter={n_iter:>2}: CR={cr:.1%}, keep~{mean_keep:.0f}/head, "
              f"Answer={answer[:50]}... Correct={'✓' if correct else '✗'}")
        del cache
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════
    # Eviction test: union of per-layer + global
    # (keep token if EITHER per-layer OR global says keep)
    # ═══════════════════════════════════════════
    print(f"\n{'='*100}")
    print("Eviction test: union(per-layer, global) iterative softmax")
    print(f"{'='*100}")

    for n_iter in [3, 5, 10]:
        # Global keep sets from retrieval layers
        global_keep = select_tokens_iterative_softmax(
            agg_attn, n_iter, n_kv_heads, n_groups, n_q, ctx_len)

        _, cache, _ = compute_per_qtoken_attention(model, tokenizer, ctx_ids, target_key)

        total_evicted = 0
        total_entries = 0

        for li in range(n_layers):
            layer_attn = all_attn[li]
            local_keep = select_tokens_iterative_softmax(
                layer_attn, n_iter, n_kv_heads, n_groups, n_q, ctx_len)

            layer_obj = cache.layers[li]
            k_cache = layer_obj.keys
            v_cache = layer_obj.values

            for hi in range(n_kv_heads):
                # Union of local and global
                combined_keep = local_keep[hi] | global_keep[hi]
                evict_pos = [p for p in range(ctx_len) if p not in combined_keep]
                if evict_pos:
                    evict_t = torch.tensor(evict_pos, dtype=torch.long, device=k_cache.device)
                    k_cache[0, hi, evict_t, :] = 0
                    v_cache[0, hi, evict_t, :] = 0
                total_evicted += len(evict_pos)
                total_entries += ctx_len

        cr = total_evicted / total_entries
        answer = generate_with_cache(model, tokenizer, cache, suffix_ids)
        correct = target_value in answer
        print(f"  iter={n_iter:>2}: CR={cr:.1%}, Answer={answer[:60]}... Correct={'✓' if correct else '✗'}")
        del cache
        torch.cuda.empty_cache()

    print("\nDONE")


if __name__ == "__main__":
    main()
