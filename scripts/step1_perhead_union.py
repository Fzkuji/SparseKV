#!/usr/bin/env python3
"""Per-head, cross-layer union eviction.

For each KV head:
  1. At each layer, select top-K tokens by Step1 score (K = keep_ratio * ctx_len)
  2. Union across all layers → keep set for this head
  3. Apply this keep set to ALL layers for this head

Test different keep ratios.
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
    suffix_ids = tokenizer.encode(q_suffix, return_tensors="pt",
                                  add_special_tokens=False).to(model.device)
    ctx_len = ctx_ids.shape[1]

    print(f"Context: {ctx_len} tokens")
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

    # Find key tokens
    key_str = "mystic-thunder"
    key_char = full_text_dec.find(f"numbers for {key_str} is:")
    key_pos = []
    if key_char >= 0:
        key_start = key_char + len("numbers for ")
        cum = ""
        for i in range(ctx_len):
            prev_len = len(cum)
            cum += tokenizer.decode(ctx_ids[0, i])
            if len(cum) > key_start and prev_len < key_start + len(key_str):
                key_pos.append(i)
    key_set = set(key_pos)

    print(f"Needle: {len(needle_pos)} tokens (pos {needle_pos[0]}-{needle_pos[-1]})")
    print(f"Key tokens: {key_pos}")

    # Find question positions
    question_str = f"What is the special magic number for {target_key} mentioned in the provided text?"
    q_char = full_text_dec.find(question_str)
    q_pos = []
    if q_char >= 0:
        cum = ""
        for i in range(ctx_len):
            prev_len = len(cum)
            cum += tokenizer.decode(ctx_ids[0, i])
            if len(cum) > q_char and prev_len < q_char + len(question_str):
                q_pos.append(i)
    q_pos_t = torch.tensor(q_pos, dtype=torch.long, device=model.device)
    n_q = len(q_pos)

    # ═══════════════════════════════════════════
    # Compute Step1 scores: [n_layers, n_kv_heads, ctx_len]
    # ═══════════════════════════════════════════
    print(f"\nComputing Step1 scores...")
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
                    k_h = k[0, hi]
                    head_scores = []
                    for gi in range(n_groups):
                        q_g = q_grouped[0, hi, gi, q_pos_t, :]
                        logits = torch.matmul(q_g, k_h.T) / scale
                        for qi in range(n_q):
                            logits[qi, q_pos[qi]+1:] = float('-inf')
                        attn = torch.softmax(logits.float(), dim=-1)
                        head_scores.append(attn)
                    stacked = torch.stack(head_scores, dim=0)
                    score = stacked.amax(dim=(0, 1))
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
    print("Step1 scores computed.")

    # ═══════════════════════════════════════════
    # Per-head, cross-layer union analysis
    # ═══════════════════════════════════════════
    keep_ratios = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    print(f"\n{'='*100}")
    print("Per-head cross-layer union: keep top-K per layer, union across layers")
    print(f"{'='*100}")

    for keep_ratio in keep_ratios:
        k = max(1, int(ctx_len * keep_ratio))

        # For each KV head, union of top-k across all layers
        head_keep_sets = []
        for hi in range(n_kv_heads):
            keep = set()
            for li in range(n_layers):
                scores = step1_scores[li, hi].numpy()
                topk_idx = np.argsort(scores)[::-1][:k]
                keep.update(topk_idx.tolist())
            head_keep_sets.append(keep)

        # Stats
        sizes = [len(s) for s in head_keep_sets]
        actual_cr = 1.0 - np.mean(sizes) / ctx_len

        # Needle coverage per head
        needle_per_head = [len(needle_set & head_keep_sets[hi]) for hi in range(n_kv_heads)]
        key_per_head = [len(key_set & head_keep_sets[hi]) for hi in range(n_kv_heads)]

        print(f"\n  keep_ratio={keep_ratio:.0%} (top-{k}/layer):")
        print(f"    Union sizes: {sizes}")
        print(f"    Actual CR: {actual_cr:.1%}")
        print(f"    Needle coverage: {needle_per_head} / {len(needle_pos)}")
        print(f"    Key coverage:    {key_per_head} / {len(key_pos)}")
        print(f"    All key kept?    {[k >= len(key_pos) for k in key_per_head]}")

    # ═══════════════════════════════════════════
    # Eviction test
    # ═══════════════════════════════════════════
    print(f"\n{'='*100}")
    print("Eviction test: per-head cross-layer union")
    print(f"{'='*100}")

    # Baseline
    cache_base = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=cache_base)
    answer = generate_with_cache(model, tokenizer, cache_base, suffix_ids)
    print(f"\n  Baseline: {answer[:60]}... Correct={'✓' if target_value in answer else '✗'}")
    del cache_base
    torch.cuda.empty_cache()

    for keep_ratio in keep_ratios:
        k = max(1, int(ctx_len * keep_ratio))

        # Compute keep sets
        head_keep_sets = []
        for hi in range(n_kv_heads):
            keep = set()
            for li in range(n_layers):
                scores = step1_scores[li, hi].numpy()
                topk_idx = np.argsort(scores)[::-1][:k]
                keep.update(topk_idx.tolist())
            head_keep_sets.append(keep)

        # Build fresh cache
        cache = DynamicCache()
        with torch.no_grad():
            model.model(input_ids=ctx_ids, past_key_values=cache)

        # Evict: zero out positions NOT in keep set, same mask for ALL layers
        total_evicted = 0
        total_entries = 0
        for li in range(n_layers):
            layer_obj = cache.layers[li]
            k_cache = layer_obj.keys
            v_cache = layer_obj.values
            for hi in range(n_kv_heads):
                evict_pos = [p for p in range(ctx_len) if p not in head_keep_sets[hi]]
                if evict_pos:
                    evict_t = torch.tensor(evict_pos, dtype=torch.long, device=k_cache.device)
                    k_cache[0, hi, evict_t, :] = 0
                    v_cache[0, hi, evict_t, :] = 0
                total_evicted += len(evict_pos)
                total_entries += ctx_len

        actual_cr = total_evicted / total_entries
        answer = generate_with_cache(model, tokenizer, cache, suffix_ids)
        correct = target_value in answer

        sizes = [len(s) for s in head_keep_sets]
        print(f"  top-{k:>3}/layer (ratio={keep_ratio:.0%}): actual_CR={actual_cr:.1%}, "
              f"union_sizes={sizes}, "
              f"Answer={answer[:50]}... {'✓' if correct else '✗'}")
        del cache
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════
    # Detail: at keep_ratio=0.05, what tokens does each head keep?
    # ═══════════════════════════════════════════
    print(f"\n{'='*100}")
    print("Detail: keep_ratio=5%, what does each head keep?")
    print(f"{'='*100}")

    k = max(1, int(ctx_len * 0.05))
    for hi in range(n_kv_heads):
        keep = set()
        for li in range(n_layers):
            scores = step1_scores[li, hi].numpy()
            topk_idx = np.argsort(scores)[::-1][:k]
            keep.update(topk_idx.tolist())

        keep_sorted = sorted(keep)
        n_sink = len([p for p in keep_sorted if p < 4])
        n_question = len([p for p in keep_sorted if p in set(q_pos)])
        n_needle = len([p for p in keep_sorted if p in needle_set])
        n_key = len([p for p in keep_sorted if p in key_set])
        n_other = len(keep_sorted) - n_sink - n_question - n_needle

        # Show interesting (non-question, non-sink) tokens
        interesting = [p for p in keep_sorted if p >= 4 and p not in set(q_pos)]
        tokens_str = []
        for p in interesting[:40]:
            tok = tokenizer.decode(ctx_ids[0, p]).replace('\n', '\\n').strip()[:8]
            marker = ""
            if p in key_set:
                marker = "[KEY]"
            elif p in needle_set:
                marker = "[NDL]"
            tokens_str.append(f"{p}:'{tok}'{marker}")

        print(f"\n  KV Head {hi}: union={len(keep_sorted)}/{ctx_len} (CR={1-len(keep_sorted)/ctx_len:.1%})")
        print(f"    sink={n_sink}, question={n_question}, needle={n_needle}(key={n_key}), other={n_other}")
        print(f"    Non-Q tokens: {', '.join(tokens_str[:20])}")
        if len(tokens_str) > 20:
            print(f"                  {', '.join(tokens_str[20:40])}")

    print("\nDONE")


if __name__ == "__main__":
    main()
