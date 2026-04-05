#!/usr/bin/env python3
"""Compare query-aware attention tracing vs KVzip scoring.

Key hypothesis: tracing from question tokens should give higher scores
to the TARGET needle, while KVzip treats all needles equally.
"""

import torch
import random
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.presses.base_press import BasePress
from kvpress.attention_patch import search_hyperplane

ADJECTIVES = ["silent", "golden", "dusty", "hollow", "bitter", "crimson",
              "gentle", "fierce", "mystic", "frozen", "swift", "ancient",
              "bright", "dark", "wild", "calm", "bold", "shy", "proud", "humble"]
NOUNS = ["rabbit", "hawk", "maple", "pine", "stream", "tide", "wolf",
         "eagle", "river", "stone", "flame", "shadow", "storm", "leaf",
         "crystal", "thunder", "ocean", "moon", "star", "wind"]


def build_prompt(num_distractors, target_key, target_value, needle_pos_frac, seed=50):
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
    prompt = (
        f"A special magic number is hidden within the following text. "
        f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {target_key} mentioned in the provided text?"
    )
    return prompt


def get_attention_scores(model, context_ids, question_ids):
    """Get attention from question tokens to context tokens.

    Returns: score_val [n_layers, 1, n_kv_heads, context_length]
    """
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    ctx_len = context_ids.shape[1]
    q_len = question_ids.shape[1]

    # Prefill context into cache
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=context_ids, past_key_values=cache)

    # Now feed question tokens with output_attentions=True
    position_ids = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        outputs = model.model(
            input_ids=question_ids,
            past_key_values=cache,
            position_ids=position_ids,
            output_attentions=True,
        )

    # outputs.attentions: tuple of [bsz, n_q_heads, q_len, kv_len] per layer
    # kv_len = ctx_len + q_len (context cache + question self-attention)
    # We only want attention to context tokens: [:, :, :, :ctx_len]

    score_val = torch.zeros(n_layers, 1, n_kv_heads, ctx_len,
                            dtype=model.dtype, device=model.device)

    for layer_idx, attn_weights in enumerate(outputs.attentions):
        # attn_weights: [1, n_q_heads, q_len, kv_len]
        # Extract attention to context tokens only
        attn_to_ctx = attn_weights[:, :, :, :ctx_len]  # [1, n_q_heads, q_len, ctx_len]

        # Group query heads by KV head
        attn_grouped = attn_to_ctx.view(1, n_kv_heads, n_groups, q_len, ctx_len)

        # Aggregate: max over query tokens and query heads within each group
        # This gives the maximum attention any question token pays to each context position
        scores = attn_grouped.amax(dim=(2, 3))  # [1, n_kv_heads, ctx_len]

        score_val[layer_idx] = scores

    del cache
    return score_val


def get_kvzip_scores(model, tokenizer, context_ids, compression_ratio=0.5):
    """Get KVzip reconstruction scores.

    Returns: score_val [n_layers, 1, n_kv_heads, context_length], cache
    """
    press = KVzipPress(compression_ratio=compression_ratio, layerwise=True)

    saved = {}
    original_compress = press.compress_post
    def patched_compress(model_arg):
        saved['score_val'] = press.score_val.clone()
        original_compress(model_arg)
    press.compress_post = patched_compress

    cache = DynamicCache()
    with torch.no_grad(), press(model):
        model.model(input_ids=context_ids, past_key_values=cache)

    return saved.get('score_val'), cache


def apply_fake_compression(model, score_val, cr, ctx_len):
    """Apply fake compression using score_val, same as KVzip's compress_post.

    Sets module.masked_key_indices on each attention layer.
    """
    n_layers, bsz, n_kv_heads, _ = score_val.shape

    # Per-layer uniform compression
    nl = int(bsz * n_kv_heads * ctx_len * cr)
    n_pruned_per_layer = nl

    for layer in model.model.layers:
        module = layer.self_attn
        layer_idx = int(module.layer_idx)
        scores = score_val[layer_idx]  # [1, n_kv_heads, ctx_len]

        n_pruned = n_pruned_per_layer
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten().cpu()

        batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
        head_indices = indices // ctx_len
        seq_indices = indices % ctx_len
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)


def clear_fake_compression(model):
    """Clear masked_key_indices from all layers."""
    for layer in model.model.layers:
        layer.self_attn.masked_key_indices = None


def generate_answer(model, tokenizer, cache, start_pos):
    """Generate answer from cache."""
    # Get first token
    # We need a dummy input to start generation
    # Use the last position in cache
    generated_ids = []
    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    cur_pos = start_pos
    # First token: get logits from the last position already in cache
    # We need to feed a dummy token to get the next prediction
    # Actually, we need the model to produce the first token
    # Let's use a small trick: the cache already has all context+question
    # We just need to sample from the last logits

    # Feed a dummy forward to get logits from the cache state
    # Actually we should have done this during question feeding
    # Let me restructure...
    return None  # placeholder


def run_test_with_scores(model, tokenizer, context_ids, question_ids, score_val, cr, label):
    """Run generation with given importance scores and compression ratio."""
    ctx_len = context_ids.shape[1]
    q_len = question_ids.shape[1]

    # Prefill context
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=context_ids, past_key_values=cache)

    # Apply fake compression based on scores
    apply_fake_compression(model, score_val, cr, ctx_len)

    # Feed question
    position_ids = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(
            input_ids=question_ids,
            past_key_values=cache,
            position_ids=position_ids,
            num_logits_to_keep=1,
        )

    # Greedy decode
    generated_ids = [outputs.logits[0, -1].argmax()]
    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    cur_pos = ctx_len + q_len
    for i in range(59):
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=torch.tensor([[cur_pos + i]], device=model.device),
            )
        next_id = outputs.logits[0, -1].argmax()
        generated_ids.append(next_id)
        if next_id.item() in eos_ids:
            break

    gen_text = tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)

    # Clear compression
    clear_fake_compression(model)
    del cache

    return gen_text


def analyze_scores(score_val, tokens, target_start, target_len, label):
    """Analyze how scores differ between target and other tokens."""
    n_layers, _, n_kv_heads, ctx_len = score_val.shape

    # Average across layers
    avg_score = score_val.mean(dim=(0, 1))  # [n_kv_heads, ctx_len]

    # Target region scores
    target_scores = avg_score[:, target_start:target_start+target_len]  # [n_kv_heads, target_len]
    # Non-target scores (excluding first 10 tokens as system/template)
    mask = torch.ones(ctx_len, dtype=torch.bool)
    mask[:10] = False  # system tokens
    mask[target_start:target_start+target_len] = False
    other_scores = avg_score[:, mask]  # [n_kv_heads, remaining]

    target_mean = target_scores.mean().item()
    other_mean = other_scores.mean().item()
    ratio = target_mean / max(other_mean, 1e-10)

    print(f"\n  [{label}] Score analysis:")
    print(f"    Target needle avg score:  {target_mean:.6f}")
    print(f"    Other tokens avg score:   {other_mean:.6f}")
    print(f"    Ratio (target/other):     {ratio:.4f}x")

    # Per KV head analysis for key heads
    for h in range(n_kv_heads):
        t_mean = target_scores[h].mean().item()
        o_mean = other_scores[h].mean().item()
        r = t_mean / max(o_mean, 1e-10)
        if r > 1.5 or h in [1, 2, 3, 5, 7]:
            print(f"    KV{h}: target={t_mean:.6f} other={o_mean:.6f} ratio={r:.2f}x")

    return ratio


def main():
    target_key = "mystic-thunder"
    target_value = "7156842"

    print("=" * 90)
    print("ATTENTION TRACING vs KVZIP: Query-aware vs Reconstruction scoring")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    # Use eager attention to get attention weights for tracing
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",  # needed for output_attentions
    )
    model.eval()

    # Build prompt
    prompt = build_prompt(30, target_key, target_value, 0.5, seed=50)

    # Tokenize with chat template
    separator = "#" * (len(prompt) + 10)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt + separator}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    context_text, question_suffix = full_text.split(separator)

    context_ids = tokenizer.encode(context_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    question_ids = tokenizer.encode(question_suffix, return_tensors="pt", add_special_tokens=False).to(model.device)
    ctx_len = context_ids.shape[1]
    q_len = question_ids.shape[1]

    tokens = [tokenizer.decode(context_ids[0, i]) for i in range(ctx_len)]

    print(f"\nContext: {ctx_len} tokens, Question: {q_len} tokens")

    # Find target needle position
    target_text = f"magic numbers for {target_key} is: {target_value}."
    target_token_ids = tokenizer.encode(target_text, add_special_tokens=False)
    target_start = None
    for i in range(ctx_len - len(target_token_ids)):
        if context_ids[0, i:i+len(target_token_ids)].tolist() == target_token_ids:
            target_start = i
            break
    if target_start is None:
        # Fallback: find "mystic-thunder"
        for i in range(ctx_len - 5):
            chunk = tokenizer.decode(context_ids[0, i:i+15])
            if "mystic" in chunk and "thunder" in chunk:
                target_start = i
                break
    target_len = len(target_token_ids) if target_start else 20
    print(f"Target needle at positions {target_start}-{target_start+target_len-1}")
    print(f"Target tokens: {tokens[target_start:target_start+target_len]}")

    # ═══════════════════════════════════════════════════════
    # Step 1: Get attention tracing scores
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("STEP 1: Computing attention tracing scores (query-aware)")
    print("=" * 90)

    tracing_scores = get_attention_scores(model, context_ids, question_ids)
    print(f"Tracing scores shape: {tracing_scores.shape}")
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # Step 2: Get KVzip scores
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("STEP 2: Computing KVzip scores (reconstruction)")
    print("=" * 90)

    kvzip_scores, _ = get_kvzip_scores(model, tokenizer, context_ids, compression_ratio=0.5)
    print(f"KVzip scores shape: {kvzip_scores.shape}")
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # Step 3: Compare score distributions
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("STEP 3: Score distribution comparison")
    print("=" * 90)

    tracing_ratio = analyze_scores(tracing_scores, tokens, target_start, target_len, "Tracing")
    kvzip_ratio = analyze_scores(kvzip_scores, tokens, target_start, target_len, "KVzip")

    print(f"\n  Summary: Tracing target/other ratio = {tracing_ratio:.4f}x, "
          f"KVzip target/other ratio = {kvzip_ratio:.4f}x")
    if tracing_ratio > kvzip_ratio:
        print(f"  → Tracing is {tracing_ratio/kvzip_ratio:.2f}x more selective for target!")
    else:
        print(f"  → KVzip is {kvzip_ratio/tracing_ratio:.2f}x more selective for target!")

    # ═══════════════════════════════════════════════════════
    # Step 4: Per-head target retention at various CRs
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("STEP 4: Target token retention at various CRs")
    print("=" * 90)

    n_layers, _, n_kv_heads, _ = tracing_scores.shape

    for cr in [0.5, 0.7, 0.9, 0.95]:
        print(f"\n  --- CR = {cr} ---")
        for method_name, scores in [("Tracing", tracing_scores), ("KVzip", kvzip_scores)]:
            total_target_kept = 0
            total_target_total = 0
            for layer_idx in [7, 9, 13, 15]:
                layer_scores = scores[layer_idx, 0]  # [n_heads, ctx_len]
                n_total = n_kv_heads * ctx_len
                n_pruned = int(n_total * cr)
                flat = layer_scores.reshape(-1)
                _, pruned_idx = torch.topk(-flat, n_pruned)
                kept_mask = torch.ones(n_total, dtype=torch.bool)
                kept_mask[pruned_idx] = False
                kept_mask = kept_mask.reshape(n_kv_heads, ctx_len)

                # Count target digits kept
                for h in range(n_kv_heads):
                    for offset in range(target_len):
                        pos = target_start + offset
                        tok = tokens[pos].strip()
                        if tok.isdigit() or tok in [target_key.split('-')[0], target_key.split('-')[1]]:
                            total_target_total += 1
                            if kept_mask[h, pos]:
                                total_target_kept += 1

            rate = total_target_kept / max(total_target_total, 1)
            print(f"    {method_name:8s}: target token retention = {total_target_kept}/{total_target_total} ({rate:.1%})")

    # ═══════════════════════════════════════════════════════
    # Step 5: End-to-end generation comparison
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("STEP 5: End-to-end generation comparison")
    print("=" * 90)

    for cr in [0.5, 0.7, 0.9, 0.95]:
        print(f"\n  --- CR = {cr} ---")

        # Tracing-based compression
        gen = run_test_with_scores(model, tokenizer, context_ids, question_ids,
                                   tracing_scores, cr, f"Tracing_cr{cr}")
        correct = target_value in gen
        print(f"    Tracing: {'OK' if correct else 'FAIL'}  output={gen[:80]}")
        torch.cuda.empty_cache()

        # KVzip-based compression
        gen = run_test_with_scores(model, tokenizer, context_ids, question_ids,
                                   kvzip_scores, cr, f"KVzip_cr{cr}")
        correct = target_value in gen
        print(f"    KVzip:   {'OK' if correct else 'FAIL'}  output={gen[:80]}")
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # Step 6: Harder test - 100 distractors, various CRs
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("STEP 6: Harder test (100 distractors, needle at 50%)")
    print("=" * 90)

    prompt2 = build_prompt(100, target_key, target_value, 0.5, seed=60)
    separator2 = "#" * (len(prompt2) + 10)
    full_text2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt2 + separator2}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    ctx_text2, q_suffix2 = full_text2.split(separator2)
    ctx_ids2 = tokenizer.encode(ctx_text2, return_tensors="pt", add_special_tokens=False).to(model.device)
    q_ids2 = tokenizer.encode(q_suffix2, return_tensors="pt", add_special_tokens=False).to(model.device)
    ctx_len2 = ctx_ids2.shape[1]

    print(f"  Context: {ctx_len2} tokens")

    # Tracing scores
    tracing_scores2 = get_attention_scores(model, ctx_ids2, q_ids2)
    torch.cuda.empty_cache()

    # KVzip scores
    kvzip_scores2, _ = get_kvzip_scores(model, tokenizer, ctx_ids2, compression_ratio=0.5)
    torch.cuda.empty_cache()

    # Find target
    target_text2 = f"magic numbers for {target_key} is: {target_value}."
    target_ids2 = tokenizer.encode(target_text2, add_special_tokens=False)
    target_start2 = None
    for i in range(ctx_len2 - len(target_ids2)):
        if ctx_ids2[0, i:i+len(target_ids2)].tolist() == target_ids2:
            target_start2 = i
            break
    tokens2 = [tokenizer.decode(ctx_ids2[0, i]) for i in range(ctx_len2)]

    if target_start2:
        analyze_scores(tracing_scores2, tokens2, target_start2, len(target_ids2), "Tracing (100 dist)")
        analyze_scores(kvzip_scores2, tokens2, target_start2, len(target_ids2), "KVzip (100 dist)")

    for cr in [0.5, 0.7, 0.9, 0.95]:
        print(f"\n  --- CR = {cr} ---")
        gen = run_test_with_scores(model, tokenizer, ctx_ids2, q_ids2,
                                   tracing_scores2, cr, f"Tracing")
        correct = target_value in gen
        print(f"    Tracing: {'OK' if correct else 'FAIL'}  output={gen[:80]}")
        torch.cuda.empty_cache()

        gen = run_test_with_scores(model, tokenizer, ctx_ids2, q_ids2,
                                   kvzip_scores2, cr, f"KVzip")
        correct = target_value in gen
        print(f"    KVzip:   {'OK' if correct else 'FAIL'}  output={gen[:80]}")
        torch.cuda.empty_cache()

    print("\n\nDONE")


if __name__ == "__main__":
    main()
