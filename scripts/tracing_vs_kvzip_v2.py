#!/usr/bin/env python3
"""Compare query-aware attention tracing vs KVzip scoring. V2.

Fix: Don't use eager attention. Instead:
1. Run KVzip normally (FlashAttention) to get reconstruction scores
2. Run a separate forward pass with hooks to compute Q*K attention for tracing
   (compute attention scores manually from Q, K without materializing full attn matrix)
"""

import torch
import random
import math
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half
from kvpress.presses.kvzip_press import KVzipPress
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


def compute_tracing_scores_via_hooks(model, context_ids, question_ids):
    """Compute attention tracing scores using forward hooks to manually compute Q*K.

    This works with FlashAttention because we intercept Q, K before the attention function.
    """
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = context_ids.shape[1]
    q_len = question_ids.shape[1]

    score_val = torch.zeros(n_layers, 1, n_kv_heads, ctx_len,
                            dtype=model.dtype, device=model.device)

    # Step 1: Prefill context into cache
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=context_ids, past_key_values=cache)

    # Step 2: Feed question tokens with hooks to capture Q*K attention
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # During question forward, we can access:
            # - The hidden states (to compute Q)
            # - The cache (to get K from context)
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return

            bsz, seq_len, _ = hidden_states.shape
            if seq_len != q_len:
                return  # skip if not the question forward pass

            past_kv = kwargs.get("past_key_values", None)
            if past_kv is None:
                return

            # Get K from cache (context part only, already has RoPE applied)
            k = past_kv.layers[layer_idx].keys[:, :, :ctx_len, :]
            # k: [bsz, n_kv_heads, ctx_len, head_dim]

            # Get Q with RoPE from position_embeddings (passed by the model)
            # The model passes position_embeddings=(cos, sin) in kwargs
            position_embeddings = kwargs.get("position_embeddings", None)

            # Compute Q from hidden states
            q_proj = module.q_proj(hidden_states)
            q = q_proj.view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)

            # Apply RoPE using position_embeddings from kwargs
            if position_embeddings is not None:
                cos, sin = position_embeddings
                q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
            else:
                # Fallback: use model's rotary_emb
                pos_ids = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
                cos, sin = model.model.rotary_emb(q, pos_ids)
                q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))

            # K already has RoPE from cache
            # Group Q by KV head and compute attention
            q_grouped = q.view(bsz, n_kv_heads, n_groups, q_len, head_dim)
            k_expanded = k.unsqueeze(2)  # [bsz, n_kv_heads, 1, ctx_len, head_dim]

            # [bsz, n_kv_heads, n_groups, q_len, ctx_len]
            attn_scores = torch.matmul(q_grouped, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # Aggregate: max over q_len and groups
            scores = attn_weights.amax(dim=(2, 3))  # [bsz, n_kv_heads, ctx_len]
            score_val[layer_idx] = scores

        return hook_fn

    for layer in model.model.layers:
        h = layer.self_attn.register_forward_hook(make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True)
        hooks.append(h)

    # Run question forward
    position_ids = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        model.model(
            input_ids=question_ids,
            past_key_values=cache,
            position_ids=position_ids,
        )

    for h in hooks:
        h.remove()

    del cache
    return score_val


def get_kvzip_scores(model, tokenizer, context_ids, compression_ratio=0.5):
    """Get KVzip reconstruction scores."""
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

    return saved.get('score_val')


def apply_fake_compression(model, score_val, cr, ctx_len):
    """Apply fake compression using score_val."""
    n_layers, bsz, n_kv_heads, _ = score_val.shape
    n_pruned_per_layer = int(bsz * n_kv_heads * ctx_len * cr)

    for layer in model.model.layers:
        module = layer.self_attn
        layer_idx = int(module.layer_idx)
        scores = score_val[layer_idx]
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned_per_layer, dim=1).indices.flatten().cpu()
        batch_indices = torch.arange(bsz).repeat_interleave(n_pruned_per_layer)
        head_indices = indices // ctx_len
        seq_indices = indices % ctx_len
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)


def clear_fake_compression(model):
    for layer in model.model.layers:
        layer.self_attn.masked_key_indices = None


def run_test_with_scores(model, tokenizer, context_ids, question_ids, score_val, cr, label):
    """Prefill → apply fake compression → feed question → generate."""
    ctx_len = context_ids.shape[1]
    q_len = question_ids.shape[1]

    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=context_ids, past_key_values=cache)

    apply_fake_compression(model, score_val, cr, ctx_len)

    position_ids = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(
            input_ids=question_ids,
            past_key_values=cache,
            position_ids=position_ids,
            num_logits_to_keep=1,
        )

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
    clear_fake_compression(model)
    del cache
    return gen_text


def analyze_scores(score_val, tokens, target_start, target_len, label, ctx_len):
    """Analyze target vs other token scores."""
    avg_score = score_val.mean(dim=(0, 1))  # [n_kv_heads, ctx_len]
    n_kv_heads = avg_score.shape[0]

    target_scores = avg_score[:, target_start:target_start+target_len]
    mask = torch.ones(ctx_len, dtype=torch.bool)
    mask[:10] = False
    mask[target_start:target_start+target_len] = False
    other_scores = avg_score[:, mask]

    target_mean = target_scores.mean().item()
    other_mean = other_scores.mean().item()
    ratio = target_mean / max(other_mean, 1e-10)

    print(f"\n  [{label}]")
    print(f"    Target needle avg score:  {target_mean:.6f}")
    print(f"    Other tokens avg score:   {other_mean:.6f}")
    print(f"    Ratio (target/other):     {ratio:.4f}x")

    for h in range(n_kv_heads):
        t_mean = target_scores[h].mean().item()
        o_mean = other_scores[h].mean().item()
        r = t_mean / max(o_mean, 1e-10)
        if r > 2.0 or h in [1, 2, 3, 5, 7]:
            print(f"      KV{h}: target={t_mean:.6f} other={o_mean:.6f} ratio={r:.2f}x")

    return ratio


def count_target_retention(score_val, target_start, target_len, tokens, cr, ctx_len):
    """Count how many target tokens are retained at given CR."""
    n_layers, _, n_kv_heads, _ = score_val.shape
    total_kept = 0
    total_possible = 0

    for layer_idx in range(n_layers):
        layer_scores = score_val[layer_idx, 0]
        n_total = n_kv_heads * ctx_len
        n_pruned = int(n_total * cr)
        flat = layer_scores.reshape(-1)
        _, pruned_idx = torch.topk(-flat, n_pruned)
        kept_mask = torch.ones(n_total, dtype=torch.bool)
        kept_mask[pruned_idx] = False
        kept_mask = kept_mask.reshape(n_kv_heads, ctx_len)

        for h in range(n_kv_heads):
            for offset in range(target_len):
                pos = target_start + offset
                if pos < ctx_len:
                    total_possible += 1
                    if kept_mask[h, pos]:
                        total_kept += 1

    return total_kept, total_possible


def main():
    target_key = "mystic-thunder"
    target_value = "7156842"

    print("=" * 90)
    print("ATTENTION TRACING vs KVZIP V2")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    for n_dist, seed in [(30, 50), (100, 60)]:
        print(f"\n{'=' * 90}")
        print(f"TEST: {n_dist} distractors, needle at 50%")
        print(f"{'=' * 90}")

        prompt = build_prompt(n_dist, target_key, target_value, 0.5, seed=seed)

        separator = "#" * (len(prompt) + 10)
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        ctx_text, q_suffix = full_text.split(separator)
        ctx_ids = tokenizer.encode(ctx_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        q_ids = tokenizer.encode(q_suffix, return_tensors="pt", add_special_tokens=False).to(model.device)
        ctx_len = ctx_ids.shape[1]
        q_len = q_ids.shape[1]
        tokens = [tokenizer.decode(ctx_ids[0, i]) for i in range(ctx_len)]

        print(f"  Context: {ctx_len} tokens, Question: {q_len} tokens")

        # Find target by searching for target_value digits in decoded tokens
        target_start = None
        target_len = 15  # default
        # Search for the value "7156842" in the token stream
        for i in range(ctx_len - 20):
            window = tokenizer.decode(ctx_ids[0, i:i+25])
            if target_key in window and target_value in window:
                # Found the needle region, now find exact start
                # Look for "for mystic" pattern
                for j in range(max(0, i-5), min(ctx_len-5, i+10)):
                    tok = tokenizer.decode(ctx_ids[0, j:j+3])
                    if "for" in tok and "myst" in tokenizer.decode(ctx_ids[0, j:j+8]):
                        target_start = j
                        # Find end (after the value digits)
                        for end in range(j+5, min(ctx_len, j+25)):
                            tok_end = tokens[end].strip()
                            if tok_end == '.' or tok_end == '\n':
                                target_len = end - j + 1
                                break
                        break
                break
        if target_start is None:
            # Fallback: search for value digits
            for i in range(ctx_len - 5):
                window = tokenizer.decode(ctx_ids[0, i:i+10])
                if target_value[:4] in window:
                    target_start = max(0, i - 10)
                    target_len = 20
                    break
        if target_start:
            print(f"  Target at pos {target_start}-{target_start+target_len-1}")
            print(f"  Target tokens: {tokens[target_start:target_start+target_len]}")
        else:
            print(f"  WARNING: Target not found, skipping detailed analysis")

        # ─── Tracing scores ───
        print(f"\n  Computing tracing scores...")
        tracing_scores = compute_tracing_scores_via_hooks(model, ctx_ids, q_ids)
        print(f"  Tracing scores shape: {tracing_scores.shape}")
        torch.cuda.empty_cache()

        # ─── KVzip scores ───
        print(f"  Computing KVzip scores...")
        kvzip_scores = get_kvzip_scores(model, tokenizer, ctx_ids, compression_ratio=0.5)
        print(f"  KVzip scores shape: {kvzip_scores.shape}")
        torch.cuda.empty_cache()

        # ─── Score analysis ───
        print(f"\n  Score distribution:")
        if target_start:
            t_ratio = analyze_scores(tracing_scores, tokens, target_start, target_len, "Tracing", ctx_len)
            k_ratio = analyze_scores(kvzip_scores, tokens, target_start, target_len, "KVzip", ctx_len)
            print(f"\n  → Tracing selectivity: {t_ratio:.4f}x, KVzip selectivity: {k_ratio:.4f}x")
            if t_ratio > k_ratio:
                print(f"  → Tracing is {t_ratio/k_ratio:.2f}x more selective for target!")

        # ─── Target retention ───
        if target_start:
            print(f"\n  Target token retention:")
            for cr in [0.5, 0.7, 0.9, 0.95]:
                t_kept, t_total = count_target_retention(tracing_scores, target_start, target_len, tokens, cr, ctx_len)
                k_kept, k_total = count_target_retention(kvzip_scores, target_start, target_len, tokens, cr, ctx_len)
                print(f"    cr={cr}: Tracing={t_kept}/{t_total} ({t_kept/t_total:.1%})  "
                      f"KVzip={k_kept}/{k_total} ({k_kept/k_total:.1%})")

        # ─── Generation comparison ───
        print(f"\n  Generation comparison:")
        for cr in [0.5, 0.7, 0.9, 0.95]:
            # Full KV baseline (no compression)
            if cr == 0.5:
                gen = run_test_with_scores(model, tokenizer, ctx_ids, q_ids,
                                           torch.ones_like(tracing_scores), 0.0, "FullKV")
                ok = target_value in gen
                print(f"    FullKV:       {'OK' if ok else 'FAIL'}  {gen[:70]}")
                torch.cuda.empty_cache()

            gen = run_test_with_scores(model, tokenizer, ctx_ids, q_ids,
                                       tracing_scores, cr, f"Tracing_cr{cr}")
            ok_t = target_value in gen
            print(f"    Tracing cr={cr}: {'OK' if ok_t else 'FAIL'}  {gen[:70]}")
            torch.cuda.empty_cache()

            gen = run_test_with_scores(model, tokenizer, ctx_ids, q_ids,
                                       kvzip_scores, cr, f"KVzip_cr{cr}")
            ok_k = target_value in gen
            print(f"    KVzip   cr={cr}: {'OK' if ok_k else 'FAIL'}  {gen[:70]}")
            torch.cuda.empty_cache()

    print("\n\nDONE")


if __name__ == "__main__":
    main()
