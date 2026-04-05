#!/usr/bin/env python3
"""Attention tracing using context-internal question text as seed.

Instead of gen_prompt → ctx attention (which has no semantic info),
use question_text_in_ctx → earlier_ctx attention from prefill.

The question text "What is the special magic number for mystic-thunder..."
is at the END of the context. During prefill, these tokens attend to all
earlier tokens through causal self-attention.
"""

import torch
import random
import math
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half
from kvpress.attention_patch import patch_attention_functions

# Patch attention functions to support masked_key_indices (fake key compression)
patch_attention_functions()

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
    return (
        f"A special magic number is hidden within the following text. "
        f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {target_key} mentioned in the provided text?"
    )


def find_positions(tokenizer, ctx_ids, ctx_len, target_key, target_value):
    """Find question text and needle positions using full decode + char mapping."""
    tokens = [tokenizer.decode(ctx_ids[0, i]) for i in range(ctx_len)]

    # Build char→token mapping by cumulative decoding
    full_text = tokenizer.decode(ctx_ids[0])
    char_to_tok = []
    cum_len = 0
    for i in range(ctx_len):
        tok_text = tokenizer.decode(ctx_ids[0, :i+1])
        new_len = len(tok_text)
        for _ in range(new_len - cum_len):
            char_to_tok.append(i)
        cum_len = new_len

    # Find target needle in full text
    # Search for the needle pattern with key AND value
    needle_pattern = f"for {target_key} is: {target_value}."
    needle_char_pos = full_text.find(needle_pattern)
    if needle_char_pos < 0:
        print(f"  FAILED: could not find '{needle_pattern}' in decoded text!")
        return None, tokens

    # Find "One of the special" before the needle
    one_pos = full_text.rfind("One of the special", 0, needle_char_pos)
    if one_pos < 0:
        one_pos = max(0, needle_char_pos - 50)

    # Find period after value
    period_pos = full_text.find(".", needle_char_pos + len(needle_pattern) - 2)
    if period_pos < 0:
        period_pos = needle_char_pos + len(needle_pattern)

    # Map char positions to token positions
    needle_start = char_to_tok[one_pos] if one_pos < len(char_to_tok) else 0
    needle_end = char_to_tok[min(period_pos, len(char_to_tok)-1)]

    # Find key token positions
    key_char_start = full_text.find(target_key, needle_char_pos - 30)
    key_char_end = key_char_start + len(target_key) - 1
    key_tok_start = char_to_tok[key_char_start] if key_char_start >= 0 and key_char_start < len(char_to_tok) else None
    key_tok_end = char_to_tok[min(key_char_end, len(char_to_tok)-1)] if key_char_start >= 0 else None
    key_pos = list(range(key_tok_start, key_tok_end + 1)) if key_tok_start is not None else []

    # Find value token positions
    val_char_start = full_text.find(target_value, needle_char_pos)
    val_char_end = val_char_start + len(target_value) - 1
    val_tok_start = char_to_tok[val_char_start] if val_char_start >= 0 and val_char_start < len(char_to_tok) else None
    val_tok_end = char_to_tok[min(val_char_end, len(char_to_tok)-1)] if val_char_start >= 0 else None
    val_pos = list(range(val_tok_start, val_tok_end + 1)) if val_tok_start is not None else []

    # Find question text at end
    q_char_start = full_text.rfind("What is the special")
    if q_char_start < 0:
        q_char_start = full_text.rfind("What is")
    q_start = char_to_tok[q_char_start] if q_char_start >= 0 and q_char_start < len(char_to_tok) else None
    q_end = ctx_len - 1

    needle_sent = list(range(needle_start, needle_end + 1))
    q_text = list(range(q_start, q_end + 1)) if q_start else []

    sent_str = tokenizer.decode(ctx_ids[0, needle_start:needle_end+1])
    print(f"  Needle: '{sent_str.strip()}' [{needle_start}-{needle_end}]")
    for i in range(needle_start, needle_end + 1):
        role = ""
        if i in key_pos: role = " <-- KEY"
        if i in val_pos: role = " <-- VALUE"
        print(f"    [{i:4d}] '{tokens[i]}'{role}")
    print(f"    Key: {key_pos} = {[tokens[i] for i in key_pos]}")
    print(f"    Value: {val_pos} = {[tokens[i] for i in val_pos]}")
    q_str = tokenizer.decode(ctx_ids[0, q_start:q_end+1]) if q_start else "NOT FOUND"
    print(f"  Question text: '{q_str.strip()[:80]}' [{q_start}-{q_end}]")

    return {
        'needle_sentence': needle_sent,
        'needle_key': key_pos,
        'needle_value': val_pos,
        'question_text': q_text,
        'sink': list(range(min(5, ctx_len))),
    }, tokens


def compute_ctx_query_scores(model, context_ids, question_ids, q_text_positions, mode="seed_only", alpha=1.0):
    """Compute importance scores using question_text_in_ctx as seed.

    mode:
      "seed_only": just use Q_text → ctx attention
      "graphexp": seed + same-layer expansion
    """
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = context_ids.shape[1]
    q_len = question_ids.shape[1]

    need_prefill = True  # always need prefill for ctx-query approach
    prefill_attn = {}
    phase = {'current': 'prefill'}

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return
            bsz, seq_len, _ = hidden_states.shape
            position_embeddings = kwargs.get("position_embeddings", None)

            with torch.no_grad():
                if phase['current'] == 'prefill' and seq_len == ctx_len:
                    q = module.q_proj(hidden_states).view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                    k = module.k_proj(hidden_states).view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                        q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                        k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
                    q_g = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                    k_e = k.unsqueeze(2)
                    attn = torch.matmul(q_g, k_e.transpose(-2, -1)) / math.sqrt(head_dim)
                    causal = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=attn.device), diagonal=1)
                    attn = torch.softmax(attn + causal[None, None, None], dim=-1)
                    # Max over query groups: [n_kv_heads, ctx_len, ctx_len]
                    prefill_attn[layer_idx] = attn.amax(dim=2)[0].float().cpu()
        return hook_fn

    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.self_attn.register_forward_hook(
            make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True))

    phase['current'] = 'prefill'
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=context_ids, past_key_values=cache)

    for h in hooks:
        h.remove()

    # Build scores using question_text positions as seed
    q_pos = torch.tensor(q_text_positions)
    final_scores = torch.zeros(n_layers, 1, n_kv_heads, ctx_len,
                               dtype=torch.float32, device='cpu')

    for l in range(n_layers):
        if l not in prefill_attn:
            continue
        pa = prefill_attn[l]  # [n_kv_heads, ctx_len, ctx_len]

        # Seed: attention FROM question_text positions TO all context positions
        # [n_kv_heads, len(q_pos), ctx_len] → max over q_pos → [n_kv_heads, ctx_len]
        seed = pa[:, q_pos, :].amax(dim=1)

        if mode == "graphexp":
            outgoing = torch.bmm(seed.unsqueeze(1), pa).squeeze(1)
            incoming = torch.bmm(pa, seed.unsqueeze(2)).squeeze(2)
            final_scores[l, 0] = seed + alpha * (outgoing + incoming)
        else:
            final_scores[l, 0] = seed

    del prefill_attn
    final_scores = final_scores.to(dtype=model.dtype, device=model.device)
    return final_scores, cache


def apply_compression(model, score_val, cr, ctx_len):
    """Apply fake compression using kvpress's masked_key_indices.

    Uses kvpress's attention_patch (already patched at import) which replaces
    masked keys with fake keys such that exp(<q, k>) ≈ 0.
    """
    n_layers, bsz, n_kv_heads, _ = score_val.shape
    n_total = n_kv_heads * ctx_len
    n_pruned = int(n_total * cr)

    for layer in model.model.layers:
        module = layer.self_attn
        li = int(module.layer_idx)
        scores = score_val[li]  # [1, n_kv_heads, ctx_len]
        flat_scores = scores.reshape(bsz, -1)  # [1, n_kv_heads * ctx_len]
        _, prune_indices = torch.topk(-flat_scores, n_pruned, dim=1)
        prune_indices = prune_indices[0]  # [n_pruned]

        hi = prune_indices // ctx_len
        si = prune_indices % ctx_len
        bi = torch.zeros_like(hi)

        module.masked_key_indices = (bi, hi, si)


def clear_compression(model):
    for layer in model.model.layers:
        layer.self_attn.masked_key_indices = None


def generate(model, tokenizer, ctx_ids, q_ids, score_val, cr):
    ctx_len = ctx_ids.shape[1]
    q_len = q_ids.shape[1]
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=cache)
    if cr > 0:
        apply_compression(model, score_val, cr, ctx_len)
    pos = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=q_ids, past_key_values=cache, position_ids=pos, num_logits_to_keep=1)
    gen = [out.logits[0, -1].argmax()]
    eos = model.generation_config.eos_token_id
    if not isinstance(eos, list):
        eos = [eos]
    cp = ctx_len + q_len
    for i in range(59):
        with torch.no_grad():
            out = model(input_ids=gen[-1].unsqueeze(0).unsqueeze(0),
                       past_key_values=cache,
                       position_ids=torch.tensor([[cp + i]], device=model.device))
        nxt = out.logits[0, -1].argmax()
        gen.append(nxt)
        if nxt.item() in eos:
            break
    text = tokenizer.decode(torch.stack(gen), skip_special_tokens=True)
    clear_compression(model)
    del cache
    return text


def analyze_scores(score_val, groups, ctx_len, label):
    """Analyze score selectivity."""
    ns = groups['needle_sentence']
    nk = groups['needle_key']
    nv = groups['needle_value']

    if not ns:
        return 0

    avg = score_val.float().mean(dim=(0, 1))  # [n_kv_heads, ctx_len]

    target_mask = torch.zeros(ctx_len, dtype=torch.bool)
    for i in ns:
        target_mask[i] = True
    other_mask = torch.ones(ctx_len, dtype=torch.bool)
    other_mask[:10] = False
    for i in ns:
        other_mask[i] = False
    for i in groups.get('question_text', []):
        other_mask[i] = False

    t_mean = avg[:, target_mask].mean().item()
    o_mean = avg[:, other_mask].mean().item()
    ratio = t_mean / max(o_mean, 1e-10)

    # Also check key vs value parts
    k_mean = avg[:, nk].mean().item() if nk else 0
    v_mean = avg[:, nv].mean().item() if nv else 0

    print(f"    {label:25s}: target={t_mean:.6f} other={o_mean:.6f} ratio={ratio:.3f}x "
          f"(key={k_mean:.6f} val={v_mean:.6f})")
    return ratio


def main():
    model_path = "Qwen/Qwen3-8B"
    target_key = "mystic-thunder"
    target_value = "7156842"
    n_dist = 30
    seed = 50

    print("=" * 100)
    print("CONTEXT-QUERY ATTENTION TRACING")
    print("  Using question_text_in_ctx as seed instead of gen_prompt tokens")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    prompt = build_prompt(n_dist, target_key, target_value, 0.5, seed=seed)
    separator = "#" * (len(prompt) + 10)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt + separator}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    ctx_text, q_suffix = full_text.split(separator)
    ctx_ids = tokenizer.encode(ctx_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    q_ids = tokenizer.encode(q_suffix, return_tensors="pt", add_special_tokens=False).to(model.device)
    ctx_len = ctx_ids.shape[1]

    print(f"\n  Context: {ctx_len} tokens, Gen prompt: {q_ids.shape[1]} tokens")

    groups, tokens = find_positions(tokenizer, ctx_ids, ctx_len, target_key, target_value)
    q_text_pos = groups['question_text']

    if not q_text_pos:
        print("  FAILED: could not find question text in context!")
        return

    # =========================================================================
    # Compute scores with different methods
    # =========================================================================
    print(f"\n  Computing ctx-query seed scores...")
    seed_scores, _ = compute_ctx_query_scores(model, ctx_ids, q_ids, q_text_pos, mode="seed_only")
    torch.cuda.empty_cache()

    print(f"  Computing ctx-query graphexp scores (alpha=1.0)...")
    ge_scores, _ = compute_ctx_query_scores(model, ctx_ids, q_ids, q_text_pos, mode="graphexp", alpha=1.0)
    torch.cuda.empty_cache()

    # Also compute old gen-prompt based scores for comparison
    print(f"  Computing old gen-prompt scores for comparison...")
    old_scores = compute_old_scores(model, ctx_ids, q_ids)
    torch.cuda.empty_cache()

    # =========================================================================
    # Score analysis
    # =========================================================================
    print(f"\n  Score selectivity:")
    analyze_scores(old_scores, groups, ctx_len, "Old (gen-prompt seed)")
    analyze_scores(seed_scores, groups, ctx_len, "Ctx-query seed")
    analyze_scores(ge_scores, groups, ctx_len, "Ctx-query + GraphExp")

    # Per-layer analysis for ctx-query
    print(f"\n  Per-layer ctx-query seed selectivity:")
    ns = groups['needle_sentence']
    qt = groups['question_text']
    for l in range(0, model.config.num_hidden_layers, 4):
        layer_scores = seed_scores[l, 0]  # [n_kv_heads, ctx_len]
        target_mask = torch.zeros(ctx_len, dtype=torch.bool)
        for i in ns: target_mask[i] = True
        other_mask = torch.ones(ctx_len, dtype=torch.bool)
        other_mask[:10] = False
        for i in ns: other_mask[i] = False
        for i in qt: other_mask[i] = False

        t = layer_scores[:, target_mask].mean().item()
        o = layer_scores[:, other_mask].mean().item()
        ratio = t / max(o, 1e-10)

        # Top tokens for this layer
        avg = layer_scores.mean(dim=0)
        top_vals, top_idxs = avg.topk(8)
        top_info = [(idx.item(), tokens[idx.item()].strip()[:12], f"{top_vals[j].item():.4f}")
                    for j, idx in enumerate(top_idxs)]
        in_needle = "Y" if any(idx.item() in ns for idx in top_idxs[:5]) else "N"
        print(f"    L{l:2d}: t={t:.5f} o={o:.5f} ratio={ratio:.2f}x  needle_in_top5={in_needle}  top={top_info[:5]}")

    # =========================================================================
    # Generation tests
    # =========================================================================
    print(f"\n  Generation tests:")
    results = {}

    for cr in [0.0, 0.3, 0.5, 0.7, 0.9]:
        print(f"\n    CR = {cr}:")

        if cr == 0.0:
            gen = generate(model, tokenizer, ctx_ids, q_ids, seed_scores, 0.0)
            ok = target_value in gen
            print(f"      {'Full KV':25s}: {'OK' if ok else 'FAIL'}  {gen[:65]}")
            results[('FullKV', 0.0)] = ok
            torch.cuda.empty_cache()
            continue

        for label, scores in [("Old gen-prompt", old_scores),
                               ("Ctx-query seed", seed_scores),
                               ("Ctx-query GraphExp", ge_scores)]:
            gen = generate(model, tokenizer, ctx_ids, q_ids, scores, cr)
            ok = target_value in gen
            print(f"      {label:25s}: {'OK' if ok else 'FAIL'}  {gen[:65]}")
            results[(label, cr)] = ok
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Method':<25s} {'CR':>5s}  Result")
    print(f"  {'-'*25} {'-'*5}  {'-'*6}")
    for key in sorted(results.keys(), key=lambda x: (x[1], x[0])):
        label, cr = key
        print(f"  {label:<25s} {cr:>5.1f}  {'OK' if results[key] else 'FAIL'}")

    print("\nDONE")


def compute_old_scores(model, context_ids, question_ids):
    """Old method: gen-prompt → context attention as seed."""
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = context_ids.shape[1]
    q_len = question_ids.shape[1]

    question_attn = {}
    phase = {'current': 'prefill'}

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return
            bsz, seq_len, _ = hidden_states.shape
            position_embeddings = kwargs.get("position_embeddings", None)

            with torch.no_grad():
                if phase['current'] == 'question' and seq_len == q_len:
                    past_kv = kwargs.get("past_key_values", None)
                    if past_kv is None:
                        return
                    k = past_kv.layers[layer_idx].keys[:, :, :ctx_len, :]
                    q = module.q_proj(hidden_states).view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                        q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                    q_g = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                    k_e = k.unsqueeze(2)
                    attn = torch.softmax(torch.matmul(q_g, k_e.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1)
                    question_attn[layer_idx] = attn.amax(dim=2)[0].float().cpu()
        return hook_fn

    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.self_attn.register_forward_hook(
            make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True))

    phase['current'] = 'prefill'
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=context_ids, past_key_values=cache)

    phase['current'] = 'question'
    pos_ids = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        model.model(input_ids=question_ids, past_key_values=cache, position_ids=pos_ids)

    for h in hooks:
        h.remove()
    del cache

    final_scores = torch.zeros(n_layers, 1, n_kv_heads, ctx_len,
                               dtype=torch.float32, device='cpu')
    for l in range(n_layers):
        if l in question_attn:
            final_scores[l, 0] = question_attn[l].amax(dim=1)
    del question_attn
    final_scores = final_scores.to(dtype=model.dtype, device=model.device)
    return final_scores


if __name__ == "__main__":
    main()
