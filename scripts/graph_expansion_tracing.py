#!/usr/bin/env python3
"""Graph expansion tracing: per-layer attention neighborhood expansion.

For each layer:
1. question → context attention as seed importance
2. Expand using same-layer prefill attention graph:
   - outgoing: seed positions attended to what during prefill?
   - incoming: who attended to seed positions during prefill?
3. Combine as final importance for that layer
"""

import torch
import random
import math
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half
from kvpress.presses.kvzip_press import KVzipPress


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


def compute_graph_expansion_scores(model, context_ids, question_ids, expansion_alpha=1.0):
    """Per-layer graph expansion tracing.

    For each layer:
    1. seed = question→context attention (max over question tokens)
    2. outgoing = seed^T @ prefill_attn (what seed attended to during prefill)
    3. incoming = prefill_attn^T @ seed (who attended to seed during prefill)
    4. score = seed + alpha * (outgoing + incoming)
    """
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = context_ids.shape[1]
    q_len = question_ids.shape[1]

    # Final scores
    final_scores = torch.zeros(n_layers, 1, n_kv_heads, ctx_len,
                               dtype=torch.float32, device='cpu')

    # We'll compute prefill attention and question attention per layer in hooks
    # and do the expansion immediately to avoid storing all prefill attention at once
    # But for simplicity let's store everything first

    prefill_attn_storage = {}  # layer_idx -> [n_kv_heads, ctx_len, ctx_len]
    question_attn_storage = {}  # layer_idx -> [n_kv_heads, q_len, ctx_len]
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
                    # Compute Q and K for prefill self-attention
                    q = module.q_proj(hidden_states)
                    q = q.view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                    k = module.k_proj(hidden_states)
                    k = k.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)

                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                        q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                        k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))

                    q_grouped = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                    k_expanded = k.unsqueeze(2)
                    attn = torch.matmul(q_grouped, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)

                    causal = torch.triu(torch.full((seq_len, seq_len), float('-inf'),
                                                   device=attn.device), diagonal=1)
                    attn = attn + causal[None, None, None, :, :]
                    attn = torch.softmax(attn, dim=-1)
                    attn_grouped = attn.amax(dim=2)  # [bsz, n_kv_heads, ctx_len, ctx_len]
                    prefill_attn_storage[layer_idx] = attn_grouped[0].float().cpu()

                elif phase['current'] == 'question' and seq_len == q_len:
                    past_kv = kwargs.get("past_key_values", None)
                    if past_kv is None:
                        return
                    k = past_kv.layers[layer_idx].keys[:, :, :ctx_len, :]

                    q = module.q_proj(hidden_states)
                    q = q.view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                        q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))

                    q_grouped = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                    k_expanded = k.unsqueeze(2)
                    attn = torch.matmul(q_grouped, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
                    attn = torch.softmax(attn, dim=-1)
                    attn_grouped = attn.amax(dim=2)  # [bsz, n_kv_heads, q_len, ctx_len]
                    question_attn_storage[layer_idx] = attn_grouped[0].float().cpu()

        return hook_fn

    hooks = []
    for layer in model.model.layers:
        h = layer.self_attn.register_forward_hook(
            make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True)
        hooks.append(h)

    # Phase 1: Prefill
    phase['current'] = 'prefill'
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=context_ids, past_key_values=cache)

    # Phase 2: Question
    phase['current'] = 'question'
    position_ids = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        model.model(input_ids=question_ids, past_key_values=cache, position_ids=position_ids)

    for h in hooks:
        h.remove()

    # Phase 3: Per-layer graph expansion
    for l in range(n_layers):
        if l not in question_attn_storage or l not in prefill_attn_storage:
            continue

        q_attn = question_attn_storage[l]  # [n_kv_heads, q_len, ctx_len]
        p_attn = prefill_attn_storage[l]   # [n_kv_heads, ctx_len, ctx_len]

        # Seed: max over question tokens
        seed = q_attn.amax(dim=1)  # [n_kv_heads, ctx_len]

        # Outgoing: what did seed positions attend to during prefill?
        # outgoing[h, j] = sum_i seed[h, i] * p_attn[h, i, j]
        outgoing = torch.bmm(seed.unsqueeze(1), p_attn).squeeze(1)  # [n_kv_heads, ctx_len]

        # Incoming: who attended to seed positions during prefill?
        # incoming[h, i] = sum_j seed[h, j] * p_attn[h, i, j]
        # = (p_attn @ seed^T)  where p_attn[h, i, j] * seed[h, j] summed over j
        incoming = torch.bmm(p_attn, seed.unsqueeze(2)).squeeze(2)  # [n_kv_heads, ctx_len]

        # Combine
        expanded = seed + expansion_alpha * (outgoing + incoming)
        final_scores[l, 0] = expanded

    # Clean up
    del prefill_attn_storage, question_attn_storage

    final_scores = final_scores.to(dtype=model.dtype, device=model.device)
    return final_scores, cache


def compute_singlehop_scores(model, context_ids, question_ids):
    """Single-hop baseline: question→context attention only."""
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = context_ids.shape[1]
    q_len = question_ids.shape[1]

    score_val = torch.zeros(n_layers, 1, n_kv_heads, ctx_len,
                            dtype=model.dtype, device=model.device)

    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=context_ids, past_key_values=cache)

    hooks = []
    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return
            bsz, seq_len, _ = hidden_states.shape
            if seq_len != q_len:
                return
            position_embeddings = kwargs.get("position_embeddings", None)
            past_kv = kwargs.get("past_key_values", None)
            if past_kv is None:
                return
            with torch.no_grad():
                k = past_kv.layers[layer_idx].keys[:, :, :ctx_len, :]
                q = module.q_proj(hidden_states)
                q = q.view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                q_grouped = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                k_expanded = k.unsqueeze(2)
                attn = torch.matmul(q_grouped, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
                attn = torch.softmax(attn, dim=-1)
                scores = attn.amax(dim=(2, 3))
                score_val[layer_idx] = scores
        return hook_fn

    for layer in model.model.layers:
        h = layer.self_attn.register_forward_hook(make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True)
        hooks.append(h)

    position_ids = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        model.model(input_ids=question_ids, past_key_values=cache, position_ids=position_ids)
    for h in hooks:
        h.remove()
    return score_val, cache


def get_kvzip_scores(model, tokenizer, context_ids):
    press = KVzipPress(compression_ratio=0.5, layerwise=True)
    saved = {}
    orig = press.compress_post
    def patched(m):
        saved['score_val'] = press.score_val.clone()
        orig(m)
    press.compress_post = patched
    cache = DynamicCache()
    with torch.no_grad(), press(model):
        model.model(input_ids=context_ids, past_key_values=cache)
    return saved.get('score_val')


def apply_fake_compression(model, score_val, cr, ctx_len):
    n_layers, bsz, n_kv_heads, _ = score_val.shape
    n_pruned = int(bsz * n_kv_heads * ctx_len * cr)
    for layer in model.model.layers:
        module = layer.self_attn
        li = int(module.layer_idx)
        scores = score_val[li]
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten().cpu()
        bi = torch.arange(bsz).repeat_interleave(n_pruned)
        hi = indices // ctx_len
        si = indices % ctx_len
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
    apply_fake_compression(model, score_val, cr, ctx_len)

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


def find_target(tokenizer, ctx_ids, ctx_len, tokens, target_key, target_value):
    for i in range(ctx_len - 20):
        window = tokenizer.decode(ctx_ids[0, i:i+25])
        if target_key in window and target_value in window:
            for j in range(max(0, i-5), min(ctx_len-5, i+10)):
                tok = tokenizer.decode(ctx_ids[0, j:j+8])
                if "for" in tok and "myst" in tok:
                    for end in range(j+5, min(ctx_len, j+25)):
                        if tokens[end].strip() in ['.', '\n', '.\n']:
                            return j, end - j + 1
                    return j, 20
    for i in range(ctx_len - 5):
        window = tokenizer.decode(ctx_ids[0, i:i+10])
        if target_value[:4] in window:
            return max(0, i - 10), 20
    return None, 15


def analyze(score_val, target_start, target_len, label, ctx_len):
    if target_start is None:
        return 0.0
    avg = score_val.float().mean(dim=(0, 1))
    t_mean = avg[:, target_start:target_start+target_len].mean().item()
    mask = torch.ones(ctx_len, dtype=torch.bool)
    mask[:10] = False
    mask[target_start:target_start+target_len] = False
    o_mean = avg[:, mask].mean().item()
    ratio = t_mean / max(o_mean, 1e-10)
    print(f"    {label:20s}: target={t_mean:.6f} other={o_mean:.6f} ratio={ratio:.4f}x")
    return ratio


def main():
    target_key = "mystic-thunder"
    target_value = "7156842"

    print("=" * 90)
    print("GRAPH EXPANSION TRACING vs SINGLE-HOP vs KVZIP")
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
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        ctx_text, q_suffix = full_text.split(separator)
        ctx_ids = tokenizer.encode(ctx_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        q_ids = tokenizer.encode(q_suffix, return_tensors="pt", add_special_tokens=False).to(model.device)
        ctx_len = ctx_ids.shape[1]
        tokens = [tokenizer.decode(ctx_ids[0, i]) for i in range(ctx_len)]

        print(f"  Context: {ctx_len} tokens, Question: {q_ids.shape[1]} tokens")

        target_start, target_len = find_target(tokenizer, ctx_ids, ctx_len, tokens, target_key, target_value)
        if target_start:
            print(f"  Target at pos {target_start}-{target_start+target_len-1}")

        # ─── Compute scores ───
        print(f"\n  Computing scores...")

        # Graph expansion with different alphas
        expansion_scores = {}
        for alpha in [0.5, 1.0, 2.0]:
            print(f"    Graph expansion (alpha={alpha})...")
            scores, _ = compute_graph_expansion_scores(model, ctx_ids, q_ids, expansion_alpha=alpha)
            expansion_scores[alpha] = scores
            torch.cuda.empty_cache()

        print(f"    Single-hop...")
        singlehop_scores, _ = compute_singlehop_scores(model, ctx_ids, q_ids)
        torch.cuda.empty_cache()

        print(f"    KVzip...")
        kvzip_scores = get_kvzip_scores(model, tokenizer, ctx_ids)
        torch.cuda.empty_cache()

        # ─── Score analysis ───
        print(f"\n  Score selectivity (target/other ratio):")
        if target_start:
            for alpha in [0.5, 1.0, 2.0]:
                analyze(expansion_scores[alpha], target_start, target_len,
                       f"GraphExp a={alpha}", ctx_len)
            analyze(singlehop_scores, target_start, target_len, "Single-hop", ctx_len)
            analyze(kvzip_scores, target_start, target_len, "KVzip", ctx_len)

        # ─── Generation ───
        print(f"\n  Generation results:")
        for cr in [0.5, 0.7, 0.9, 0.95]:
            print(f"\n    cr = {cr}:")
            if cr == 0.5:
                gen = generate(model, tokenizer, ctx_ids, q_ids,
                             torch.ones_like(singlehop_scores), 0.0)
                print(f"      {'FullKV':20s}: {'OK' if target_value in gen else 'FAIL'}  {gen[:65]}")
                torch.cuda.empty_cache()

            for alpha in [0.5, 1.0, 2.0]:
                gen = generate(model, tokenizer, ctx_ids, q_ids,
                             expansion_scores[alpha], cr)
                ok = target_value in gen
                print(f"      {'GraphExp a='+str(alpha):20s}: {'OK' if ok else 'FAIL'}  {gen[:65]}")
                torch.cuda.empty_cache()

            gen = generate(model, tokenizer, ctx_ids, q_ids, singlehop_scores, cr)
            print(f"      {'Single-hop':20s}: {'OK' if target_value in gen else 'FAIL'}  {gen[:65]}")
            torch.cuda.empty_cache()

            gen = generate(model, tokenizer, ctx_ids, q_ids, kvzip_scores, cr)
            print(f"      {'KVzip':20s}: {'OK' if target_value in gen else 'FAIL'}  {gen[:65]}")
            torch.cuda.empty_cache()

    print("\n\nDONE")


if __name__ == "__main__":
    main()
