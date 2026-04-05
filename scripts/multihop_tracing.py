#!/usr/bin/env python3
"""Multi-hop attention tracing vs KVzip.

Store full prefill attention at each layer, then trace backward from question tokens.
This is NOT efficient (can't use FlashAttention), but validates the concept.
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
    return (
        f"A special magic number is hidden within the following text. "
        f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {target_key} mentioned in the provided text?"
    )


def compute_multihop_scores(model, context_ids, question_ids):
    """Full multi-hop attention tracing.

    1. Prefill: store attention weights at each layer (via hooks computing Q*K)
    2. Question: store question→context attention at each layer
    3. Trace backward: propagate importance from question through prefill attention
    """
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = context_ids.shape[1]
    q_len = question_ids.shape[1]

    # Storage for prefill attention: [n_layers, n_kv_heads, ctx_len, ctx_len]
    # Grouped by KV head (max over query group)
    prefill_attn = torch.zeros(n_layers, n_kv_heads, ctx_len, ctx_len,
                               dtype=torch.float32, device='cpu')
    # Storage for question attention: [n_layers, n_kv_heads, q_len, ctx_len]
    question_attn = torch.zeros(n_layers, n_kv_heads, q_len, ctx_len,
                                dtype=torch.float32, device='cpu')

    phase = {'current': 'prefill'}  # mutable for closure

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return

            bsz, seq_len, _ = hidden_states.shape
            position_embeddings = kwargs.get("position_embeddings", None)

            with torch.no_grad():
                # Compute Q
                q = module.q_proj(hidden_states)
                q = q.view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)

                # Apply RoPE
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))

                if phase['current'] == 'prefill':
                    # During prefill: compute self-attention Q*K for all positions
                    # K = key projection of hidden states (before cache)
                    k = module.k_proj(hidden_states)
                    k = k.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                    # Apply RoPE to K
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                        # cos/sin shape: [bsz, seq_len, head_dim] — need to handle for K (n_kv_heads)
                        k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))

                    # Group Q: [bsz, n_kv_heads, n_groups, seq_len, head_dim]
                    q_grouped = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                    k_expanded = k.unsqueeze(2)  # [bsz, n_kv_heads, 1, seq_len, head_dim]

                    # Attention: [bsz, n_kv_heads, n_groups, seq_len, seq_len]
                    attn = torch.matmul(q_grouped, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)

                    # Causal mask
                    causal = torch.triu(torch.full((seq_len, seq_len), float('-inf'),
                                                   device=attn.device), diagonal=1)
                    attn = attn + causal.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    attn = torch.softmax(attn, dim=-1)

                    # Max over groups: [bsz, n_kv_heads, seq_len, seq_len]
                    attn_grouped = attn.amax(dim=2)
                    prefill_attn[layer_idx] = attn_grouped[0].float().cpu()

                elif phase['current'] == 'question':
                    # During question: compute Q_question * K_context attention
                    past_kv = kwargs.get("past_key_values", None)
                    if past_kv is None:
                        return

                    k = past_kv.layers[layer_idx].keys[:, :, :ctx_len, :]
                    # K already has RoPE from cache

                    q_grouped = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                    k_expanded = k.unsqueeze(2)

                    # [bsz, n_kv_heads, n_groups, q_len, ctx_len]
                    attn = torch.matmul(q_grouped, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
                    # No causal mask needed (question can attend to all context)
                    # But there's also question self-attention and question-to-context
                    # We only care about question-to-context part
                    attn = torch.softmax(attn, dim=-1)

                    # Max over groups: [bsz, n_kv_heads, q_len, ctx_len]
                    attn_grouped = attn.amax(dim=2)
                    question_attn[layer_idx] = attn_grouped[0].float().cpu()

        return hook_fn

    # Register hooks
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
    print(f"    Prefill attention stored: {prefill_attn.shape} ({prefill_attn.nbytes / 1e6:.0f}MB)")

    # Phase 2: Question forward
    phase['current'] = 'question'
    position_ids = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        model.model(input_ids=question_ids, past_key_values=cache, position_ids=position_ids)
    print(f"    Question attention stored: {question_attn.shape}")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Phase 3: Multi-hop backward tracing
    print(f"    Running multi-hop tracing...")

    # Initialize per-layer importance from question attention
    # question_attn[l]: [n_kv_heads, q_len, ctx_len]
    # Take max over question tokens: [n_kv_heads, ctx_len]
    layer_importance = torch.zeros(n_layers, n_kv_heads, ctx_len)
    for l in range(n_layers):
        layer_importance[l] = question_attn[l].amax(dim=1)  # max over question positions

    # Backward propagation with residual
    # If position i is important at layer l+1, trace which positions it attended to at layer l
    for l in range(n_layers - 2, -1, -1):
        # prefill_attn[l]: [n_kv_heads, ctx_len, ctx_len]
        # layer_importance[l+1]: [n_kv_heads, ctx_len]
        # propagated[h, j] = sum_i importance[h, i] * prefill_attn[l][h, i, j]
        propagated = torch.bmm(
            layer_importance[l + 1].unsqueeze(1),  # [n_kv_heads, 1, ctx_len]
            prefill_attn[l]                         # [n_kv_heads, ctx_len, ctx_len]
        ).squeeze(1)  # [n_kv_heads, ctx_len]

        # Combine: direct question attention + propagated from higher layers
        # Use residual connection factor (attention + skip connection)
        layer_importance[l] = layer_importance[l] + 0.5 * propagated

    # Final score: [n_layers, 1, n_kv_heads, ctx_len] for compatibility with KVzip format
    final_scores = layer_importance.unsqueeze(1)  # [n_layers, 1, n_kv_heads, ctx_len]
    final_scores = final_scores.to(dtype=model.dtype, device=model.device)

    del prefill_attn, question_attn
    return final_scores, cache


def compute_singlehop_scores(model, context_ids, question_ids):
    """Single-hop: only question → context attention (for comparison)."""
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
    """Get KVzip scores."""
    press = KVzipPress(compression_ratio=0.5, layerwise=True)
    saved = {}
    original_compress = press.compress_post
    def patched(m):
        saved['score_val'] = press.score_val.clone()
        original_compress(m)
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
        layer_idx = int(module.layer_idx)
        scores = score_val[layer_idx]
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten().cpu()
        batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
        head_indices = indices // ctx_len
        seq_indices = indices % ctx_len
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)


def clear_fake_compression(model):
    for layer in model.model.layers:
        layer.self_attn.masked_key_indices = None


def generate_with_scores(model, tokenizer, context_ids, question_ids, score_val, cr):
    """Prefill → compress → question → generate."""
    ctx_len = context_ids.shape[1]
    q_len = question_ids.shape[1]

    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=context_ids, past_key_values=cache)

    apply_fake_compression(model, score_val, cr, ctx_len)

    position_ids = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids=question_ids, past_key_values=cache,
                       position_ids=position_ids, num_logits_to_keep=1)

    gen_ids = [outputs.logits[0, -1].argmax()]
    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    cur_pos = ctx_len + q_len
    for i in range(59):
        with torch.no_grad():
            outputs = model(input_ids=gen_ids[-1].unsqueeze(0).unsqueeze(0),
                          past_key_values=cache,
                          position_ids=torch.tensor([[cur_pos + i]], device=model.device))
        nxt = outputs.logits[0, -1].argmax()
        gen_ids.append(nxt)
        if nxt.item() in eos_ids:
            break

    text = tokenizer.decode(torch.stack(gen_ids), skip_special_tokens=True)
    clear_fake_compression(model)
    del cache
    return text


def find_target(tokenizer, ctx_ids, ctx_len, tokens, target_key, target_value):
    """Find target needle position."""
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
    # Fallback
    for i in range(ctx_len - 5):
        window = tokenizer.decode(ctx_ids[0, i:i+10])
        if target_value[:4] in window:
            return max(0, i - 10), 20
    return None, 15


def analyze(score_val, target_start, target_len, label, ctx_len):
    """Quick score analysis."""
    if target_start is None:
        return 0.0
    avg = score_val.mean(dim=(0, 1))  # [n_kv_heads, ctx_len]
    target_mean = avg[:, target_start:target_start+target_len].mean().item()
    mask = torch.ones(ctx_len, dtype=torch.bool)
    mask[:10] = False
    mask[target_start:target_start+target_len] = False
    other_mean = avg[:, mask].mean().item()
    ratio = target_mean / max(other_mean, 1e-10)
    print(f"    {label}: target={target_mean:.6f} other={other_mean:.6f} ratio={ratio:.4f}x")
    return ratio


def main():
    target_key = "mystic-thunder"
    target_value = "7156842"

    print("=" * 90)
    print("MULTI-HOP ATTENTION TRACING vs SINGLE-HOP vs KVZIP")
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
        q_len = q_ids.shape[1]
        tokens = [tokenizer.decode(ctx_ids[0, i]) for i in range(ctx_len)]

        print(f"  Context: {ctx_len} tokens, Question: {q_len} tokens")

        target_start, target_len = find_target(tokenizer, ctx_ids, ctx_len, tokens, target_key, target_value)
        if target_start:
            print(f"  Target at pos {target_start}-{target_start+target_len-1}")

        # ─── Compute all three scoring methods ───
        print(f"\n  [1] Multi-hop tracing...")
        multihop_scores, _ = compute_multihop_scores(model, ctx_ids, q_ids)
        torch.cuda.empty_cache()

        print(f"\n  [2] Single-hop tracing...")
        singlehop_scores, _ = compute_singlehop_scores(model, ctx_ids, q_ids)
        torch.cuda.empty_cache()

        print(f"\n  [3] KVzip reconstruction...")
        kvzip_scores = get_kvzip_scores(model, tokenizer, ctx_ids)
        torch.cuda.empty_cache()

        # ─── Score analysis ───
        print(f"\n  Score selectivity (target/other ratio):")
        r_multi = analyze(multihop_scores, target_start, target_len, "Multi-hop", ctx_len)
        r_single = analyze(singlehop_scores, target_start, target_len, "Single-hop", ctx_len)
        r_kvzip = analyze(kvzip_scores, target_start, target_len, "KVzip", ctx_len)

        if r_multi > r_kvzip:
            print(f"  → Multi-hop is {r_multi/max(r_kvzip,1e-10):.2f}x more selective than KVzip!")
        if r_multi > r_single:
            print(f"  → Multi-hop is {r_multi/max(r_single,1e-10):.2f}x more selective than Single-hop!")

        # ─── Generation comparison ───
        print(f"\n  Generation results:")
        for cr in [0.5, 0.7, 0.9, 0.95]:
            print(f"\n    cr = {cr}:")

            if cr == 0.5:
                # Full KV baseline
                gen = generate_with_scores(model, tokenizer, ctx_ids, q_ids,
                                          torch.ones_like(multihop_scores), 0.0)
                print(f"      FullKV:     {'OK' if target_value in gen else 'FAIL'}  {gen[:70]}")
                torch.cuda.empty_cache()

            for label, scores in [("Multi-hop", multihop_scores),
                                   ("Single-hop", singlehop_scores),
                                   ("KVzip", kvzip_scores)]:
                gen = generate_with_scores(model, tokenizer, ctx_ids, q_ids, scores, cr)
                ok = target_value in gen
                print(f"      {label:11s}: {'OK' if ok else 'FAIL'}  {gen[:70]}")
                torch.cuda.empty_cache()

    print("\n\nDONE")


if __name__ == "__main__":
    main()
