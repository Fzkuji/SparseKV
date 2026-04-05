#!/usr/bin/env python3
"""Focused analysis: does "mystic-thunder" in question specifically attend to
"mystic-thunder" in the target needle vs other distractor keys?

This is the KEY question: if we can show that question's "mystic" tokens
preferentially attend to the needle's "mystic" tokens (not other keys),
then the attention signal IS discriminative.
"""

import torch
import random
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half

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


def find_all_needle_keys(tokenizer, ctx_ids, ctx_len, target_key, all_keys):
    """Find token positions for ALL needle keys (target + distractors)."""
    tokens = [tokenizer.decode(ctx_ids[0, i]) for i in range(ctx_len)]
    full_text = tokenizer.decode(ctx_ids[0])

    # Build char→token mapping
    char_to_tok = []
    cum_len = 0
    for i in range(ctx_len):
        tok_text = tokenizer.decode(ctx_ids[0, :i+1])
        new_len = len(tok_text)
        for _ in range(new_len - cum_len):
            char_to_tok.append(i)
        cum_len = new_len

    # Find each key's token positions in the needles (not in question text)
    # Only search up to the question text area
    q_char = full_text.rfind("What is the special")
    search_end = q_char if q_char > 0 else len(full_text)

    key_positions = {}  # key_name → list of token positions
    sentence_positions = {}  # key_name → list of token positions for full sentence
    value_positions = {}  # key_name → list of token positions for value

    for key_name in all_keys:
        # Find "for {key_name} is:" pattern in the needle area
        pattern = f"for {key_name} is:"
        idx = full_text.find(pattern, 0, search_end)
        if idx < 0:
            continue

        # Key token positions
        key_start_char = full_text.find(key_name, idx, idx + len(pattern))
        key_end_char = key_start_char + len(key_name) - 1
        if key_start_char >= 0 and key_start_char < len(char_to_tok):
            ks = char_to_tok[key_start_char]
            ke = char_to_tok[min(key_end_char, len(char_to_tok)-1)]
            key_positions[key_name] = list(range(ks, ke + 1))

        # Full sentence: find "One of the special" before and "." after
        one_pos = full_text.rfind("One of the special", 0, idx)
        period_pos = full_text.find(".", idx + len(pattern))
        if one_pos >= 0 and period_pos >= 0:
            ss = char_to_tok[one_pos]
            se = char_to_tok[min(period_pos, len(char_to_tok)-1)]
            sentence_positions[key_name] = list(range(ss, se + 1))

        # Value: digits after ":"
        colon_pos = full_text.find(":", idx + len("for "))
        if colon_pos >= 0:
            # Find digits after colon
            val_start = colon_pos + 2  # skip ": "
            val_end = full_text.find(".", val_start)
            if val_end > val_start and val_start < len(char_to_tok):
                vs = char_to_tok[val_start]
                ve = char_to_tok[min(val_end - 1, len(char_to_tok)-1)]
                value_positions[key_name] = list(range(vs, ve + 1))

    # Find question text key reference ("mystic-thunder" in question)
    q_key_char = full_text.find(target_key, q_char)
    q_key_positions = []
    if q_key_char >= 0 and q_key_char < len(char_to_tok):
        qks = char_to_tok[q_key_char]
        qke = char_to_tok[min(q_key_char + len(target_key) - 1, len(char_to_tok)-1)]
        q_key_positions = list(range(qks, qke + 1))

    # Question text positions
    q_start = char_to_tok[q_char] if q_char >= 0 else ctx_len - 20
    q_text_positions = list(range(q_start, ctx_len))

    # Print summary
    print(f"\n  Found {len(key_positions)} needle keys:")
    print(f"  Target: '{target_key}' at tokens {key_positions.get(target_key, 'NOT FOUND')}")
    print(f"    = {[tokens[i] for i in key_positions.get(target_key, [])]}")
    print(f"  Question key ref: {q_key_positions} = {[tokens[i] for i in q_key_positions]}")
    print(f"  Example distractors:")
    for i, (k, v) in enumerate(key_positions.items()):
        if k != target_key and i < 5:
            print(f"    '{k}' at {v} = {[tokens[j] for j in v]}")

    return key_positions, sentence_positions, value_positions, q_key_positions, q_text_positions, tokens


def compute_prefill_attention(model, ctx_ids):
    """Compute prefill self-attention."""
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = ctx_ids.shape[1]

    prefill_attn = {}

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
                q_g = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                k_e = k.unsqueeze(2)
                attn = torch.matmul(q_g, k_e.transpose(-2, -1)) / math.sqrt(head_dim)
                causal = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=attn.device), diagonal=1)
                attn = torch.softmax(attn + causal[None, None, None], dim=-1)
                prefill_attn[layer_idx] = attn.amax(dim=2)[0].float().cpu()
        return hook_fn

    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.self_attn.register_forward_hook(
            make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True))

    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=cache)

    for h in hooks:
        h.remove()
    del cache
    return prefill_attn


def main():
    model_path = "Qwen/Qwen3-8B"
    target_key = "mystic-thunder"
    target_value = "7156842"
    n_dist = 30
    seed = 50

    print("=" * 100)
    print("KEY MATCHING ANALYSIS")
    print("Does 'mystic-thunder' in question specifically attend to 'mystic-thunder' in needle?")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    # Build prompt and get all key names
    random.seed(seed)
    all_keys = [target_key]
    used_keys = {target_key}
    for _ in range(n_dist):
        while True:
            key = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
            if key not in used_keys:
                used_keys.add(key)
                all_keys.append(key)
                break

    prompt = build_prompt(n_dist, target_key, target_value, 0.5, seed=seed)
    separator = "#" * (len(prompt) + 10)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt + separator}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    ctx_text, q_suffix = full_text.split(separator)
    ctx_ids = tokenizer.encode(ctx_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    ctx_len = ctx_ids.shape[1]

    print(f"\n  Context: {ctx_len} tokens, {len(all_keys)} keys")

    key_pos, sent_pos, val_pos, q_key_pos, q_text_pos, tokens = \
        find_all_needle_keys(tokenizer, ctx_ids, ctx_len, target_key, all_keys)

    print(f"\n  Computing prefill attention...")
    prefill_attn = compute_prefill_attention(model, ctx_ids)
    n_layers = model.config.num_hidden_layers
    n_kv_heads = prefill_attn[0].shape[0]

    # =====================================================================
    print(f"\n{'='*100}")
    print(f"TEST 1: 'mystic-thunder' in question → each needle's key tokens")
    print(f"  Question key tokens: {q_key_pos} = {[tokens[i] for i in q_key_pos]}")
    print(f"{'='*100}")

    for l in range(n_layers):
        if l not in prefill_attn:
            continue
        pa = prefill_attn[l]  # [n_kv_heads, ctx_len, ctx_len]

        # Attention from q_key_pos to each needle's key tokens
        # pa[:, q_key_pos, :] → [n_kv_heads, len(q_key_pos), ctx_len]
        q_attn = pa[:, q_key_pos, :].amax(dim=1)  # [n_kv_heads, ctx_len] max over q_key tokens

        results = []
        for key_name in all_keys:
            if key_name not in key_pos:
                continue
            kp = key_pos[key_name]
            attn_to_key = q_attn[:, kp].sum(dim=1).mean().item()  # avg over heads
            is_target = "★" if key_name == target_key else " "
            results.append((key_name, attn_to_key, is_target))

        results.sort(key=lambda x: -x[1])

        if l % 4 == 0:
            target_rank = next(i for i, (k, _, _) in enumerate(results) if k == target_key)
            target_score = next(s for k, s, _ in results if k == target_key)
            top_score = results[0][1]
            avg_score = sum(s for _, s, _ in results) / len(results)
            print(f"\n  Layer {l:2d}: target_rank={target_rank+1}/{len(results)}  "
                  f"target_score={target_score:.6f}  top={top_score:.6f}  avg={avg_score:.6f}  "
                  f"target/avg={target_score/max(avg_score,1e-10):.2f}x")
            # Show top 5 and target
            for i, (k, s, star) in enumerate(results[:5]):
                print(f"    {star} {i+1:2d}. {k:20s} {s:.6f}")
            if target_rank >= 5:
                print(f"    ...")
                print(f"    {results[target_rank][2]} {target_rank+1:2d}. {results[target_rank][0]:20s} {results[target_rank][1]:.6f}")

    # =====================================================================
    print(f"\n{'='*100}")
    print(f"TEST 2: Per-head breakdown at best layers")
    print(f"{'='*100}")

    # Find best layers (where target rank is highest)
    layer_ranks = []
    for l in range(n_layers):
        if l not in prefill_attn:
            continue
        pa = prefill_attn[l]
        q_attn = pa[:, q_key_pos, :].amax(dim=1)
        results = []
        for key_name in all_keys:
            if key_name not in key_pos:
                continue
            kp = key_pos[key_name]
            attn_to_key = q_attn[:, kp].sum(dim=1).mean().item()
            results.append((key_name, attn_to_key))
        results.sort(key=lambda x: -x[1])
        rank = next(i for i, (k, _) in enumerate(results) if k == target_key) + 1
        layer_ranks.append((l, rank))

    layer_ranks.sort(key=lambda x: x[1])
    print(f"\n  Best layers (target key rank closest to 1):")
    for l, rank in layer_ranks[:10]:
        print(f"    Layer {l:2d}: rank {rank}/{len(all_keys)}")

    # For top 3 best layers, show per-head detail
    for l, rank in layer_ranks[:3]:
        pa = prefill_attn[l]
        print(f"\n  Layer {l} (target rank={rank}) per-head:")
        for h in range(n_kv_heads):
            h_attn = pa[h, q_key_pos, :].amax(dim=0)  # [ctx_len]
            target_kp = key_pos[target_key]
            target_attn = h_attn[target_kp].sum().item()

            # Compare to other keys
            other_attns = []
            for key_name in all_keys:
                if key_name == target_key or key_name not in key_pos:
                    continue
                kp = key_pos[key_name]
                other_attns.append(h_attn[kp].sum().item())
            avg_other = sum(other_attns) / len(other_attns) if other_attns else 0
            max_other = max(other_attns) if other_attns else 0

            ratio = target_attn / max(avg_other, 1e-10)
            print(f"    KV{h}: target={target_attn:.5f}  avg_other={avg_other:.5f}  "
                  f"max_other={max_other:.5f}  target/avg={ratio:.2f}x")

    # =====================================================================
    print(f"\n{'='*100}")
    print(f"TEST 3: Full question text → each needle's ENTIRE sentence (key+value+template)")
    print(f"{'='*100}")

    for l in range(0, n_layers, 4):
        if l not in prefill_attn:
            continue
        pa = prefill_attn[l]
        q_attn = pa[:, q_text_pos, :].amax(dim=1)  # [n_kv_heads, ctx_len]

        results = []
        for key_name in all_keys:
            if key_name not in sent_pos:
                continue
            sp = sent_pos[key_name]
            attn_to_sent = q_attn[:, sp].sum(dim=1).mean().item()
            is_target = "★" if key_name == target_key else " "
            results.append((key_name, attn_to_sent, is_target))

        results.sort(key=lambda x: -x[1])
        target_rank = next(i for i, (k, _, _) in enumerate(results) if k == target_key) + 1
        target_score = next(s for k, s, _ in results if k == target_key)
        avg_score = sum(s for _, s, _ in results) / len(results)

        print(f"\n  Layer {l:2d}: target_rank={target_rank}/{len(results)}  "
              f"target={target_score:.6f}  avg={avg_score:.6f}  ratio={target_score/max(avg_score,1e-10):.2f}x")
        for i, (k, s, star) in enumerate(results[:3]):
            print(f"    {star} {i+1:2d}. {k:20s} {s:.6f}")
        if target_rank > 3:
            print(f"    ...")
            print(f"    {results[target_rank-1][2]} {target_rank:2d}. {results[target_rank-1][0]:20s} {results[target_rank-1][1]:.6f}")

    # =====================================================================
    print(f"\n{'='*100}")
    print(f"TEST 4: If we use ONLY q_key tokens (not full question text) as seed,")
    print(f"        and EXCLUDE question text positions from scores,")
    print(f"        how does target needle rank?")
    print(f"{'='*100}")

    for l in range(n_layers):
        if l not in prefill_attn:
            continue
        pa = prefill_attn[l]

        # Score = attention FROM q_key_pos TO each position, excluding q_text positions
        seed = pa[:, q_key_pos, :].amax(dim=1)  # [n_kv_heads, ctx_len]
        # Zero out question text positions
        seed[:, q_text_pos] = 0

        results = []
        for key_name in all_keys:
            if key_name not in sent_pos:
                continue
            sp = sent_pos[key_name]
            score = seed[:, sp].sum(dim=1).mean().item()
            results.append((key_name, score))

        results.sort(key=lambda x: -x[1])
        target_rank = next((i for i, (k, _) in enumerate(results) if k == target_key), -1) + 1
        target_score = next((s for k, s in results if k == target_key), 0)
        avg_score = sum(s for _, s in results) / len(results)

        if l % 4 == 0:
            print(f"  L{l:2d}: target_rank={target_rank}/{len(results)}  "
                  f"score={target_score:.6f}  avg={avg_score:.6f}  ratio={target_score/max(avg_score,1e-10):.2f}x")

    print("\n\nDONE")


if __name__ == "__main__":
    main()
