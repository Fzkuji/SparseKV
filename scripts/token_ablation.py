#!/usr/bin/env python3
"""Token ablation: find the minimal set of tokens needed for correct NIAH answer.

For each test, we manually force-keep a specific subset of tokens (+ sink/recent)
and evict everything else, then check if the model can still answer correctly.
"""

import torch
import random
import math
import re
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
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
    """Find token positions that correspond to a given string."""
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
        # Token overlaps with target string
        if len(cum) > char_pos and prev_len < char_pos + len(search_str):
            positions.append(i)
    return positions


def find_needle_components(tokenizer, ctx_ids, target_key, target_value):
    """Find positions of different needle components."""
    # Full needle: "One of the special magic numbers for mystic-thunder is: 7156842."
    full_needle = f"One of the special magic numbers for {target_key} is: {target_value}."
    full_pos = find_token_positions(tokenizer, ctx_ids, full_needle)

    # Key entity: "mystic-thunder"
    key_pos = find_token_positions(tokenizer, ctx_ids, target_key)

    # Value: "7156842"
    value_pos = find_token_positions(tokenizer, ctx_ids, target_value)

    # Period at end of needle
    period_pos = find_token_positions(tokenizer, ctx_ids, f"{target_value}.")
    # Period is the last token of value_pos range
    if period_pos:
        period_only = [p for p in period_pos if p not in value_pos]
        if not period_only:
            # Period might be part of last value token
            period_only = [period_pos[-1]]
    else:
        period_only = []

    # "is: " separator
    sep_pos = find_token_positions(tokenizer, ctx_ids, f"is: {target_value}")
    sep_only = [p for p in sep_pos if p not in key_pos and p not in value_pos]

    # Prefix: "One of the special magic numbers for "
    prefix_pos = find_token_positions(tokenizer, ctx_ids, "One of the special magic numbers for " + target_key)
    prefix_only = [p for p in prefix_pos if p not in key_pos]

    # Print what we found
    ctx_len = ctx_ids.shape[1]
    print(f"\n  Needle token positions (context length: {ctx_len}):")
    print(f"    Full needle: {full_pos[0]}-{full_pos[-1]} ({len(full_pos)} tokens)")
    print(f"    Prefix 'One of...for ': {prefix_only} ({len(prefix_only)} tokens)")
    for p in prefix_only[:5]:
        print(f"      pos {p}: '{tokenizer.decode(ctx_ids[0, p])}'")
    print(f"    Key '{target_key}': {key_pos} ({len(key_pos)} tokens)")
    for p in key_pos:
        print(f"      pos {p}: '{tokenizer.decode(ctx_ids[0, p])}'")
    print(f"    Sep 'is: ': {sep_only} ({len(sep_only)} tokens)")
    for p in sep_only:
        print(f"      pos {p}: '{tokenizer.decode(ctx_ids[0, p])}'")
    print(f"    Value '{target_value}': {value_pos} ({len(value_pos)} tokens)")
    for p in value_pos:
        print(f"      pos {p}: '{tokenizer.decode(ctx_ids[0, p])}'")
    print(f"    Period: {period_only} ({len(period_only)} tokens)")
    for p in period_only:
        print(f"      pos {p}: '{tokenizer.decode(ctx_ids[0, p])}'")

    return {
        'full': full_pos,
        'prefix': prefix_only,
        'key': key_pos,
        'sep': sep_only,
        'value': value_pos,
        'period': period_only,
    }


def apply_manual_mask(model, keep_positions, ctx_len, n_layers, n_kv_heads):
    """Force evict everything except keep_positions (applied to ALL layers and heads)."""
    keep_set = set(keep_positions)
    evict_positions = [i for i in range(ctx_len) if i not in keep_set]

    if not evict_positions:
        for layer in model.model.layers:
            layer.self_attn.masked_key_indices = None
        return

    evict_t = torch.tensor(evict_positions, dtype=torch.long)
    for layer in model.model.layers:
        li = int(layer.self_attn.layer_idx)
        # Evict same positions in all heads
        bi = torch.zeros(len(evict_positions) * n_kv_heads, dtype=torch.long)
        hi = torch.arange(n_kv_heads, dtype=torch.long).repeat_interleave(len(evict_positions))
        si = evict_t.repeat(n_kv_heads)
        layer.self_attn.masked_key_indices = (bi, hi, si)


def clear_compression(model):
    for layer in model.model.layers:
        layer.self_attn.masked_key_indices = None


def generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids, keep_positions,
                               n_layers, n_kv_heads, max_new=60):
    """Generate answer keeping only specified positions."""
    ctx_len = ctx_ids.shape[1]
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=cache)

    apply_manual_mask(model, keep_positions, ctx_len, n_layers, n_kv_heads)

    q_len = q_ids.shape[1]
    pos = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=q_ids, past_key_values=cache, position_ids=pos, num_logits_to_keep=1)
    gen = [out.logits[0, -1].argmax()]
    eos = model.generation_config.eos_token_id
    if not isinstance(eos, list):
        eos = [eos]
    cp = ctx_len + q_len
    for i in range(max_new - 1):
        with torch.no_grad():
            out = model(input_ids=gen[-1].unsqueeze(0).unsqueeze(0),
                       past_key_values=cache,
                       position_ids=torch.tensor([[cp + i]], device=model.device))
        nxt = out.logits[0, -1].argmax()
        gen.append(nxt)
        if nxt.item() in eos:
            break
    text = tokenizer.decode(torch.stack(gen), skip_special_tokens=True)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    clear_compression(model)
    del cache
    return text


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
    print(f"Model loaded. {n_layers} layers, {n_kv_heads} KV heads.")

    for n_dist in [30]:
        print(f"\n{'=' * 80}")
        print(f"  {n_dist} distractors, needle at 50%")
        print(f"{'=' * 80}")

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
        print(f"  Context: {ctx_len} tokens, Question: {q_ids.shape[1]} tokens")

        # Find all needle component positions
        components = find_needle_components(tokenizer, ctx_ids, target_key, target_value)

        # Base tokens always kept: sink (first 4) + recent (last 32)
        sink = list(range(4))
        recent = list(range(max(4, ctx_len - 32), ctx_len))
        base = sorted(set(sink + recent))
        n_base = len(base)

        print(f"\n  Base (always kept): sink[0:4] + recent[{ctx_len-32}:{ctx_len}] = {n_base} tokens")
        print(f"  Total context: {ctx_len} tokens")

        # ═══ Test 0: Full KV (baseline) ═══
        print(f"\n  {'─' * 60}")
        print(f"  Test 0: Full KV (no compression)")
        full = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                          list(range(ctx_len)), n_layers, n_kv_heads)
        ok = target_value in full
        print(f"    Result: {'OK' if ok else 'FAIL'}  {full[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 1: Only base (sink + recent, NO needle at all) ═══
        print(f"\n  {'─' * 60}")
        print(f"  Test 1: Only base (sink + recent) — {n_base} tokens kept")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         base, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 2: base + full needle sentence ═══
        keep = sorted(set(base + components['full']))
        print(f"\n  {'─' * 60}")
        print(f"  Test 2: base + full needle — {len(keep)} tokens kept (+{len(keep)-n_base} needle)")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 3: base + key + value only ═══
        keep = sorted(set(base + components['key'] + components['value']))
        print(f"\n  {'─' * 60}")
        print(f"  Test 3: base + key + value — {len(keep)} tokens kept (+{len(keep)-n_base})")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 4: base + key only ═══
        keep = sorted(set(base + components['key']))
        print(f"\n  {'─' * 60}")
        print(f"  Test 4: base + key only — {len(keep)} tokens kept (+{len(keep)-n_base})")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 5: base + value only ═══
        keep = sorted(set(base + components['value']))
        print(f"\n  {'─' * 60}")
        print(f"  Test 5: base + value only — {len(keep)} tokens kept (+{len(keep)-n_base})")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 6: base + key + value + period ═══
        keep = sorted(set(base + components['key'] + components['value'] + components['period']))
        print(f"\n  {'─' * 60}")
        print(f"  Test 6: base + key + value + period — {len(keep)} tokens kept (+{len(keep)-n_base})")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 7: base + key + value + sep ═══
        keep = sorted(set(base + components['key'] + components['value'] + components['sep']))
        print(f"\n  {'─' * 60}")
        print(f"  Test 7: base + key + value + sep('is: ') — {len(keep)} tokens kept (+{len(keep)-n_base})")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 8: base + key + value + period + sep ═══
        keep = sorted(set(base + components['key'] + components['value'] + components['period'] + components['sep']))
        print(f"\n  {'─' * 60}")
        print(f"  Test 8: base + key + value + period + sep — {len(keep)} tokens kept (+{len(keep)-n_base})")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 9: base + full needle + surrounding context (±5 tokens) ═══
        surround = []
        if components['full']:
            start = max(0, components['full'][0] - 5)
            end = min(ctx_len, components['full'][-1] + 6)
            surround = list(range(start, end))
        keep = sorted(set(base + surround))
        print(f"\n  {'─' * 60}")
        print(f"  Test 9: base + needle ±5 tokens — {len(keep)} tokens kept (+{len(keep)-n_base})")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 10: base + prefix + key (no value at all!) ═══
        keep = sorted(set(base + components['prefix'] + components['key']))
        print(f"\n  {'─' * 60}")
        print(f"  Test 10: base + prefix + key (NO value) — {len(keep)} tokens kept (+{len(keep)-n_base})")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 11: base + value + period (no key!) ═══
        keep = sorted(set(base + components['value'] + components['period']))
        print(f"\n  {'─' * 60}")
        print(f"  Test 11: base + value + period (NO key) — {len(keep)} tokens kept (+{len(keep)-n_base})")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 12: base + ONLY period ═══
        keep = sorted(set(base + components['period']))
        print(f"\n  {'─' * 60}")
        print(f"  Test 12: base + period only — {len(keep)} tokens kept (+{len(keep)-n_base})")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 13: base + ALL distractor needles' key-value (but NOT target) ═══
        # This tests if distractors alone confuse the model
        all_but_target = [i for i in range(ctx_len) if i not in components['full']]
        keep = sorted(set(base + all_but_target))
        print(f"\n  {'─' * 60}")
        print(f"  Test 13: base + everything EXCEPT target needle — {len(keep)} tokens kept")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 14: base + all punctuation in entire context ═══
        punct_pos = []
        for i in range(ctx_len):
            tok = tokenizer.decode(ctx_ids[0, i]).strip()
            if tok in ['.', ',', ':', ';', '!', '?', '-', '\n']:
                punct_pos.append(i)
        keep = sorted(set(base + components['key'] + components['value'] + punct_pos))
        print(f"\n  {'─' * 60}")
        print(f"  Test 14: base + key + value + ALL punctuation — {len(keep)} tokens kept (+{len(keep)-n_base})")
        gen = generate_with_manual_keep(model, tokenizer, ctx_ids, q_ids,
                                         keep, n_layers, n_kv_heads)
        ok = target_value in gen
        print(f"    Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
        torch.cuda.empty_cache()

        # ═══ Test 15: Per-layer ablation — keep needle only in top-k layers ═══
        # Instead of same mask for all layers, keep needle in layers 20-35 only
        print(f"\n  {'─' * 60}")
        print(f"  Test 15: Per-layer — needle kept only in layers 20-35")
        needle_pos = sorted(set(components['full']))
        cache = DynamicCache()
        with torch.no_grad():
            model.model(input_ids=ctx_ids, past_key_values=cache)

        # Apply per-layer mask
        for layer in model.model.layers:
            li = int(layer.self_attn.layer_idx)
            if 20 <= li <= 35:
                # Keep base + needle
                keep_set = set(base + needle_pos)
            else:
                # Keep only base
                keep_set = set(base)

            evict = [i for i in range(ctx_len) if i not in keep_set]
            if not evict:
                layer.self_attn.masked_key_indices = None
                continue
            evict_t = torch.tensor(evict, dtype=torch.long)
            bi = torch.zeros(len(evict) * n_kv_heads, dtype=torch.long)
            hi = torch.arange(n_kv_heads, dtype=torch.long).repeat_interleave(len(evict))
            si = evict_t.repeat(n_kv_heads)
            layer.self_attn.masked_key_indices = (bi, hi, si)

        q_len = q_ids.shape[1]
        pos = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=q_ids, past_key_values=cache, position_ids=pos, num_logits_to_keep=1)
        gen_tokens = [out.logits[0, -1].argmax()]
        eos = model.generation_config.eos_token_id
        if not isinstance(eos, list):
            eos = [eos]
        cp = ctx_len + q_len
        for i in range(59):
            with torch.no_grad():
                out = model(input_ids=gen_tokens[-1].unsqueeze(0).unsqueeze(0),
                           past_key_values=cache,
                           position_ids=torch.tensor([[cp + i]], device=model.device))
            nxt = out.logits[0, -1].argmax()
            gen_tokens.append(nxt)
            if nxt.item() in eos:
                break
        text = tokenizer.decode(torch.stack(gen_tokens), skip_special_tokens=True)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        ok = target_value in text
        print(f"    Result: {'OK' if ok else 'FAIL'}  {text[:100]}")
        clear_compression(model)
        del cache
        torch.cuda.empty_cache()

        # ═══ Test 16: Per-layer — needle kept only in layers 0-19 ═══
        print(f"\n  {'─' * 60}")
        print(f"  Test 16: Per-layer — needle kept only in layers 0-19")
        cache = DynamicCache()
        with torch.no_grad():
            model.model(input_ids=ctx_ids, past_key_values=cache)

        for layer in model.model.layers:
            li = int(layer.self_attn.layer_idx)
            if li <= 19:
                keep_set = set(base + needle_pos)
            else:
                keep_set = set(base)
            evict = [i for i in range(ctx_len) if i not in keep_set]
            if not evict:
                layer.self_attn.masked_key_indices = None
                continue
            evict_t = torch.tensor(evict, dtype=torch.long)
            bi = torch.zeros(len(evict) * n_kv_heads, dtype=torch.long)
            hi = torch.arange(n_kv_heads, dtype=torch.long).repeat_interleave(len(evict))
            si = evict_t.repeat(n_kv_heads)
            layer.self_attn.masked_key_indices = (bi, hi, si)

        pos = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=q_ids, past_key_values=cache, position_ids=pos, num_logits_to_keep=1)
        gen_tokens = [out.logits[0, -1].argmax()]
        cp = ctx_len + q_len
        for i in range(59):
            with torch.no_grad():
                out = model(input_ids=gen_tokens[-1].unsqueeze(0).unsqueeze(0),
                           past_key_values=cache,
                           position_ids=torch.tensor([[cp + i]], device=model.device))
            nxt = out.logits[0, -1].argmax()
            gen_tokens.append(nxt)
            if nxt.item() in eos:
                break
        text = tokenizer.decode(torch.stack(gen_tokens), skip_special_tokens=True)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        ok = target_value in text
        print(f"    Result: {'OK' if ok else 'FAIL'}  {text[:100]}")
        clear_compression(model)
        del cache
        torch.cuda.empty_cache()

    print("\n\nDONE")


if __name__ == "__main__":
    main()
