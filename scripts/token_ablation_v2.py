#!/usr/bin/env python3
"""Token ablation v2: test whether value-only success is due to "only number visible".

Key question: if we keep multiple distractors' values alongside target value,
can the model still pick the correct one without the key?
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
    distractor_info = []  # (key, value) for each distractor
    for _ in range(num_distractors):
        while True:
            key = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
            if key not in used_keys:
                used_keys.add(key)
                break
        value = str(random.randint(1000000, 9999999))
        needles.append(f"One of the special magic numbers for {key} is: {value}.")
        distractor_info.append((key, value))
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
    return prompt, distractor_info


def find_token_positions(tokenizer, ctx_ids, search_str):
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
        if len(cum) > char_pos and prev_len < char_pos + len(search_str):
            positions.append(i)
    return positions


def find_needle_full(tokenizer, ctx_ids, key, value):
    """Find positions of a complete needle sentence."""
    needle = f"One of the special magic numbers for {key} is: {value}."
    return find_token_positions(tokenizer, ctx_ids, needle)


def find_value_positions(tokenizer, ctx_ids, value):
    return find_token_positions(tokenizer, ctx_ids, value)


def find_key_positions(tokenizer, ctx_ids, key):
    return find_token_positions(tokenizer, ctx_ids, key)


def apply_manual_mask(model, keep_positions, ctx_len, n_kv_heads):
    keep_set = set(keep_positions)
    evict_positions = [i for i in range(ctx_len) if i not in keep_set]
    if not evict_positions:
        for layer in model.model.layers:
            layer.self_attn.masked_key_indices = None
        return
    evict_t = torch.tensor(evict_positions, dtype=torch.long)
    for layer in model.model.layers:
        bi = torch.zeros(len(evict_positions) * n_kv_heads, dtype=torch.long)
        hi = torch.arange(n_kv_heads, dtype=torch.long).repeat_interleave(len(evict_positions))
        si = evict_t.repeat(n_kv_heads)
        layer.self_attn.masked_key_indices = (bi, hi, si)


def clear_compression(model):
    for layer in model.model.layers:
        layer.self_attn.masked_key_indices = None


def generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep_positions,
                       n_kv_heads, max_new=60):
    ctx_len = ctx_ids.shape[1]
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=cache)
    apply_manual_mask(model, keep_positions, ctx_len, n_kv_heads)
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

    n_dist = 30
    print(f"\n{'=' * 80}")
    print(f"  {n_dist} distractors, needle at 50%")
    print(f"{'=' * 80}")

    prompt, distractor_info = build_niah_prompt(n_dist, target_key, target_value, 0.5, seed=50)
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

    # Base tokens
    sink = list(range(4))
    recent = list(range(max(4, ctx_len - 32), ctx_len))
    base = sorted(set(sink + recent))

    # Find target components
    target_key_pos = find_key_positions(tokenizer, ctx_ids, target_key)
    target_val_pos = find_value_positions(tokenizer, ctx_ids, target_value)
    target_full_pos = find_needle_full(tokenizer, ctx_ids, target_key, target_value)
    print(f"  Target key '{target_key}': pos {target_key_pos}")
    print(f"  Target value '{target_value}': pos {target_val_pos}")

    # Find distractor components
    print(f"\n  Distractor positions:")
    distractor_values = {}  # idx -> value_positions
    distractor_keys = {}
    distractor_fulls = {}
    for idx, (dk, dv) in enumerate(distractor_info):
        dv_pos = find_value_positions(tokenizer, ctx_ids, dv)
        dk_pos = find_key_positions(tokenizer, ctx_ids, dk)
        df_pos = find_needle_full(tokenizer, ctx_ids, dk, dv)
        distractor_values[idx] = dv_pos
        distractor_keys[idx] = dk_pos
        distractor_fulls[idx] = df_pos
        if idx < 5 or idx == n_dist - 1:
            print(f"    D{idx}: key='{dk}' pos={dk_pos}, val='{dv}' pos={dv_pos}")

    # ═══════════════════════════════════════════════════════
    # SECTION A: Is "value only" success just because it's the only visible number?
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  SECTION A: Testing if 'value only' success is misleading")
    print(f"{'=' * 80}")

    # A1: base + target value + 5 nearest distractor values (NO keys at all)
    # Pick 5 distractors closest to the target needle position
    target_center = sum(target_val_pos) / len(target_val_pos)
    dists = []
    for idx, dv_pos in distractor_values.items():
        if dv_pos:
            center = sum(dv_pos) / len(dv_pos)
            dists.append((abs(center - target_center), idx))
    dists.sort()
    nearby_5 = [idx for _, idx in dists[:5]]

    extra_val_pos = []
    for idx in nearby_5:
        extra_val_pos.extend(distractor_values[idx])
    keep = sorted(set(base + target_val_pos + extra_val_pos))
    print(f"\n  A1: base + target value + 5 nearest distractor values (NO keys)")
    print(f"      Distractors: {[distractor_info[i] for i in nearby_5]}")
    print(f"      {len(keep)} tokens kept")
    gen = generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep, n_kv_heads)
    ok = target_value in gen
    print(f"      Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
    torch.cuda.empty_cache()

    # A2: base + target value + ALL 30 distractor values (NO keys at all)
    all_dist_val_pos = []
    for idx, dv_pos in distractor_values.items():
        all_dist_val_pos.extend(dv_pos)
    keep = sorted(set(base + target_val_pos + all_dist_val_pos))
    print(f"\n  A2: base + target value + ALL 30 distractor values (NO keys)")
    print(f"      {len(keep)} tokens kept")
    gen = generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep, n_kv_heads)
    ok = target_value in gen
    print(f"      Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
    torch.cuda.empty_cache()

    # A3: base + WRONG value only (a distractor's value, no keys)
    wrong_idx = nearby_5[0]
    wrong_val = distractor_info[wrong_idx][1]
    wrong_val_pos = distractor_values[wrong_idx]
    keep = sorted(set(base + wrong_val_pos))
    print(f"\n  A3: base + WRONG value only ('{wrong_val}' from distractor {wrong_idx})")
    print(f"      {len(keep)} tokens kept")
    gen = generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep, n_kv_heads)
    ok_wrong = wrong_val in gen
    ok_target = target_value in gen
    print(f"      Result: outputs wrong='{wrong_val}'? {ok_wrong}, outputs target='{target_value}'? {ok_target}")
    print(f"      Output: {gen[:100]}")
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # SECTION B: What's the minimal set WITH disambiguation?
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  SECTION B: Minimal set with disambiguation (multiple values present)")
    print(f"{'=' * 80}")

    # B1: base + target full needle + all distractor values
    keep = sorted(set(base + target_full_pos + all_dist_val_pos))
    print(f"\n  B1: base + target FULL needle + all distractor values")
    print(f"      {len(keep)} tokens kept")
    gen = generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep, n_kv_heads)
    ok = target_value in gen
    print(f"      Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
    torch.cuda.empty_cache()

    # B2: base + target key+value + all distractor values (no other keys)
    keep = sorted(set(base + target_key_pos + target_val_pos + all_dist_val_pos))
    print(f"\n  B2: base + target key + target value + all distractor values")
    print(f"      {len(keep)} tokens kept")
    gen = generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep, n_kv_heads)
    ok = target_value in gen
    print(f"      Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
    torch.cuda.empty_cache()

    # B3: base + target key+value + all distractor key+values (full info, no sentence structure)
    all_dist_key_pos = []
    for idx, dk_pos in distractor_keys.items():
        all_dist_key_pos.extend(dk_pos)
    keep = sorted(set(base + target_key_pos + target_val_pos + all_dist_key_pos + all_dist_val_pos))
    print(f"\n  B3: base + ALL keys + ALL values (no sentence structure)")
    print(f"      {len(keep)} tokens kept")
    gen = generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep, n_kv_heads)
    ok = target_value in gen
    print(f"      Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
    torch.cuda.empty_cache()

    # B4: base + ALL full needle sentences (everything kept = full context, sanity check)
    all_full_pos = list(target_full_pos)
    for idx, df_pos in distractor_fulls.items():
        all_full_pos.extend(df_pos)
    keep = sorted(set(base + all_full_pos))
    print(f"\n  B4: base + ALL full needle sentences")
    print(f"      {len(keep)} tokens kept")
    gen = generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep, n_kv_heads)
    ok = target_value in gen
    print(f"      Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
    torch.cuda.empty_cache()

    # B5: base + target key+value + 5 nearest distractor FULL needles
    nearby_full_pos = []
    for idx in nearby_5:
        nearby_full_pos.extend(distractor_fulls[idx])
    keep = sorted(set(base + target_key_pos + target_val_pos + nearby_full_pos))
    print(f"\n  B5: base + target key+value + 5 nearest distractor FULL needles")
    print(f"      {len(keep)} tokens kept")
    gen = generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep, n_kv_heads)
    ok = target_value in gen
    print(f"      Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
    torch.cuda.empty_cache()

    # B6: base + target value ONLY + 5 nearest distractor FULL needles
    keep = sorted(set(base + target_val_pos + nearby_full_pos))
    print(f"\n  B6: base + target value ONLY + 5 nearest distractor FULL needles")
    print(f"      {len(keep)} tokens kept")
    gen = generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep, n_kv_heads)
    ok = target_value in gen
    print(f"      Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # SECTION C: Does the model really need the key for disambiguation?
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  SECTION C: Key's role in disambiguation")
    print(f"{'=' * 80}")

    # C1: base + target full needle + 5 nearest full distractors
    keep = sorted(set(base + target_full_pos + nearby_full_pos))
    print(f"\n  C1: base + target FULL needle + 5 nearest FULL distractors")
    print(f"      {len(keep)} tokens kept")
    gen = generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep, n_kv_heads)
    ok = target_value in gen
    print(f"      Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
    torch.cuda.empty_cache()

    # C2: base + target full needle + ALL full distractors
    keep = sorted(set(base + all_full_pos))
    print(f"\n  C2: base + ALL full needles (= ~full context)")
    print(f"      {len(keep)} tokens kept")
    gen = generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep, n_kv_heads)
    ok = target_value in gen
    print(f"      Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
    torch.cuda.empty_cache()

    # C3: base + target key ONLY + ALL full distractors (target has key but NO value)
    keep = sorted(set(base + target_key_pos + [p for f in distractor_fulls.values() for p in f]))
    print(f"\n  C3: base + target key ONLY + ALL full distractors (no target value!)")
    print(f"      {len(keep)} tokens kept")
    gen = generate_with_keep(model, tokenizer, ctx_ids, q_ids, keep, n_kv_heads)
    ok = target_value in gen
    print(f"      Result: {'OK' if ok else 'FAIL'}  {gen[:100]}")
    torch.cuda.empty_cache()

    print("\n\nDONE")


if __name__ == "__main__":
    main()
