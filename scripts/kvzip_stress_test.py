#!/usr/bin/env python3
"""Stress test KVzip to find failure cases.

Test dimensions:
1. Higher compression ratios (0.5, 0.7, 0.8)
2. More distractors (30, 60, 100, 150)
3. Needle position (0.1=early, 0.5=middle, 0.9=late)
4. Multiple needles (ask about 2nd needle after compressing for 1st)
"""

import torch
import random
import itertools
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvpress.presses.kvzip_press import KVzipPress
from kvpress import SnapKVPress

ADJECTIVES = ["silent", "golden", "dusty", "hollow", "bitter", "crimson",
              "gentle", "fierce", "mystic", "frozen", "swift", "ancient",
              "bright", "dark", "wild", "calm", "bold", "shy", "proud", "humble"]
NOUNS = ["rabbit", "hawk", "maple", "pine", "stream", "tide", "wolf",
         "eagle", "river", "stone", "flame", "shadow", "storm", "leaf",
         "crystal", "thunder", "ocean", "moon", "star", "wind"]


def make_sample(num_distractors, target_key, target_value, needle_position_frac,
                extra_needles=None):
    """Create a needle-in-a-haystack prompt.
    extra_needles: list of (key, value) tuples for additional needles.
    """
    needles = []
    used_keys = {target_key}
    if extra_needles:
        for k, v in extra_needles:
            used_keys.add(k)

    for _ in range(num_distractors):
        while True:
            key = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
            if key not in used_keys:
                used_keys.add(key)
                break
        value = str(random.randint(1000000, 9999999))
        needles.append(f"One of the special magic numbers for {key} is: {value}.")

    # Insert target needle
    target_needle = f"One of the special magic numbers for {target_key} is: {target_value}."
    target_pos = max(0, min(num_distractors, int(num_distractors * needle_position_frac)))
    needles.insert(target_pos, target_needle)

    # Insert extra needles at different positions
    if extra_needles:
        for i, (k, v) in enumerate(extra_needles):
            extra_needle = f"One of the special magic numbers for {k} is: {v}."
            # Spread extra needles across the text
            frac = (i + 1) / (len(extra_needles) + 1)
            pos = max(0, min(len(needles), int(len(needles) * frac)))
            needles.insert(pos, extra_needle)

    context = "\n".join(needles)
    prompt = (
        f"A special magic number is hidden within the following text. "
        f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {target_key} mentioned in the provided text?"
    )
    return prompt


def run_test(model, tokenizer, prompt, target_value, press_obj, label):
    """Run a single test with a given press."""
    # Split into context and question
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

    context_length = context_ids.shape[1]

    # Prefill with press
    cache = DynamicCache()
    with torch.no_grad(), press_obj(model):
        model.model(input_ids=context_ids, past_key_values=cache)

    compressed_len = cache.get_seq_length()

    # Feed question
    q_len = question_ids.shape[1]
    position_ids = torch.arange(compressed_len, compressed_len + q_len, device=model.device).unsqueeze(0)
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
    cur_pos = compressed_len + q_len
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
    correct = target_value in gen_text
    del cache
    return {
        "label": label,
        "context_length": context_length,
        "compressed_length": compressed_len,
        "correct": correct,
        "output": gen_text[:120],
    }


def run_full_kv(model, tokenizer, prompt, target_value):
    """Run full KV baseline."""
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

    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=context_ids, past_key_values=cache)

    q_pos = torch.arange(cache.get_seq_length(), cache.get_seq_length() + question_ids.shape[1],
                          device=model.device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids=question_ids, past_key_values=cache, position_ids=q_pos, num_logits_to_keep=1)

    generated_ids = [outputs.logits[0, -1].argmax()]
    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    cur_pos = cache.get_seq_length()
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
    correct = target_value in gen_text
    ctx_len = context_ids.shape[1]
    del cache
    return {
        "label": "FullKV",
        "context_length": ctx_len,
        "compressed_length": ctx_len,
        "correct": correct,
        "output": gen_text[:120],
    }


def main():
    target_key = "mystic-thunder"
    target_value = "7156842"

    print("=" * 90)
    print("KVZIP STRESS TEST: Finding failure cases")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    results = []

    # ═══════════════════════════════════════════════════════
    # Test 1: Varying compression ratio
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TEST 1: Varying compression ratio (30 distractors, needle at 10%)")
    print("=" * 90)

    random.seed(43)
    prompt = make_sample(30, target_key, target_value, 0.1)

    # Full KV baseline
    res = run_full_kv(model, tokenizer, prompt, target_value)
    res["test"] = "cr_sweep"
    res["cr"] = 0.0
    res["n_dist"] = 30
    res["pos"] = 0.1
    results.append(res)
    print(f"  FullKV: ctx={res['context_length']}, {'OK' if res['correct'] else 'FAIL'}, {res['output'][:60]}")
    torch.cuda.empty_cache()

    for cr in [0.3, 0.5, 0.7, 0.8]:
        for method_name, press_cls, kwargs in [
            ("SnapKV", SnapKVPress, {}),
            ("KVzip", KVzipPress, {"layerwise": True}),
        ]:
            press = press_cls(compression_ratio=cr, **kwargs)
            res = run_test(model, tokenizer, prompt, target_value, press, f"{method_name}_cr{cr}")
            res["test"] = "cr_sweep"
            res["cr"] = cr
            res["method"] = method_name
            res["n_dist"] = 30
            res["pos"] = 0.1
            results.append(res)
            status = "OK" if res["correct"] else "FAIL"
            print(f"  {method_name} cr={cr}: ctx={res['context_length']}, {status}, {res['output'][:60]}")
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # Test 2: Varying context length (more distractors)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TEST 2: Varying context length (cr=0.5, needle at 10%)")
    print("=" * 90)

    for n_dist in [30, 60, 100, 150]:
        random.seed(44 + n_dist)
        prompt = make_sample(n_dist, target_key, target_value, 0.1)

        # Full KV baseline
        res = run_full_kv(model, tokenizer, prompt, target_value)
        res["test"] = "length_sweep"
        res["cr"] = 0.0
        res["n_dist"] = n_dist
        res["pos"] = 0.1
        results.append(res)
        print(f"  n={n_dist} FullKV: ctx={res['context_length']}, {'OK' if res['correct'] else 'FAIL'}")
        torch.cuda.empty_cache()

        for method_name, press_cls, kwargs in [
            ("SnapKV", SnapKVPress, {}),
            ("KVzip", KVzipPress, {"layerwise": True}),
        ]:
            press = press_cls(compression_ratio=0.5, **kwargs)
            res = run_test(model, tokenizer, prompt, target_value, press, f"{method_name}_n{n_dist}")
            res["test"] = "length_sweep"
            res["cr"] = 0.5
            res["method"] = method_name
            res["n_dist"] = n_dist
            res["pos"] = 0.1
            results.append(res)
            status = "OK" if res["correct"] else "FAIL"
            print(f"  n={n_dist} {method_name}: ctx={res['context_length']}, {status}, {res['output'][:60]}")
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # Test 3: Needle position (lost in the middle)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TEST 3: Needle position (100 distractors, cr=0.5)")
    print("=" * 90)

    for pos in [0.1, 0.3, 0.5, 0.7, 0.9]:
        random.seed(45)
        prompt = make_sample(100, target_key, target_value, pos)

        res = run_full_kv(model, tokenizer, prompt, target_value)
        res["test"] = "position_sweep"
        res["cr"] = 0.0
        res["n_dist"] = 100
        res["pos"] = pos
        results.append(res)
        print(f"  pos={pos} FullKV: ctx={res['context_length']}, {'OK' if res['correct'] else 'FAIL'}")
        torch.cuda.empty_cache()

        for method_name, press_cls, kwargs in [
            ("SnapKV", SnapKVPress, {}),
            ("KVzip", KVzipPress, {"layerwise": True}),
        ]:
            press = press_cls(compression_ratio=0.5, **kwargs)
            res = run_test(model, tokenizer, prompt, target_value, press, f"{method_name}_pos{pos}")
            res["test"] = "position_sweep"
            res["cr"] = 0.5
            res["method"] = method_name
            res["n_dist"] = 100
            res["pos"] = pos
            results.append(res)
            status = "OK" if res["correct"] else "FAIL"
            print(f"  pos={pos} {method_name}: ctx={res['context_length']}, {status}, {res['output'][:60]}")
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # Test 4: High compression + long context combo
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TEST 4: High compression + long context (150 distractors, needle at 50%)")
    print("=" * 90)

    random.seed(46)
    prompt = make_sample(150, target_key, target_value, 0.5)

    res = run_full_kv(model, tokenizer, prompt, target_value)
    res["test"] = "hard_combo"
    results.append(res)
    print(f"  FullKV: ctx={res['context_length']}, {'OK' if res['correct'] else 'FAIL'}")
    torch.cuda.empty_cache()

    for cr in [0.5, 0.7, 0.8]:
        for method_name, press_cls, kwargs in [
            ("SnapKV", SnapKVPress, {}),
            ("KVzip", KVzipPress, {"layerwise": True}),
        ]:
            press = press_cls(compression_ratio=cr, **kwargs)
            res = run_test(model, tokenizer, prompt, target_value, press, f"{method_name}_hard_cr{cr}")
            res["test"] = "hard_combo"
            res["cr"] = cr
            res["method"] = method_name
            results.append(res)
            status = "OK" if res["correct"] else "FAIL"
            print(f"  cr={cr} {method_name}: ctx={res['context_length']}, {status}, {res['output'][:60]}")
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # Test 5: Multiple different seeds (robustness)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TEST 5: Robustness (5 seeds, 100 distractors, cr=0.5, pos=0.5)")
    print("=" * 90)

    target_keys_values = [
        ("mystic-thunder", "7156842"),
        ("golden-hawk", "3928451"),
        ("silent-river", "8462019"),
        ("fierce-storm", "5137264"),
        ("frozen-crystal", "9284637"),
    ]

    for tk, tv in target_keys_values:
        random.seed(hash(tk) % 10000)
        prompt = make_sample(100, tk, tv, 0.5)

        res = run_full_kv(model, tokenizer, prompt, tv)
        res["test"] = "robustness"
        res["target_key"] = tk
        results.append(res)
        print(f"  {tk} FullKV: {'OK' if res['correct'] else 'FAIL'}")
        torch.cuda.empty_cache()

        for method_name, press_cls, kwargs in [
            ("SnapKV", SnapKVPress, {}),
            ("KVzip", KVzipPress, {"layerwise": True}),
        ]:
            press = press_cls(compression_ratio=0.5, **kwargs)
            res = run_test(model, tokenizer, prompt, tv, press, f"{method_name}_{tk}")
            res["test"] = "robustness"
            res["cr"] = 0.5
            res["method"] = method_name
            res["target_key"] = tk
            results.append(res)
            status = "OK" if res["correct"] else "FAIL"
            print(f"  {tk} {method_name}: {status}, {res['output'][:60]}")
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Group by test
    for test_name in ["cr_sweep", "length_sweep", "position_sweep", "hard_combo", "robustness"]:
        test_results = [r for r in results if r.get("test") == test_name]
        if not test_results:
            continue
        print(f"\n  {test_name}:")
        for r in test_results:
            status = "OK" if r["correct"] else "FAIL"
            label = r["label"]
            print(f"    {label:<30s}: {status}  ctx={r['context_length']}  out={r['output'][:50]}")

    # Count failures
    print("\n  Failure summary:")
    kvzip_results = [r for r in results if r.get("method") == "KVzip"]
    snapkv_results = [r for r in results if r.get("method") == "SnapKV"]
    fullkv_results = [r for r in results if r["label"] == "FullKV"]

    kvzip_fails = sum(1 for r in kvzip_results if not r["correct"])
    snapkv_fails = sum(1 for r in snapkv_results if not r["correct"])
    fullkv_fails = sum(1 for r in fullkv_results if not r["correct"])

    print(f"    FullKV:  {fullkv_fails}/{len(fullkv_results)} failures")
    print(f"    SnapKV:  {snapkv_fails}/{len(snapkv_results)} failures")
    print(f"    KVzip:   {kvzip_fails}/{len(kvzip_results)} failures")

    # Save raw results
    with open("kvzip_stress_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n  Results saved to kvzip_stress_results.json")

    print("\n\nDONE")


if __name__ == "__main__":
    main()
