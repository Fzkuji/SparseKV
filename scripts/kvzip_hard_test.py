#!/usr/bin/env python3
"""Harder tests to find KVzip's limits.

Focus on:
1. Extreme compression ratios (0.9, 0.95)
2. Very long context (500 distractors ~10K tokens)
3. Multi-hop reasoning (need two pieces of info)
4. Counting/aggregation tasks
"""

import torch
import random
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


def run_pipeline(model, tokenizer, prompt, target_value, press_obj=None, label=""):
    """Run test with optional press. If press_obj is None, use full KV."""
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

    cache = DynamicCache()
    if press_obj is not None:
        with torch.no_grad(), press_obj(model):
            model.model(input_ids=context_ids, past_key_values=cache)
    else:
        with torch.no_grad():
            model.model(input_ids=context_ids, past_key_values=cache)

    compressed_len = cache.get_seq_length()

    q_len = question_ids.shape[1]
    position_ids = torch.arange(compressed_len, compressed_len + q_len, device=model.device).unsqueeze(0)
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
    cur_pos = compressed_len + q_len
    for i in range(99):
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
        "output": gen_text[:150],
    }


def make_kv_pairs(num_distractors, target_key, target_value, needle_position_frac):
    """Standard needle-in-a-haystack."""
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
    target_pos = max(0, min(num_distractors, int(num_distractors * needle_position_frac)))
    needles.insert(target_pos, target_needle)
    return needles


def test_extreme_cr(model, tokenizer, results):
    """Test 1: Extreme compression ratios (0.9, 0.95)."""
    print("\n" + "=" * 90)
    print("TEST 1: Extreme compression ratios (100 distractors, needle at 50%)")
    print("=" * 90)

    target_key, target_value = "mystic-thunder", "7156842"
    random.seed(50)
    needles = make_kv_pairs(100, target_key, target_value, 0.5)
    context = "\n".join(needles)
    prompt = (
        f"A special magic number is hidden within the following text. "
        f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {target_key} mentioned in the provided text?"
    )

    res = run_pipeline(model, tokenizer, prompt, target_value, None, "FullKV")
    res["test"] = "extreme_cr"
    results.append(res)
    print(f"  FullKV: ctx={res['context_length']}, {'OK' if res['correct'] else 'FAIL'}")
    torch.cuda.empty_cache()

    for cr in [0.5, 0.7, 0.8, 0.9, 0.95]:
        for name, cls, kw in [("SnapKV", SnapKVPress, {}), ("KVzip", KVzipPress, {"layerwise": True})]:
            press = cls(compression_ratio=cr, **kw)
            res = run_pipeline(model, tokenizer, prompt, target_value, press, f"{name}_cr{cr}")
            res["test"] = "extreme_cr"
            res["cr"] = cr
            res["method"] = name
            results.append(res)
            status = "OK" if res["correct"] else "FAIL"
            print(f"  {name} cr={cr}: ctx={res['context_length']}, {status}, {res['output'][:60]}")
            torch.cuda.empty_cache()


def test_very_long(model, tokenizer, results):
    """Test 2: Very long context (300, 500 distractors)."""
    print("\n" + "=" * 90)
    print("TEST 2: Very long context (cr=0.5, needle at 50%)")
    print("=" * 90)

    target_key, target_value = "mystic-thunder", "7156842"

    for n_dist in [300, 500]:
        random.seed(51 + n_dist)
        # Use more words to avoid key collisions
        needles = []
        used_keys = {target_key}
        for idx in range(n_dist):
            key = f"item-{idx:04d}"
            used_keys.add(key)
            value = str(random.randint(1000000, 9999999))
            needles.append(f"One of the special magic numbers for {key} is: {value}.")
        target_needle = f"One of the special magic numbers for {target_key} is: {target_value}."
        target_pos = n_dist // 2
        needles.insert(target_pos, target_needle)
        context = "\n".join(needles)
        prompt = (
            f"A special magic number is hidden within the following text. "
            f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
            f"{context}\n"
            f"What is the special magic number for {target_key} mentioned in the provided text?"
        )

        res = run_pipeline(model, tokenizer, prompt, target_value, None, f"FullKV_n{n_dist}")
        res["test"] = "very_long"
        res["n_dist"] = n_dist
        results.append(res)
        print(f"  n={n_dist} FullKV: ctx={res['context_length']}, {'OK' if res['correct'] else 'FAIL'}")
        torch.cuda.empty_cache()

        for name, cls, kw in [("SnapKV", SnapKVPress, {}), ("KVzip", KVzipPress, {"layerwise": True})]:
            press = cls(compression_ratio=0.5, **kw)
            res = run_pipeline(model, tokenizer, prompt, target_value, press, f"{name}_n{n_dist}")
            res["test"] = "very_long"
            res["cr"] = 0.5
            res["method"] = name
            res["n_dist"] = n_dist
            results.append(res)
            status = "OK" if res["correct"] else "FAIL"
            print(f"  n={n_dist} {name}: ctx={res['context_length']}, {status}, {res['output'][:60]}")
            torch.cuda.empty_cache()


def test_multi_hop(model, tokenizer, results):
    """Test 3: Multi-hop reasoning — need info from two separate locations."""
    print("\n" + "=" * 90)
    print("TEST 3: Multi-hop reasoning (cr=0.5)")
    print("=" * 90)

    random.seed(52)

    # Scenario: "X lives in city Y" scattered in text, "city Y has ZIP code Z" scattered elsewhere
    # Question: "What is the ZIP code of the city where X lives?"
    people = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank"]
    cities = ["Springfield", "Riverdale", "Oakville", "Pinehurst", "Maplewood",
              "Cedarburg", "Elmwood", "Ashfield"]
    zips = ["10742", "20853", "30961", "40572", "50483", "60194", "70265", "80376"]

    # Create person-city pairs and city-zip pairs
    person_city = list(zip(people, cities))
    city_zip = list(zip(cities, zips))
    random.shuffle(person_city)
    random.shuffle(city_zip)

    # Target: where does Alice live? → Springfield → ZIP 10742
    target_person = "Alice"
    target_city = "Springfield"
    target_zip = "10742"

    # Build text with interleaved facts and distractors
    facts = []
    for p, c in person_city:
        facts.append(f"{p} currently lives in {c}.")
    # Add some filler
    for _ in range(40):
        facts.append(f"The weather today is {random.choice(['sunny', 'cloudy', 'rainy', 'windy', 'snowy'])} "
                      f"with a temperature of {random.randint(10, 35)} degrees.")
    for c, z in city_zip:
        facts.append(f"The ZIP code of {c} is {z}.")
    # Add more filler
    for _ in range(40):
        facts.append(f"The population of district-{random.randint(100,999)} is {random.randint(1000,99999)}.")

    random.shuffle(facts)
    context = "\n".join(facts)
    prompt = (
        f"Read the following facts carefully and answer the question.\n"
        f"{context}\n"
        f"What is the ZIP code of the city where {target_person} lives? "
        f"Answer with just the ZIP code number."
    )

    for cr_label, press_obj in [
        ("FullKV", None),
        ("SnapKV_cr0.5", SnapKVPress(compression_ratio=0.5)),
        ("KVzip_cr0.5", KVzipPress(compression_ratio=0.5, layerwise=True)),
        ("SnapKV_cr0.7", SnapKVPress(compression_ratio=0.7)),
        ("KVzip_cr0.7", KVzipPress(compression_ratio=0.7, layerwise=True)),
    ]:
        res = run_pipeline(model, tokenizer, prompt, target_zip, press_obj, cr_label)
        res["test"] = "multi_hop"
        results.append(res)
        status = "OK" if res["correct"] else "FAIL"
        print(f"  {cr_label}: ctx={res['context_length']}, {status}, {res['output'][:80]}")
        torch.cuda.empty_cache()


def test_counting(model, tokenizer, results):
    """Test 4: Counting — need to aggregate info across many positions."""
    print("\n" + "=" * 90)
    print("TEST 4: Counting/aggregation (cr=0.5)")
    print("=" * 90)

    random.seed(53)

    # Scenario: count how many items of a specific color appear
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "white", "black", "brown"]
    items = ["apple", "ball", "car", "dress", "egg", "flag", "gem", "hat", "ink", "jar"]

    facts = []
    target_color = "red"
    red_count = 0
    for _ in range(100):
        color = random.choice(colors)
        item = random.choice(items)
        count = random.randint(1, 5)
        facts.append(f"There are {count} {color} {item}s in the warehouse.")
        if color == target_color:
            red_count += count

    random.shuffle(facts)
    context = "\n".join(facts)
    prompt = (
        f"Read the inventory list carefully.\n"
        f"{context}\n"
        f"How many red items are there in total in the warehouse? "
        f"Add up all the red items. Answer with just the number."
    )
    target_answer = str(red_count)

    for cr_label, press_obj in [
        ("FullKV", None),
        ("SnapKV_cr0.5", SnapKVPress(compression_ratio=0.5)),
        ("KVzip_cr0.5", KVzipPress(compression_ratio=0.5, layerwise=True)),
    ]:
        res = run_pipeline(model, tokenizer, prompt, target_answer, press_obj, cr_label)
        res["test"] = "counting"
        res["expected"] = target_answer
        results.append(res)
        status = "OK" if res["correct"] else "FAIL"
        print(f"  {cr_label}: ctx={res['context_length']}, {status} (expected={target_answer}), {res['output'][:80]}")
        torch.cuda.empty_cache()


def test_multi_needle(model, tokenizer, results):
    """Test 5: Multiple needles — retrieve 3 different values."""
    print("\n" + "=" * 90)
    print("TEST 5: Multiple needles (3 targets, cr=0.5)")
    print("=" * 90)

    targets = [
        ("mystic-thunder", "7156842"),
        ("golden-hawk", "3928451"),
        ("silent-river", "8462019"),
    ]

    random.seed(54)
    needles = []
    used_keys = set(k for k, v in targets)
    for _ in range(100):
        while True:
            key = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
            if key not in used_keys:
                used_keys.add(key)
                break
        value = str(random.randint(1000000, 9999999))
        needles.append(f"One of the special magic numbers for {key} is: {value}.")

    # Insert targets at different positions
    for i, (tk, tv) in enumerate(targets):
        needle = f"One of the special magic numbers for {tk} is: {tv}."
        pos = int(len(needles) * (i + 1) / (len(targets) + 1))
        needles.insert(pos, needle)

    context = "\n".join(needles)

    # Ask about each target separately
    for tk, tv in targets:
        prompt = (
            f"Several special magic numbers are hidden within the following text. "
            f"Make sure to memorize them. I will quiz you about the numbers afterwards.\n"
            f"{context}\n"
            f"What is the special magic number for {tk} mentioned in the provided text?"
        )

        for cr_label, press_obj in [
            ("FullKV", None),
            ("SnapKV_cr0.5", SnapKVPress(compression_ratio=0.5)),
            ("KVzip_cr0.5", KVzipPress(compression_ratio=0.5, layerwise=True)),
        ]:
            res = run_pipeline(model, tokenizer, prompt, tv, press_obj, f"{cr_label}_{tk}")
            res["test"] = "multi_needle"
            res["target_key"] = tk
            results.append(res)
            status = "OK" if res["correct"] else "FAIL"
            print(f"  {tk} {cr_label}: {status}, {res['output'][:60]}")
            torch.cuda.empty_cache()


def test_extreme_cr_long(model, tokenizer, results):
    """Test 6: Extreme CR + long context (the hardest combo)."""
    print("\n" + "=" * 90)
    print("TEST 6: Extreme CR + long context (200 dist, cr=0.9)")
    print("=" * 90)

    target_key, target_value = "mystic-thunder", "7156842"
    random.seed(55)

    needles = []
    used_keys = {target_key}
    for idx in range(200):
        key = f"item-{idx:04d}"
        used_keys.add(key)
        value = str(random.randint(1000000, 9999999))
        needles.append(f"One of the special magic numbers for {key} is: {value}.")
    target_needle = f"One of the special magic numbers for {target_key} is: {target_value}."
    needles.insert(100, target_needle)
    context = "\n".join(needles)
    prompt = (
        f"A special magic number is hidden within the following text. "
        f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {target_key} mentioned in the provided text?"
    )

    for cr_label, press_obj in [
        ("FullKV", None),
        ("SnapKV_cr0.9", SnapKVPress(compression_ratio=0.9)),
        ("KVzip_cr0.9", KVzipPress(compression_ratio=0.9, layerwise=True)),
        ("KVzip_cr0.95", KVzipPress(compression_ratio=0.95, layerwise=True)),
    ]:
        res = run_pipeline(model, tokenizer, prompt, target_value, press_obj, cr_label)
        res["test"] = "extreme_cr_long"
        results.append(res)
        status = "OK" if res["correct"] else "FAIL"
        print(f"  {cr_label}: ctx={res['context_length']}, {status}, {res['output'][:60]}")
        torch.cuda.empty_cache()


def main():
    print("=" * 90)
    print("KVZIP HARD TEST: Finding failure boundaries")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    results = []

    test_extreme_cr(model, tokenizer, results)
    test_very_long(model, tokenizer, results)
    test_multi_hop(model, tokenizer, results)
    test_counting(model, tokenizer, results)
    test_multi_needle(model, tokenizer, results)
    test_extreme_cr_long(model, tokenizer, results)

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    for test_name in ["extreme_cr", "very_long", "multi_hop", "counting", "multi_needle", "extreme_cr_long"]:
        test_results = [r for r in results if r.get("test") == test_name]
        if not test_results:
            continue
        print(f"\n  {test_name}:")
        for r in test_results:
            status = "OK" if r["correct"] else "FAIL"
            print(f"    {r['label']:<30s}: {status}  ctx={r['context_length']}  {r['output'][:50]}")

    # Method-level summary
    print("\n  Method failure rates:")
    for method in ["FullKV", "SnapKV", "KVzip"]:
        method_results = [r for r in results if method in r["label"]]
        if method_results:
            fails = sum(1 for r in method_results if not r["correct"])
            print(f"    {method}: {fails}/{len(method_results)} failures")

    with open("kvzip_hard_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n  Results saved to kvzip_hard_results.json")

    print("\n\nDONE")


if __name__ == "__main__":
    main()
