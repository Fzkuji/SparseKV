#!/usr/bin/env python3
"""Compare different KV press methods on known failure cases.

Test: SnapKV, CriticalKV(SnapKV), ExpectedAttention, Knorm
on the same representative failure case + a batch of failures.

For each method: which value digits are kept/evicted per retrieval head?
"""

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress import SnapKVPress, ExpectedAttentionPress, KnormPress
from kvpress.presses.criticalkv_press import CriticalKVPress
from kvpress.presses.scorer_press import ScorerPress

ADJECTIVES = ["silent", "golden", "dusty", "hollow", "bitter", "crimson",
              "gentle", "fierce", "mystic", "frozen", "swift", "ancient",
              "bright", "dark", "wild", "calm", "bold", "shy", "proud", "humble"]
NOUNS = ["rabbit", "hawk", "maple", "pine", "stream", "tide", "wolf",
         "eagle", "river", "stone", "flame", "shadow", "storm", "leaf",
         "crystal", "thunder", "ocean", "moon", "star", "wind"]


def make_sample(num_distractors, target_key, target_value, needle_position_frac):
    needles = []
    used_keys = set()
    for _ in range(num_distractors):
        while True:
            key = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
            if key != target_key and key not in used_keys:
                used_keys.add(key)
                break
        value = str(random.randint(1000000, 9999999))
        needles.append(f"One of the special magic numbers for {key} is: {value}.")
    target_needle = f"One of the special magic numbers for {target_key} is: {target_value}."
    target_pos = max(0, min(num_distractors, int(num_distractors * needle_position_frac)))
    needles.insert(target_pos, target_needle)
    context = "\n".join(needles)
    prompt = (
        f"A special magic number is hidden within the following text. "
        f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {target_key} mentioned in the provided text?"
    )
    return prompt


def find_needle_tokens(tokenizer, input_ids, target_key, target_value):
    tokens = input_ids[0].tolist()
    decoded = [tokenizer.decode([t]) for t in tokens]
    lines = []
    line_start = 0
    for i, tok in enumerate(decoded):
        if "\n" in tok or i == len(decoded) - 1:
            lines.append((line_start, i + 1))
            line_start = i + 1
    needle_start = needle_end = None
    key_start = key_end = val_start = val_end = None
    for li, (s, e) in enumerate(lines):
        line_text = "".join(decoded[s:e])
        if target_key in line_text and target_value in line_text:
            needle_start, needle_end = s, e
            char_offset = 0
            tok_char_ranges = []
            for i in range(s, e):
                tok_len = len(decoded[i])
                tok_char_ranges.append((char_offset, char_offset + tok_len, i))
                char_offset += tok_len
            kcs = line_text.find(target_key)
            kce = kcs + len(target_key)
            if kcs >= 0:
                for cs, ce, ti in tok_char_ranges:
                    if cs < kce and ce > kcs:
                        if key_start is None: key_start = ti
                        key_end = ti + 1
            vcs = line_text.find(target_value)
            vce = vcs + len(target_value)
            if vcs >= 0:
                for cs, ce, ti in tok_char_ranges:
                    if cs < vce and ce > vcs:
                        if val_start is None: val_start = ti
                        val_end = ti + 1
            break
    return {
        "needle_start": needle_start, "needle_end": needle_end,
        "key_start": key_start, "key_end": key_end,
        "val_start": val_start, "val_end": val_end,
        "decoded": decoded,
    }


# ─── Eviction logging ───
_eviction_log = []
_orig_compress = ScorerPress.compress

def _debug_compress(self, module, hidden_states, keys, values, attentions, kwargs):
    scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
    k_len = keys.shape[2]
    n_kept = int(k_len * (1 - self.compression_ratio))
    kept_indices = scores.topk(n_kept, dim=-1).indices
    _eviction_log.append({
        "layer": module.layer_idx,
        "kept_indices": kept_indices[0].cpu(),
        "scores": scores[0].cpu().float(),
    })
    return _orig_compress(self, module, hidden_states, keys, values, attentions, kwargs)

ScorerPress.compress = _debug_compress


def run_with_press(model, tokenizer, input_ids, press_obj, target_value, info, max_new=40):
    global _eviction_log
    _eviction_log = []

    with torch.no_grad(), press_obj(model):
        out = model.generate(
            input_ids, max_new_tokens=max_new, do_sample=False,
            temperature=1.0, top_p=1.0,
        )

    generated = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Check value digit retention
    vs, ve = info["val_start"], info["val_end"]
    ks, ke = info["key_start"], info["key_end"]
    decoded = info["decoded"]

    # Retrieval heads: L7H22(KV5), L13H14(KV3), L9H9(KV2), L15H13(KV3)
    retrieval_heads = [(7, 5, "L7H22"), (13, 3, "L13H14"), (9, 2, "L9H9"), (15, 3, "L15H13")]

    retention = {}
    for layer_idx, kv_head, name in retrieval_heads:
        layer_log = None
        for entry in _eviction_log:
            if entry["layer"] == layer_idx:
                layer_log = entry
                break
        if layer_log is None:
            retention[name] = "no_log"
            continue

        kept = layer_log["kept_indices"][kv_head].tolist()  # (n_kept,)
        val_kept = [i for i in range(vs, ve) if i in kept]
        val_evicted = [i for i in range(vs, ve) if i not in kept]
        key_kept = [i for i in range(ks, ke) if i in kept]
        retention[name] = {
            "val_kept": [(i, decoded[i].strip()) for i in val_kept],
            "val_evicted": [(i, decoded[i].strip()) for i in val_evicted],
            "key_kept_count": len(key_kept),
            "key_total": ke - ks,
        }

    correct = target_value in generated[:20]
    return generated, correct, retention


def main():
    target_key = "mystic-thunder"
    target_value = "7156842"

    print("=" * 80)
    print("METHOD COMPARISON: SnapKV vs CriticalKV vs ExpectedAttention vs Knorm")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    # ─── Test 1: Representative failure case ───
    print("\n" + "=" * 80)
    print("[1] REPRESENTATIVE CASE: dist=30, pos=0.1, cr=0.5")
    print("=" * 80)

    random.seed(42 + 1)
    prompt = make_sample(30, target_key, target_value, 0.1)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    info = find_needle_tokens(tokenizer, input_ids, target_key, target_value)

    print(f"\n  Seq len: {input_ids.shape[1]}, VAL positions: {info['val_start']}-{info['val_end']}")

    methods = [
        ("SnapKV", SnapKVPress(compression_ratio=0.5)),
        ("CriticalKV(SnapKV)", CriticalKVPress(SnapKVPress(compression_ratio=0.5))),
        ("ExpectedAttention", ExpectedAttentionPress(compression_ratio=0.5)),
        ("Knorm", KnormPress(compression_ratio=0.5)),
    ]

    for method_name, press_obj in methods:
        print(f"\n  --- {method_name} ---")
        try:
            gen, correct, retention = run_with_press(
                model, tokenizer, input_ids, press_obj, target_value, info
            )
            print(f"  Output: {gen[:60]}")
            print(f"  Correct: {correct}")
            for head_name, ret in retention.items():
                if ret == "no_log":
                    print(f"    {head_name}: no eviction log")
                    continue
                kept_str = ", ".join(f"'{d}'" for _, d in ret["val_kept"])
                evicted_str = ", ".join(f"'{d}'" for _, d in ret["val_evicted"])
                print(f"    {head_name}: kept=[{kept_str}]  evicted=[{evicted_str}]  key={ret['key_kept_count']}/{ret['key_total']}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # ─── Test 2: Batch of failure-prone cases ───
    print("\n" + "=" * 80)
    print("[2] BATCH TEST: Multiple cases at cr=0.5")
    print("=" * 80)

    test_cases = [
        (30, 0.1, "mystic-thunder", "7156842", 1),
        (30, 0.25, "mystic-thunder", "7156842", 1),
        (30, 0.5, "mystic-thunder", "7156842", 1),
        (50, 0.1, "swift-crystal", "3847291", 2),
        (50, 0.25, "swift-crystal", "3847291", 2),
        (50, 0.5, "swift-crystal", "3847291", 2),
        (80, 0.1, "bright-ocean", "9261548", 3),
        (80, 0.25, "bright-ocean", "9261548", 3),
        (80, 0.5, "bright-ocean", "9261548", 3),
    ]

    results = {m: {"correct": 0, "total": 0} for m, _ in methods}

    for dist, pos, key, val, seed_offset in test_cases:
        random.seed(42 + seed_offset)
        prompt = make_sample(dist, key, val, pos)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        case_info = find_needle_tokens(tokenizer, input_ids, key, val)

        if case_info["val_start"] is None:
            print(f"\n  SKIP: dist={dist}, pos={pos}, key={key} — needle not found")
            continue

        # First check full KV
        with torch.no_grad():
            full_out = model.generate(input_ids, max_new_tokens=40, do_sample=False)
        full_gen = tokenizer.decode(full_out[0][input_ids.shape[1]:], skip_special_tokens=True)
        if val not in full_gen[:20]:
            print(f"\n  SKIP: dist={dist}, pos={pos}, key={key} — full KV wrong ({full_gen[:30]})")
            continue

        print(f"\n  Case: dist={dist}, pos={pos}, key={key}")
        for method_name, press_obj in methods:
            try:
                # Reset compression ratio in case it was modified
                if hasattr(press_obj, 'compression_ratio'):
                    press_obj.compression_ratio = 0.5
                gen, correct, retention = run_with_press(
                    model, tokenizer, input_ids, press_obj, val, case_info
                )
                results[method_name]["total"] += 1
                if correct:
                    results[method_name]["correct"] += 1
                status = "✓" if correct else "✗"
                # Count kept value digits for L7H22
                l7_ret = retention.get("L7H22", {})
                if isinstance(l7_ret, dict):
                    n_kept = len(l7_ret.get("val_kept", []))
                else:
                    n_kept = "?"
                print(f"    {method_name:<25s}: {status}  L7H22 val kept={n_kept}/7  out={gen[:30]}")
            except Exception as e:
                print(f"    {method_name:<25s}: ERROR {e}")

    # ─── Summary ───
    print("\n" + "=" * 80)
    print("[3] SUMMARY")
    print("=" * 80)
    for method_name, _ in methods:
        r = results[method_name]
        if r["total"] > 0:
            acc = r["correct"] / r["total"] * 100
            print(f"  {method_name:<25s}: {r['correct']}/{r['total']} correct ({acc:.1f}%)")
        else:
            print(f"  {method_name:<25s}: no valid cases")

    print("\n\nDONE")


if __name__ == "__main__":
    main()
