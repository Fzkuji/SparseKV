#!/usr/bin/env python3
"""Test KVzip on the representative failure case.

KVzip uses context reconstruction (2-3× overhead) to score KV importance.
We intercept score_val before reset to see which value digits are kept/evicted.
"""

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress.presses.kvzip_press import KVzipPress

ADJECTIVES = ["silent", "golden", "dusty", "hollow", "bitter", "crimson",
              "gentle", "fierce", "mystic", "frozen", "swift", "ancient",
              "bright", "dark", "wild", "calm", "bold", "shy", "proud", "humble"]
NOUNS = ["rabbit", "hawk", "maple", "pine", "stream", "tide", "wolf",
         "eagle", "river", "stone", "flame", "shadow", "storm", "leaf",
         "crystal", "thunder", "ocean", "moon", "star", "wind"]

RETRIEVAL_LAYERS = [
    (7, 5, "L7 KV5 (H22)"),
    (13, 3, "L13 KV3 (H14)"),
    (9, 2, "L9 KV2 (H9)"),
    (15, 3, "L15 KV3 (H13)"),
]


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


# ─── Monkey-patch to save score_val before reset ───
_saved_scores = {}
_test_counter = [0]

_orig_reset = KVzipPress._reset_internal_parameters

def _patched_reset(self):
    if hasattr(self, 'score_val') and self.score_val is not None:
        _saved_scores[_test_counter[0]] = self.score_val.clone().cpu()
        print(f"    [DEBUG] Saved score_val #{_test_counter[0]}: shape={self.score_val.shape}")
    _orig_reset(self)

KVzipPress._reset_internal_parameters = _patched_reset


def analyze_scores(scores, vs, ve, ks, ke, decoded, seq_len, label):
    """Analyze which value digits are kept/evicted based on scores."""
    print(f"\n  {label}")
    print(f"    score_val shape: {scores.shape}")
    n_layer, bsz, n_kv_heads, ctx_len = scores.shape
    print(f"    ctx_len in scores: {ctx_len}")

    if ctx_len < ve:
        print(f"    WARNING: ctx_len ({ctx_len}) < val_end ({ve}), cannot analyze")
        return

    for layer_idx, kv_head, name in RETRIEVAL_LAYERS:
        if layer_idx >= n_layer or kv_head >= n_kv_heads:
            continue
        layer_scores = scores[layer_idx, 0, kv_head]  # (ctx_len,)
        val_scores = layer_scores[vs:ve].float().tolist()
        key_scores = layer_scores[ks:ke].float().tolist()

        # Compute ranks
        sorted_indices = layer_scores.argsort(descending=True)
        ranks = torch.zeros_like(layer_scores, dtype=torch.long)
        ranks[sorted_indices] = torch.arange(len(layer_scores))
        val_ranks = ranks[vs:ve].tolist()

        n_kept = int(ctx_len * 0.5)
        kept = [(i, decoded[i].strip()) for i in range(vs, ve) if ranks[i] < n_kept]
        evicted = [(i, decoded[i].strip()) for i in range(vs, ve) if ranks[i] >= n_kept]

        print(f"\n    {name}:")
        print(f"      Value digit scores: {[f'{s:.6f}' for s in val_scores]}")
        print(f"      Value digit ranks:  {[f'{r}/{ctx_len}' for r in val_ranks]}")
        print(f"      Kept (cr=0.5):  {kept} ({len(kept)}/7)")
        print(f"      Evicted:        {evicted} ({len(evicted)}/7)")
        print(f"      Key scores:     {[f'{s:.6f}' for s in key_scores]}")


def main():
    target_key = "mystic-thunder"
    target_value = "7156842"

    print("=" * 80)
    print("KVZIP TEST: Context reconstruction scoring")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    # ─── Test case ───
    random.seed(42 + 1)
    prompt = make_sample(30, target_key, target_value, 0.1)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    info = find_needle_tokens(tokenizer, input_ids, target_key, target_value)

    seq_len = input_ids.shape[1]
    vs, ve = info["val_start"], info["val_end"]
    ks, ke = info["key_start"], info["key_end"]
    decoded = info["decoded"]

    print(f"\n  Seq len: {seq_len}")
    print(f"  VAL positions: {vs}-{ve}")
    print(f"  KEY positions: {ks}-{ke}")
    print(f"  Value tokens: {[decoded[i].strip() for i in range(vs, ve)]}")
    print(f"  Key tokens: {[decoded[i].strip() for i in range(ks, ke)]}")

    # ─── Full KV baseline ───
    print("\n" + "=" * 80)
    print("[0] FULL KV BASELINE")
    print("=" * 80)
    with torch.no_grad():
        full_out = model.generate(input_ids, max_new_tokens=60, do_sample=False)
    full_gen = tokenizer.decode(full_out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Output: {full_gen[:120]}")
    print(f"  Contains value: {target_value in full_gen}")

    # ─── KVzip layerwise ───
    print("\n" + "=" * 80)
    print("[1] KVZIP (cr=0.5, layerwise=True)")
    print("=" * 80)

    _test_counter[0] = 1
    kvzip = KVzipPress(compression_ratio=0.5, layerwise=True)

    with torch.no_grad(), kvzip(model):
        kvzip_out = model.generate(input_ids, max_new_tokens=60, do_sample=False,
                                    temperature=1.0, top_p=1.0)

    kvzip_gen = tokenizer.decode(kvzip_out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Output: {kvzip_gen[:120]}")
    print(f"  Contains value: {target_value in kvzip_gen}")

    if 1 in _saved_scores:
        analyze_scores(_saved_scores[1], vs, ve, ks, ke, decoded, seq_len, "KVzip layerwise scores:")

    # ─── KVzip non-uniform ───
    print("\n" + "=" * 80)
    print("[2] KVZIP (cr=0.5, layerwise=False)")
    print("=" * 80)

    _test_counter[0] = 2
    kvzip2 = KVzipPress(compression_ratio=0.5, layerwise=False)

    with torch.no_grad(), kvzip2(model):
        kvzip2_out = model.generate(input_ids, max_new_tokens=60, do_sample=False,
                                     temperature=1.0, top_p=1.0)

    kvzip2_gen = tokenizer.decode(kvzip2_out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Output: {kvzip2_gen[:120]}")
    print(f"  Contains value: {target_value in kvzip2_gen}")

    if 2 in _saved_scores:
        analyze_scores(_saved_scores[2], vs, ve, ks, ke, decoded, seq_len, "KVzip non-uniform scores:")

    # ─── KVzip+ ───
    print("\n" + "=" * 80)
    print("[3] KVZIP+ (cr=0.5, layerwise=True, normalization=True)")
    print("=" * 80)

    _test_counter[0] = 3
    kvzip_plus = KVzipPress(compression_ratio=0.5, layerwise=True, kvzip_plus_normalization=True)

    with torch.no_grad(), kvzip_plus(model):
        kvzip_plus_out = model.generate(input_ids, max_new_tokens=60, do_sample=False,
                                         temperature=1.0, top_p=1.0)

    kvzip_plus_gen = tokenizer.decode(kvzip_plus_out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Output: {kvzip_plus_gen[:120]}")
    print(f"  Contains value: {target_value in kvzip_plus_gen}")

    if 3 in _saved_scores:
        analyze_scores(_saved_scores[3], vs, ve, ks, ke, decoded, seq_len, "KVzip+ scores:")

    # ─── Summary ───
    print("\n" + "=" * 80)
    print("[4] SUMMARY")
    print("=" * 80)
    results = [
        ("Full KV", full_gen),
        ("KVzip (layerwise)", kvzip_gen),
        ("KVzip (non-uniform)", kvzip2_gen),
        ("KVzip+", kvzip_plus_gen),
    ]
    for name, gen in results:
        has_val = target_value in gen
        status = "CORRECT" if has_val else "WRONG"
        print(f"  {name:<25s}: {status}  output={gen[:60]}")

    print("\n\nDONE")


if __name__ == "__main__":
    main()
