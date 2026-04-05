#!/usr/bin/env python3
"""Test KVzip using the correct kvpress pipeline pattern.

The correct pattern is:
1. with press(model): model.model(context_ids, past_key_values=cache)  # prefill only
2. KVzip compression happens when context manager exits
3. Generate with compressed cache OUTSIDE the context manager
"""

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvpress.presses.kvzip_press import KVzipPress
from kvpress import SnapKVPress, KnormPress

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
    for i, tok in enumerate(decoded):
        pass  # just decode all
    # Find the needle line
    line_start = 0
    lines = []
    for i, tok in enumerate(decoded):
        if "\n" in tok or i == len(decoded) - 1:
            lines.append((line_start, i + 1))
            line_start = i + 1
    val_start = val_end = key_start = key_end = None
    for li, (s, e) in enumerate(lines):
        line_text = "".join(decoded[s:e])
        if target_key in line_text and target_value in line_text:
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
        "key_start": key_start, "key_end": key_end,
        "val_start": val_start, "val_end": val_end,
        "decoded": decoded,
    }


def generate_greedy(model, tokenizer, cache, context_length, max_new_tokens=60):
    """Generate greedily using a pre-filled cache, like kvpress pipeline does."""
    # Add the question suffix (assistant prompt)
    generated_ids = []
    position_ids = torch.tensor([[context_length]], device=model.device)

    # First token: get from cache
    # We need a dummy input to kick off generation
    # Actually the cache already has the full context. We just need to start generating.
    # Use the last token position to get the first generated token
    dummy_input = torch.tensor([[tokenizer.eos_token_id]], device=model.device)

    # Better approach: just call model with empty input to get next token prediction
    # Actually we should use the approach from kvpress pipeline
    with torch.no_grad():
        outputs = model(
            input_ids=dummy_input,
            past_key_values=cache,
            position_ids=position_ids,
            num_logits_to_keep=1,
        )
    next_id = outputs.logits[0, -1].argmax()
    generated_ids.append(next_id)

    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    for i in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(
                input_ids=next_id.unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i + 1,
            )
        next_id = outputs.logits[0, -1].argmax()
        generated_ids.append(next_id)
        if next_id.item() in eos_ids:
            break

    return tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)


# ─── Monkey-patch to save score_val ───
_saved_scores = {}
_test_id = [0]
_orig_reset = KVzipPress._reset_internal_parameters

def _patched_reset(self):
    if hasattr(self, 'score_val') and self.score_val is not None:
        _saved_scores[_test_id[0]] = self.score_val.clone().cpu()
        print(f"    [DEBUG] Saved score_val #{_test_id[0]}: shape={self.score_val.shape}")
    _orig_reset(self)

KVzipPress._reset_internal_parameters = _patched_reset


def analyze_scores(scores, vs, ve, ks, ke, decoded, label):
    """Analyze which value digits are kept/evicted based on scores."""
    print(f"\n  {label}")
    n_layer, bsz, n_kv_heads, ctx_len = scores.shape
    print(f"    score_val shape: {scores.shape}, ctx_len={ctx_len}")

    if ctx_len < ve:
        print(f"    WARNING: ctx_len ({ctx_len}) < val_end ({ve}), scores invalid")
        return

    for layer_idx, kv_head, name in RETRIEVAL_LAYERS:
        if layer_idx >= n_layer or kv_head >= n_kv_heads:
            continue
        layer_scores = scores[layer_idx, 0, kv_head]
        val_scores = layer_scores[vs:ve].float().tolist()

        sorted_indices = layer_scores.argsort(descending=True)
        ranks = torch.zeros_like(layer_scores, dtype=torch.long)
        ranks[sorted_indices] = torch.arange(len(layer_scores))
        val_ranks = ranks[vs:ve].tolist()

        n_kept = int(ctx_len * 0.5)
        kept = [(i, decoded[i].strip()) for i in range(vs, ve) if ranks[i] < n_kept]
        evicted = [(i, decoded[i].strip()) for i in range(vs, ve) if ranks[i] >= n_kept]

        print(f"\n    {name}:")
        print(f"      Value digit scores: {[f'{s:.6f}' for s in val_scores]}")
        print(f"      Ranks: {[f'{r}/{ctx_len}' for r in val_ranks]}")
        print(f"      Kept (cr=0.5):  {kept} ({len(kept)}/7)")
        print(f"      Evicted:        {evicted} ({len(evicted)}/7)")


def test_press_pipeline(model, tokenizer, context_ids, question_ids, press_obj, label, info, target_value):
    """Test a press using the correct kvpress pipeline pattern."""
    print(f"\n  --- {label} ---")

    cache = DynamicCache()
    context_length = context_ids.shape[1]

    # Step 1: Prefill with press (compression happens on context manager exit)
    with torch.no_grad(), press_obj(model):
        model.model(input_ids=context_ids, past_key_values=cache)

    # Check compressed cache length
    compressed_len = cache.get_seq_length()
    print(f"    Context length: {context_length} → Compressed: {compressed_len} (ratio: {1 - compressed_len/context_length:.2f})")

    # Step 2: Feed question into compressed cache
    q_len = question_ids.shape[1]
    position_ids = torch.arange(compressed_len, compressed_len + q_len, device=model.device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            input_ids=question_ids,
            past_key_values=cache,
            position_ids=position_ids,
            num_logits_to_keep=1,
        )

    # Step 3: Greedy decode
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
    print(f"    Output: {gen_text[:100]}")
    print(f"    Contains value: {correct}")
    return gen_text, correct


def main():
    target_key = "mystic-thunder"
    target_value = "7156842"

    print("=" * 80)
    print("KVZIP CORRECT PIPELINE TEST")
    print("Using kvpress pattern: prefill in context manager, generate outside")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    # Build prompt
    random.seed(42 + 1)
    prompt = make_sample(30, target_key, target_value, 0.1)
    messages = [{"role": "user", "content": prompt}]

    # Split into context and question like kvpress pipeline does
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

    # Also make full input for needle finding
    full_input_ids = torch.cat([context_ids, question_ids], dim=1)
    info = find_needle_tokens(tokenizer, full_input_ids, target_key, target_value)

    vs, ve = info["val_start"], info["val_end"]
    ks, ke = info["key_start"], info["key_end"]
    decoded = info["decoded"]

    print(f"\n  Context length: {context_ids.shape[1]}")
    print(f"  Question length: {question_ids.shape[1]}")
    print(f"  Total: {full_input_ids.shape[1]}")
    if vs is not None:
        print(f"  VAL positions: {vs}-{ve}")
        print(f"  Value tokens: {[decoded[i].strip() for i in range(vs, ve)]}")

    # ─── Full KV baseline ───
    print("\n" + "=" * 80)
    print("[0] FULL KV BASELINE (using pipeline pattern)")
    print("=" * 80)

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
    full_gen = tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)
    print(f"  Cache length: {cache.get_seq_length()}")
    print(f"  Output: {full_gen[:100]}")
    print(f"  Contains value: {target_value in full_gen}")

    del cache
    torch.cuda.empty_cache()

    # ─── SnapKV ───
    print("\n" + "=" * 80)
    print("[1] SNAPKV (cr=0.5)")
    print("=" * 80)
    test_press_pipeline(model, tokenizer, context_ids, question_ids,
                         SnapKVPress(compression_ratio=0.5),
                         "SnapKV", info, target_value)
    torch.cuda.empty_cache()

    # ─── Knorm ───
    print("\n" + "=" * 80)
    print("[2] KNORM (cr=0.5)")
    print("=" * 80)
    test_press_pipeline(model, tokenizer, context_ids, question_ids,
                         KnormPress(compression_ratio=0.5),
                         "Knorm", info, target_value)
    torch.cuda.empty_cache()

    # ─── KVzip layerwise ───
    print("\n" + "=" * 80)
    print("[3] KVZIP (cr=0.5, layerwise=True)")
    print("=" * 80)
    _test_id[0] = 3
    gen3, correct3 = test_press_pipeline(model, tokenizer, context_ids, question_ids,
                         KVzipPress(compression_ratio=0.5, layerwise=True),
                         "KVzip layerwise", info, target_value)
    if 3 in _saved_scores and vs is not None:
        analyze_scores(_saved_scores[3], vs, ve, ks, ke, decoded, "KVzip layerwise scores:")
    torch.cuda.empty_cache()

    # ─── KVzip non-uniform ───
    print("\n" + "=" * 80)
    print("[4] KVZIP (cr=0.5, layerwise=False)")
    print("=" * 80)
    _test_id[0] = 4
    gen4, correct4 = test_press_pipeline(model, tokenizer, context_ids, question_ids,
                         KVzipPress(compression_ratio=0.5, layerwise=False),
                         "KVzip non-uniform", info, target_value)
    if 4 in _saved_scores and vs is not None:
        analyze_scores(_saved_scores[4], vs, ve, ks, ke, decoded, "KVzip non-uniform scores:")
    torch.cuda.empty_cache()

    # ─── Summary ───
    print("\n" + "=" * 80)
    print("[5] SUMMARY")
    print("=" * 80)
    print(f"  Full KV:           {'CORRECT' if target_value in full_gen else 'WRONG'}  {full_gen[:50]}")
    # The other results are printed inline

    print("\n\nDONE")


if __name__ == "__main__":
    main()
