#!/usr/bin/env python3
"""Deeper analysis of structural tokens in prefill attention.

Two key questions:
1. Do other structural tokens (colon ':', space ' ', newline, 'is', 'One') also
   broadly attend to value digits like '.' does?
2. Do the QUESTION tokens (last W tokens, used by SnapKV for scoring) attend to
   the '.' tokens? If question → '.' → value digits, then '.' is a relay.
"""

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        "lines": lines, "decoded": decoded,
    }


def main():
    target_key = "mystic-thunder"
    target_value = "7156842"
    num_distractors = 30
    needle_pos_frac = 0.1
    sample_idx = 1

    random.seed(42 + sample_idx)
    prompt = make_sample(num_distractors, target_key, target_value, needle_pos_frac)

    print("=" * 80)
    print("STRUCTURAL TOKEN ANALYSIS")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    seq_len = input_ids.shape[1]
    print(f"\n  Sequence length: {seq_len}")

    info = find_needle_tokens(tokenizer, input_ids, target_key, target_value)
    decoded = info["decoded"]
    ns, ne = info["needle_start"], info["needle_end"]
    ks, ke = info["key_start"], info["key_end"]
    vs, ve = info["val_start"], info["val_end"]

    print(f"\n  Needle: pos {ns}-{ne}, KEY: {ks}-{ke}, VAL: {vs}-{ve}")
    print(f"  Needle tokens:")
    for i in range(ns, ne):
        tag = "KEY" if ks <= i < ke else ("VAL" if vs <= i < ve else "ctx")
        print(f"    {i:4d} [{tag}] '{decoded[i]}'")

    # Identify structural tokens in the needle line
    # Needle line: "One of the special magic numbers for mystic-thunder is: 7156842.\n"
    # Structural tokens: "One", "is", ":", " " (before value), "." (after value), "\n" (in ".\n")
    struct_tokens = {}
    for i in range(ns, ne):
        t = decoded[i].strip()
        if t == ":":
            struct_tokens.setdefault("colon ':'", []).append(i)
        elif t == "is":
            struct_tokens.setdefault("'is'", []).append(i)
        elif t == "." or (t == "" and "\n" in decoded[i] and "." in decoded[i]):
            # Period might be combined with newline
            struct_tokens.setdefault("period '.'", []).append(i)
        elif decoded[i].strip() == "" and i > ns and i < ve:
            # Space token between : and value
            if decoded[i-1].strip() == ":":
                struct_tokens.setdefault("space after ':'", []).append(i)

    # The period/newline token - check if combined
    period_pos = ve  # right after last value digit
    struct_tokens["period '.' (pos {})".format(period_pos)] = [period_pos]
    print(f"\n  Period/newline token at pos {period_pos}: repr='{repr(decoded[period_pos])}'")

    # Find ALL structural tokens in the ENTIRE sequence
    all_period_positions = []
    all_colon_positions = []
    all_newline_positions = []
    all_is_positions = []
    all_space_after_colon = []
    for i in range(seq_len):
        t = decoded[i]
        ts = t.strip()
        if "." in t and "\n" in t:
            all_period_positions.append(i)
        elif ts == ".":
            all_period_positions.append(i)
        if ts == ":":
            all_colon_positions.append(i)
        if "\n" in t:
            all_newline_positions.append(i)
        if ts == "is":
            all_is_positions.append(i)
        if ts == "" and i > 0 and decoded[i-1].strip() == ":":
            all_space_after_colon.append(i)

    print(f"\n  Structural token counts across entire sequence:")
    print(f"    Period '.' positions: {len(all_period_positions)} tokens")
    print(f"    Colon ':' positions: {len(all_colon_positions)} tokens")
    print(f"    Space after ':' positions: {len(all_space_after_colon)} tokens")
    print(f"    'is' positions: {len(all_is_positions)} tokens")

    # Identify question tokens (last part of sequence)
    # Find where question starts - look for "What is the special"
    question_start = None
    for i in range(seq_len - 1, 0, -1):
        if "What" in decoded[i]:
            question_start = i
            break
    if question_start is None:
        # Fallback: last 30 tokens
        question_start = seq_len - 30

    print(f"\n  Question starts at pos {question_start}")
    print(f"  Question tokens ({question_start} to {seq_len-1}):")
    for i in range(question_start, min(question_start + 20, seq_len)):
        print(f"    {i:4d} '{decoded[i]}'")
    if seq_len - 1 > question_start + 20:
        print(f"    ... ({seq_len - 1 - question_start} tokens total)")

    # ─── Run prefill ───
    print(f"\n  Running prefill with output_attentions=True...")
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, use_cache=False)

    num_layers = len(out.attentions)
    num_heads = out.attentions[0].shape[1]
    num_kv_heads = 8
    heads_per_group = num_heads // num_kv_heads

    # ================================================================
    # PART 1: Compare structural tokens' attention to value digits
    # ================================================================
    print("\n" + "=" * 80)
    print("[1] STRUCTURAL TOKEN COMPARISON: Which tokens in needle attend to value digits?")
    print("    Comparing: period '.', colon ':', space ' ', 'is', last value digit")
    print("=" * 80)

    # Tokens to analyze as queries (all in the needle line)
    query_tokens = {}
    for i in range(ns, ne):
        ts = decoded[i].strip()
        if ts == ":":
            query_tokens["colon ':' (pos {})".format(i)] = i
        elif ts == "is":
            query_tokens["'is' (pos {})".format(i)] = i
        elif ts == "" and i > 0 and decoded[i-1].strip() == ":":
            query_tokens["space after ':' (pos {})".format(i)] = i
    query_tokens["period '.' (pos {})".format(period_pos)] = period_pos
    query_tokens["last digit '{}' (pos {})".format(decoded[ve-1].strip(), ve-1)] = ve - 1
    # Also check the first token of the NEXT line if it exists
    if ne < seq_len:
        query_tokens["next line start '{}' (pos {})".format(decoded[ne].strip(), ne)] = ne

    # For each query token, compute average attention to value digits across all layers
    print(f"\n  Average attention to value digits (avg across all 36 layers, all 32 heads):")
    print(f"  {'Query token':<40s} | {'→ val total':>10s} | per-digit distribution")
    print(f"  " + "-" * 120)

    for name, qi in sorted(query_tokens.items(), key=lambda x: x[1]):
        # Can only attend to positions <= qi (causal)
        visible_val = [vi for vi in range(vs, ve) if vi < qi]
        if not visible_val:
            print(f"  {name:<40s} | {'N/A':>10s} | (cannot see value digits due to causal mask)")
            continue

        total_avg = 0
        per_digit_avg = [0.0] * len(visible_val)
        for layer_idx in range(num_layers):
            attn = out.attentions[layer_idx][0]  # (32, seq, seq)
            avg_attn = attn[:, qi, :].mean(dim=0)  # (seq,)
            for di, vi in enumerate(visible_val):
                a = avg_attn[vi].float().item() * 100
                per_digit_avg[di] += a
                total_avg += a
        total_avg /= num_layers
        per_digit_avg = [a / num_layers for a in per_digit_avg]

        digits_str = ", ".join(f"'{decoded[vi].strip()}'={a:.2f}%" for vi, a in zip(visible_val, per_digit_avg))
        print(f"  {name:<40s} | {total_avg:>9.2f}% | [{digits_str}]")

    # Now per-layer detail for the key structural tokens
    key_queries = {
        "colon ':'": None,
        "space after ':'": None,
        "period '.'": period_pos,
        "last digit": ve - 1,
    }
    for i in range(ns, ne):
        ts = decoded[i].strip()
        if ts == ":" and key_queries["colon ':'"] is None:
            key_queries["colon ':'"] = i
        if ts == "" and i > 0 and decoded[i-1].strip() == ":" and key_queries["space after ':'"] is None:
            key_queries["space after ':'"] = i

    print(f"\n  Per-layer total attention to value digits (avg across 32 heads):")
    print(f"  {'Layer':>5s}", end="")
    for name in ["colon ':'", "space ' '", "period '.'", "last digit"]:
        print(f" | {name:>12s}", end="")
    print()
    print(f"  " + "-" * 70)

    for layer_idx in range(num_layers):
        attn = out.attentions[layer_idx][0]
        print(f"  L{layer_idx:>3d}", end="")
        for name, qi in [("colon ':'", key_queries["colon ':'"]),
                         ("space ' '", key_queries["space after ':'"]),
                         ("period '.'", period_pos),
                         ("last digit", ve - 1)]:
            if qi is None:
                print(f" | {'N/A':>12s}", end="")
                continue
            visible_val = [vi for vi in range(vs, ve) if vi < qi]
            if not visible_val:
                print(f" | {'causal':>12s}", end="")
                continue
            avg_attn = attn[:, qi, :].mean(dim=0)
            total = sum(avg_attn[vi].float().item() * 100 for vi in visible_val)
            print(f" | {total:>11.2f}%", end="")
        print()

    # ================================================================
    # PART 2: Do question tokens attend to period '.' tokens?
    # ================================================================
    print("\n" + "=" * 80)
    print("[2] QUESTION → PERIOD RELAY: Do question tokens attend to '.' tokens?")
    print("    If yes, then question → '.' → value digits forms an importance relay.")
    print("=" * 80)

    # For each question token, check attention to ALL period positions
    print(f"\n  All '.' positions in sequence: {all_period_positions}")
    print(f"  Needle '.' at pos {period_pos}")

    # Average across question tokens and all heads, per layer
    print(f"\n  Question tokens' average attention to ALL '.' tokens vs needle '.' specifically:")
    print(f"  {'Layer':>5s} | {'→ all periods':>14s} | {'→ needle period':>15s} | {'→ all colons':>13s} | {'→ needle val':>13s}")
    print(f"  " + "-" * 80)

    q_range = range(question_start, seq_len)
    n_q = len(q_range)

    for layer_idx in range(num_layers):
        attn = out.attentions[layer_idx][0]  # (32, seq, seq)
        # Average over all question tokens and all heads
        q_attn = attn[:, question_start:seq_len, :].mean(dim=0).mean(dim=0)  # (seq,)

        total_periods = sum(q_attn[p].float().item() * 100 for p in all_period_positions)
        needle_period = q_attn[period_pos].float().item() * 100
        total_colons = sum(q_attn[p].float().item() * 100 for p in all_colon_positions)
        total_val = sum(q_attn[vi].float().item() * 100 for vi in range(vs, ve))

        print(f"  L{layer_idx:>3d} | {total_periods:>13.3f}% | {needle_period:>14.4f}% | {total_colons:>12.3f}% | {total_val:>12.4f}%")

    # ================================================================
    # PART 2b: Per-head detail for question → periods (retrieval layers)
    # ================================================================
    print("\n" + "=" * 80)
    print("[2b] Per-head: Question → needle period '.' (retrieval head layers)")
    print("=" * 80)

    retrieval_layers = [7, 9, 13, 15]
    for layer_idx in retrieval_layers:
        attn = out.attentions[layer_idx][0]
        print(f"\n  Layer {layer_idx}:")
        for h in range(num_heads):
            # Average over question tokens
            q_attn_h = attn[h, question_start:seq_len, :].mean(dim=0)  # (seq,)
            needle_period_attn = q_attn_h[period_pos].float().item() * 100
            all_periods_attn = sum(q_attn_h[p].float().item() * 100 for p in all_period_positions)
            val_attn = sum(q_attn_h[vi].float().item() * 100 for vi in range(vs, ve))
            if needle_period_attn > 0.5 or all_periods_attn > 5.0:
                kvh = h // heads_per_group
                print(f"    H{h:2d} (KV{kvh}): needle '.'={needle_period_attn:.3f}%  "
                      f"all '.'s={all_periods_attn:.2f}%  "
                      f"needle val={val_attn:.3f}%")

    # ================================================================
    # PART 3: Per question token breakdown - which attend most to periods?
    # ================================================================
    print("\n" + "=" * 80)
    print("[3] INDIVIDUAL QUESTION TOKENS → periods and value digits")
    print("    Which specific question tokens attend most to '.' tokens?")
    print("=" * 80)

    print(f"\n  Per question token (avg all heads, avg all layers):")
    print(f"  {'pos':>5s} {'token':>12s} | {'→ all .':>9s} {'→ ndl .':>9s} {'→ all :':>9s} {'→ ndl val':>10s} {'→ ndl key':>10s}")
    print(f"  " + "-" * 80)

    for qi in range(question_start, seq_len):
        total_periods = 0
        needle_per = 0
        total_colons = 0
        total_val = 0
        total_key = 0
        for layer_idx in range(num_layers):
            attn = out.attentions[layer_idx][0]
            avg_attn = attn[:, qi, :].mean(dim=0)
            total_periods += sum(avg_attn[p].float().item() * 100 for p in all_period_positions)
            needle_per += avg_attn[period_pos].float().item() * 100
            total_colons += sum(avg_attn[p].float().item() * 100 for p in all_colon_positions)
            total_val += sum(avg_attn[vi].float().item() * 100 for vi in range(vs, ve))
            total_key += sum(avg_attn[ki].float().item() * 100 for ki in range(ks, ke))
        total_periods /= num_layers
        needle_per /= num_layers
        total_colons /= num_layers
        total_val /= num_layers
        total_key /= num_layers

        print(f"  {qi:>5d} '{decoded[qi].strip():>10s}' | {total_periods:>8.3f}% {needle_per:>8.4f}% "
              f"{total_colons:>8.3f}% {total_val:>9.4f}% {total_key:>9.4f}%")

    # ================================================================
    # PART 4: Compare attention to needle '.' vs other '.' tokens
    # ================================================================
    print("\n" + "=" * 80)
    print("[4] NEEDLE '.' vs OTHER '.' TOKENS: Does needle '.' get special attention?")
    print("    Average question attention to each '.' in the sequence.")
    print("=" * 80)

    print(f"\n  Per '.' position (avg question tokens, avg all heads, avg all layers):")
    print(f"  {'pos':>5s} {'line content':>30s} | {'q→. attention':>14s}")
    print(f"  " + "-" * 60)

    for p in all_period_positions:
        total = 0
        for layer_idx in range(num_layers):
            attn = out.attentions[layer_idx][0]
            q_attn = attn[:, question_start:seq_len, :].mean(dim=0).mean(dim=0)
            total += q_attn[p].float().item() * 100
        total /= num_layers

        # Get context
        ctx = ""
        for li, (s, e) in enumerate(info["lines"]):
            if s <= p < e:
                ctx = f"line {li}"
                break
        marker = " ← NEEDLE" if ns <= p < ne else ""
        print(f"  {p:>5d} {ctx:>15s}{marker:>15s} | {total:>13.4f}%")

    # ================================================================
    # PART 5: The relay hypothesis quantified
    # ================================================================
    print("\n" + "=" * 80)
    print("[5] RELAY HYPOTHESIS: Quantify question → '.' → value indirect signal")
    print("    For each layer: (question→period_attn) × (period→value_attn)")
    print("    Compare with direct: question → value_attn")
    print("=" * 80)

    print(f"\n  {'Layer':>5s} | {'direct q→val':>13s} | {'q→ndl_period':>13s} | {'ndl_period→val':>15s} | {'indirect signal':>16s}")
    print(f"  " + "-" * 80)

    for layer_idx in range(num_layers):
        attn = out.attentions[layer_idx][0]

        # Direct: question → value
        q_attn = attn[:, question_start:seq_len, :].mean(dim=0).mean(dim=0)
        direct_val = sum(q_attn[vi].float().item() * 100 for vi in range(vs, ve))

        # Question → needle period
        q_to_period = q_attn[period_pos].float().item() * 100

        # Needle period → value
        period_attn = attn[:, period_pos, :].mean(dim=0)
        period_to_val = sum(period_attn[vi].float().item() * 100 for vi in range(vs, ve))

        # Indirect signal (product, as proxy for information flow)
        indirect = q_to_period * period_to_val / 100

        print(f"  L{layer_idx:>3d} | {direct_val:>12.4f}% | {q_to_period:>12.4f}% | {period_to_val:>14.2f}% | {indirect:>15.4f}%")

    del out
    torch.cuda.empty_cache()
    print("\n\nDONE")


if __name__ == "__main__":
    main()
