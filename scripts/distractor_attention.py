#!/usr/bin/env python3
"""Analyze attention patterns FROM distractor tokens.
How do tokens at various positions attend to earlier content?
"""

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(42)


def make_sample(num_distractors=50, target_key="brave-falcon", target_value="4829301"):
    adjectives = ["silent", "golden", "dusty", "hollow", "bitter", "crimson",
                  "gentle", "fierce", "mystic", "frozen", "swift", "ancient",
                  "bright", "dark", "wild", "calm", "bold", "shy", "proud", "humble"]
    nouns = ["rabbit", "hawk", "maple", "pine", "stream", "tide", "wolf",
             "eagle", "river", "stone", "flame", "shadow", "storm", "leaf",
             "crystal", "thunder", "ocean", "moon", "star", "wind"]

    needles = []
    for _ in range(num_distractors):
        key = f"{random.choice(adjectives)}-{random.choice(nouns)}"
        value = str(random.randint(1000000, 9999999))
        needles.append(f"One of the special magic numbers for {key} is: {value}.")

    target_needle = f"One of the special magic numbers for {target_key} is: {target_value}."
    target_pos = random.randint(num_distractors // 4, 3 * num_distractors // 4)
    needles.insert(target_pos, target_needle)

    context = "\n".join(needles)
    prompt = (
        f"A special magic number is hidden within the following text. "
        f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {target_key} mentioned in the provided text?"
    )
    return prompt, target_pos


def build_line_ranges(decoded):
    lines = []
    line_start = 0
    for i, tok in enumerate(decoded):
        if "\n" in tok or i == len(decoded) - 1:
            lines.append((line_start, i + 1))
            line_start = i + 1
    return lines


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    prompt, target_pos = make_sample(50)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    tokens = input_ids[0].tolist()
    decoded = [tokenizer.decode([t]) for t in tokens]

    lines = build_line_ranges(decoded)
    line_texts = []
    needle_line = None
    for i, (s, e) in enumerate(lines):
        text = "".join(decoded[s:e]).replace("\n", "")
        line_texts.append(text)
        if "brave-falcon" in text and "4829301" in text:
            needle_line = i

    print("=" * 80)
    print("DISTRACTOR ATTENTION ANALYSIS")
    print(f"Needle at line {needle_line}, seq_len={seq_len}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()

    ids = input_ids.to(model.device)
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(ids, output_attentions=True, return_dict=True)

    attentions = outputs.attentions
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    # ═══════════════════════════════════════════════════════════
    # Pick query lines to analyze: several distractors + needle + question
    # ═══════════════════════════════════════════════════════════
    query_lines = [
        (1, "distractor_1st (line 1)"),
        (5, "distractor_early (line 5)"),
        (10, "distractor_mid_before (line 10)"),
        (19, "distractor_just_before_needle (line 19)"),
        (needle_line, "NEEDLE (line 20)"),
        (21, "distractor_just_after_needle (line 21)"),
        (30, "distractor_mid_after (line 30)"),
        (40, "distractor_late (line 40)"),
        (50, "distractor_2nd_last (line 50)"),
        (51, "distractor_last (line 51)"),
        (52, "QUESTION (line 52)"),
    ]

    # ═══════════════════════════════════════════════════════════
    # [1] For each query line, show per-LINE attention (avg all heads)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[1] PER-LINE ATTENTION FROM LAST TOKEN OF EACH LINE")
    print("    (avg over all layers and heads)")
    print("    Each column = a query line's last token looking at all earlier lines")
    print("=" * 80)

    # Use last token of each query line as the query position
    query_positions = []
    for line_idx, label in query_lines:
        if line_idx < len(lines):
            pos = lines[line_idx][1] - 1  # last token of line
            query_positions.append((pos, line_idx, label))

    # Compute attention from each query position to each target line
    print(f"\n  {'Target line':>35} |", end="")
    for pos, li, label in query_positions:
        short = f"L{li}"
        print(f" {short:>7}", end="")
    print(" |")
    print(f"  {'-'*35}-|" + "-" * (8 * len(query_positions)) + "|")

    for target_li in range(len(lines)):
        ts, te = lines[target_li]
        text = line_texts[target_li][:30]
        marker = ""
        if target_li == needle_line:
            marker = " ◄NEEDLE"
        elif target_li == 0:
            marker = " ◄INSTR"
        elif target_li == len(lines) - 1:
            marker = " ◄QUESTION"

        label = f"{text}{marker}"
        print(f"  {label:>35} |", end="")

        for pos, query_li, qlabel in query_positions:
            if target_li > query_li:
                # Can't attend to future lines (causal)
                print(f"     -- ", end="")
                continue

            # Avg attention from pos to target line
            attn_sum = 0.0
            for l in range(num_layers):
                a = attentions[l][0, :, pos, ts:min(te, pos+1)].float().mean(dim=0).sum().item()
                attn_sum += a
            attn_sum /= num_layers
            pct = attn_sum * 100
            print(f" {pct:>6.2f}%", end="")

        print(" |")

    # ═══════════════════════════════════════════════════════════
    # [2] Detailed view: what does each distractor's KEY token attend to?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[2] KEY TOKEN ATTENTION (the animal-name token in each line)")
    print("    For the distinctive key token in each distractor line,")
    print("    what does it attend to? (avg all layers & heads)")
    print("=" * 80)

    # For each line, find the key token (the hyphenated animal name)
    # It's the token right after "for " in each line
    key_positions = {}
    for li, (s, e) in enumerate(lines):
        for i in range(s, e):
            tok = decoded[i]
            if "-" in tok and len(tok.strip()) > 1 and i > s + 3:
                key_positions[li] = i
                break

    print(f"\n  For each line's key token, attention to: instruction | needle line | own line | other distractors")
    print(f"  {'Line':>4} {'Key token':>15} {'pos':>5} | {'→ instr':>8} {'→ needle':>8} {'→ own':>8} {'→ other':>8} | L7H22 →needle")
    print(f"  {'-'*90}")

    ns, ne = lines[needle_line]

    for li in sorted(key_positions.keys()):
        if li == 0 or li >= len(lines) - 1:
            continue
        pos = key_positions[li]
        tok = decoded[pos].replace("\n", "\\n").strip()[:12]
        ls, le = lines[li]

        # Avg attention
        a_instr = 0
        a_needle = 0
        a_own = 0
        a_other = 0
        for l in range(num_layers):
            avec = attentions[l][0, :, pos, :pos+1].float().mean(dim=0)  # (pos+1,)
            a_instr += avec[:27].sum().item()
            a_needle += avec[ns:min(ne, pos+1)].sum().item()
            a_own += avec[ls:min(le, pos+1)].sum().item()
            a_other += avec.sum().item() - avec[:27].sum().item() - avec[ns:min(ne, pos+1)].sum().item() - avec[ls:min(le, pos+1)].sum().item()
        a_instr /= num_layers
        a_needle /= num_layers
        a_own /= num_layers
        a_other /= num_layers

        # L7H22 specifically
        if pos < attentions[7].shape[2]:
            a7 = attentions[7][0, 22, pos, :pos+1].float()
            l7_needle = a7[ns:min(ne, pos+1)].sum().item() if pos >= ns else 0
        else:
            l7_needle = 0

        marker = " ◄NEEDLE" if li == needle_line else ""
        print(f"  {li:>4} {tok:>15} {pos:>5} | {a_instr*100:>7.2f}% {a_needle*100:>7.4f}% {a_own*100:>7.2f}% {a_other*100:>7.2f}% | {l7_needle*100:>7.4f}%{marker}")

    # ═══════════════════════════════════════════════════════════
    # [3] Retrieval head view: does L7H22 treat needle differently
    #     when processing distractor lines?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[3] RETRIEVAL HEAD (L7H22): ATTENTION TO NEEDLE FROM EVERY LINE'S LAST TOKEN")
    print("    Does L7H22 pay special attention to the needle even during distractor processing?")
    print("=" * 80)

    print(f"\n  {'Line':>4} {'Text':>40} | {'→ needle (L7H22)':>18} {'→ needle (avg)':>16} {'→ instr (L7H22)':>17}")
    print(f"  {'-'*100}")

    for li, (s, e) in enumerate(lines):
        pos = e - 1  # last token of line
        text = line_texts[li][:35]

        # L7H22
        a7 = attentions[7][0, 22, pos, :pos+1].float()
        l7_needle = a7[ns:min(ne, pos+1)].sum().item() if pos >= ns else 0
        l7_instr = a7[:min(27, pos+1)].sum().item()

        # Avg all heads
        avg_needle = 0
        for l in range(num_layers):
            avec = attentions[l][0, :, pos, :pos+1].float().mean(dim=0)
            avg_needle += avec[ns:min(ne, pos+1)].sum().item() if pos >= ns else 0
        avg_needle /= num_layers

        marker = ""
        if li == needle_line:
            marker = " ◄NEEDLE"
        elif li == 0:
            marker = " ◄INSTR"
        elif li == len(lines) - 1:
            marker = " ◄QUESTION"

        bar = "█" * int(l7_needle * 200)
        print(f"  {li:>4} {text:>40} | {l7_needle*100:>14.4f}%   {avg_needle*100:>12.4f}%   {l7_instr*100:>13.2f}% {marker} {bar}")

    # ═══════════════════════════════════════════════════════════
    # [4] Token-by-token within a distractor line: internal attention pattern
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[4] INTERNAL ATTENTION PATTERN WITHIN A DISTRACTOR LINE")
    print("    How does attention distribute within a typical distractor?")
    print("    Using line 10 as example.")
    print("=" * 80)

    example_line = 10
    es, ee = lines[example_line]
    print(f"\n  Line {example_line}: '{line_texts[example_line]}'")
    print(f"  Tokens [{es}, {ee})")

    print(f"\n  For each token in this line (avg all layers & heads):")
    print(f"  {'Pos':>5} {'Token':>15} | {'→ instr':>8} {'→ prev_dist':>11} {'→ own_line':>10} | {'→ sink(pos0)':>13}")
    print(f"  {'-'*75}")

    for i in range(es, ee):
        tok = decoded[i].replace("\n", "\\n").strip()[:12]

        attn_sum_instr = 0
        attn_sum_prev = 0
        attn_sum_own = 0
        attn_sum_sink = 0
        for l in range(num_layers):
            avec = attentions[l][0, :, i, :i+1].float().mean(dim=0)
            attn_sum_instr += avec[:27].sum().item()
            attn_sum_prev += avec[27:es].sum().item()
            attn_sum_own += avec[es:i+1].sum().item()
            attn_sum_sink += avec[0].item()
        attn_sum_instr /= num_layers
        attn_sum_prev /= num_layers
        attn_sum_own /= num_layers
        attn_sum_sink /= num_layers

        print(f"  {i:>5} {tok:>15} | {attn_sum_instr*100:>7.2f}% {attn_sum_prev*100:>10.2f}% {attn_sum_own*100:>9.2f}% | {attn_sum_sink*100:>12.2f}%")

    # Same for needle line
    print(f"\n  For comparison, NEEDLE line {needle_line}:")
    print(f"  '{line_texts[needle_line]}'")
    print(f"  {'Pos':>5} {'Token':>15} | {'→ instr':>8} {'→ prev_dist':>11} {'→ own_line':>10} | {'→ sink(pos0)':>13}")
    print(f"  {'-'*75}")

    for i in range(ns, ne):
        tok = decoded[i].replace("\n", "\\n").strip()[:12]

        attn_sum_instr = 0
        attn_sum_prev = 0
        attn_sum_own = 0
        attn_sum_sink = 0
        for l in range(num_layers):
            avec = attentions[l][0, :, i, :i+1].float().mean(dim=0)
            attn_sum_instr += avec[:27].sum().item()
            attn_sum_prev += avec[27:ns].sum().item()
            attn_sum_own += avec[ns:i+1].sum().item()
            attn_sum_sink += avec[0].item()
        attn_sum_instr /= num_layers
        attn_sum_prev /= num_layers
        attn_sum_own /= num_layers
        attn_sum_sink /= num_layers

        print(f"  {i:>5} {tok:>15} | {attn_sum_instr*100:>7.2f}% {attn_sum_prev*100:>10.02f}% {attn_sum_own*100:>9.02f}% | {attn_sum_sink*100:>12.02f}%")

    print("\nDONE")

    del model, attentions, outputs
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
