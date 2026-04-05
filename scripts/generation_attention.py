#!/usr/bin/env python3
"""Analyze attention during token-by-token generation.
Compare full KV vs SnapKV: what does each generated digit attend to?

Uses the representative failure case:
  dist=30, pos=0.1, key=mystic-thunder, value=7156842
  Full KV → correct (7156842)
  SnapKV cr=0.5 → wrong (7247410 or similar)
"""

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress import SnapKVPress
from kvpress.presses.scorer_press import ScorerPress

# ─── Eviction logging ───
_eviction_log = []
_orig_compress = ScorerPress.compress

def _debug_compress(self, module, hidden_states, keys, values, attentions, kwargs):
    scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
    k_len = keys.shape[2]
    n_kept = int(k_len * (1 - self.compression_ratio))
    kept_indices = scores.topk(n_kept, dim=-1).indices  # (B, H, n_kept)
    _eviction_log.append({
        "layer": module.layer_idx,
        "kept_indices": kept_indices[0].cpu(),  # (H, n_kept)
        "scores": scores[0].cpu(),
    })
    return _orig_compress(self, module, hidden_states, keys, values, attentions, kwargs)

ScorerPress.compress = _debug_compress


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
    sample_idx = 1  # matches failure #0 from previous analysis

    random.seed(42 + sample_idx)
    prompt = make_sample(num_distractors, target_key, target_value, needle_pos_frac)

    print("=" * 80)
    print("GENERATION ATTENTION ANALYSIS: Full KV vs SnapKV")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    seq_len = input_ids.shape[1]

    ni = find_needle_tokens(tokenizer, input_ids, target_key, target_value)
    decoded = ni["decoded"]
    ns, ne = ni["needle_start"], ni["needle_end"]
    vs, ve = ni["val_start"], ni["val_end"]
    ks, ke = ni["key_start"], ni["key_end"]

    print(f"\nPrompt: {seq_len} tokens")
    print(f"Needle line: tokens [{ns}, {ne})")
    print(f"Needle key:  tokens [{ks}, {ke})")
    print(f"Needle val:  tokens [{vs}, {ve})")
    print(f"\nNeedle tokens:")
    for i in range(ns, ne):
        cat = "ctx"
        if ks <= i < ke: cat = "KEY"
        elif vs <= i < ve: cat = "VAL"
        print(f"  {i:>4}: [{cat:>3}] '{decoded[i]}'")

    # ═══════════════════════════════════════════════════════════
    # [1] FULL KV: Step-by-step generation with attention
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[1] FULL KV: Token-by-token generation with attention analysis")
    print("=" * 80)

    # Manual generation loop
    gen_ids = input_ids.clone()
    gen_tokens = []
    max_gen = 40

    # First, do a full prefill to get past_key_values
    print("\n  Prefilling...")
    with torch.no_grad():
        out = model(gen_ids, use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    next_token = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    print(f"\n  Generating with attention capture...")
    print(f"\n  {'Step':>4} {'Token':>15} | {'→ needle':>10} {'→ key':>10} {'→ val':>10} {'→ instr':>10} {'→ quest':>10} | L7H22→val  L13H14→val")
    print(f"  {'-'*110}")

    # Identify question and instruction ranges
    instr_end = 27  # approx
    q_start = ni["lines"][-1][0] if ni["lines"] else seq_len - 16

    for step in range(max_gen):
        # Forward with single token, output_attentions
        with torch.no_grad():
            out = model(
                next_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

        past_kv = out.past_key_values
        attentions = out.attentions  # list of (1, num_heads, 1, current_seq_len)

        curr_pos = seq_len + step
        tok_text = tokenizer.decode(next_token[0], skip_special_tokens=False).replace("\n", "\\n")
        gen_tokens.append((next_token[0, 0].item(), tok_text))

        # Compute attention from this generated token to various regions
        num_layers = len(attentions)
        kv_len = attentions[0].shape[-1]  # should be curr_pos + 1

        # Average over all layers and heads
        avg_attn = torch.zeros(kv_len)
        for l in range(num_layers):
            avg_attn += attentions[l][0, :, 0, :].float().mean(dim=0).cpu()
        avg_attn /= num_layers

        a_needle = avg_attn[ns:ne].sum().item() * 100
        a_key = avg_attn[ks:ke].sum().item() * 100
        a_val = avg_attn[vs:ve].sum().item() * 100
        a_instr = avg_attn[:instr_end].sum().item() * 100
        a_quest = avg_attn[q_start:seq_len].sum().item() * 100

        # Retrieval heads specifically
        # L7H22: query head 22
        a7 = attentions[7][0, 22, 0, :].float().cpu()
        l7_val = a7[vs:ve].sum().item() * 100

        # L13H14: query head 14
        a13 = attentions[13][0, 14, 0, :].float().cpu()
        l13_val = a13[vs:ve].sum().item() * 100

        print(f"  {step:>4} {tok_text:>15} | {a_needle:>9.3f}% {a_key:>9.3f}% {a_val:>9.3f}% {a_instr:>9.2f}% {a_quest:>9.2f}% | {l7_val:>9.3f}% {l13_val:>9.3f}%")

        # Show per-needle-token attention for digit steps (when generating digits)
        if tok_text.strip() in "0123456789":
            print(f"        Attention to each needle token (avg all heads):")
            for i in range(ns, ne):
                pct = avg_attn[i].item() * 100
                cat = "ctx"
                if ks <= i < ke: cat = "KEY"
                elif vs <= i < ve: cat = "VAL"
                bar = "█" * int(pct * 20)
                print(f"          {i:>4} [{cat:>3}] '{decoded[i].strip()[:10]:>10}' {pct:>7.3f}% {bar}")

            print(f"        L7H22 attention to each needle token:")
            for i in range(ns, ne):
                pct = a7[i].item() * 100
                cat = "ctx"
                if ks <= i < ke: cat = "KEY"
                elif vs <= i < ve: cat = "VAL"
                bar = "█" * int(pct * 5)
                print(f"          {i:>4} [{cat:>3}] '{decoded[i].strip()[:10]:>10}' {pct:>7.3f}% {bar}")

        # Next token
        next_token = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

        # Stop conditions
        tok_id = gen_tokens[-1][0]
        if tok_id == tokenizer.eos_token_id:
            break

    full_gen_text = "".join(t[1] for t in gen_tokens)
    print(f"\n  Full generation: {full_gen_text[:120]}")
    print(f"  Correct answer: {target_value}")
    print(f"  Match: {target_value in full_gen_text}")

    # Clear attention cache
    del attentions, past_kv, out
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # [2] SNAPKV cr=0.5: Same analysis with eviction
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[2] SNAPKV cr=0.5: Token-by-token generation after eviction")
    print("=" * 80)

    global _eviction_log
    _eviction_log = []

    press = SnapKVPress(compression_ratio=0.5, window_size=64, kernel_size=5)

    # Prefill with press (this triggers eviction)
    print("\n  Prefilling with SnapKV (cr=0.5)...")
    with torch.no_grad(), press(model):
        out = model(input_ids, use_cache=True, return_dict=True)

    past_kv_snap = out.past_key_values
    eviction_log = _eviction_log.copy()
    next_token = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    # Analyze what was kept
    # DynamicCache API: layers[i].keys has shape (B, H, L, D)
    compressed_len = past_kv_snap.layers[0].keys.shape[2]
    print(f"\n  Original seq_len: {seq_len}")
    print(f"  Compressed KV len: {compressed_len}")

    # Show which needle tokens are in the compressed cache
    # The kept_indices tell us which original positions were retained
    print(f"\n  Needle token retention (per retrieval head's KV group):")
    retrieval_map = {"L7H22": (7, 5), "L13H14": (13, 3), "L9H9": (9, 2)}

    # Build position mapping: for each layer, which original positions are kept?
    # The eviction log has kept_indices per layer per head
    layer_kept = {}
    for entry in eviction_log:
        layer = entry["layer"]
        layer_kept[layer] = entry["kept_indices"]  # (H, n_kept)

    for name, (layer_idx, kv_head) in retrieval_map.items():
        if layer_idx in layer_kept:
            kept = set(layer_kept[layer_idx][kv_head].tolist())
            print(f"\n    {name} (layer {layer_idx}, KV head {kv_head}):")
            print(f"    Kept {len(kept)} / {seq_len} positions")
            for i in range(ns, ne):
                cat = "ctx"
                if ks <= i < ke: cat = "KEY"
                elif vs <= i < ve: cat = "VAL"
                status = "✓ KEPT" if i in kept else "✗ EVICTED"
                print(f"      {i:>4} [{cat:>3}] '{decoded[i].strip()[:10]:>10}' {status}")

    # Now generate with attention capture
    print(f"\n  Generating with attention capture...")
    print(f"\n  {'Step':>4} {'Token':>15} | {'→ needle_kept':>14} {'→ instr':>10} {'→ quest':>10} | L7H22→needle  L13H14→needle")
    print(f"  {'-'*100}")

    gen_tokens_snap = []

    # For the compressed cache, we need to map original positions to compressed positions
    # Get the kept indices for a reference layer (say layer 0)
    # Actually, position mapping differs per layer. Let's use layer 7 for L7H22 analysis.
    # The compressed cache has positions [0, 1, ..., compressed_len-1] corresponding to
    # kept_indices sorted.

    for step in range(max_gen):
        with torch.no_grad():
            out = model(
                next_token,
                past_key_values=past_kv_snap,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

        past_kv_snap = out.past_key_values
        attentions = out.attentions

        curr_kv_len = attentions[0].shape[-1]
        tok_text = tokenizer.decode(next_token[0], skip_special_tokens=False).replace("\n", "\\n")
        gen_tokens_snap.append((next_token[0, 0].item(), tok_text))

        # For compressed cache, positions are remapped.
        # The KV cache after SnapKV contains:
        # - First compressed_len positions are the selected tokens from prefill
        # - Then positions after that are newly generated tokens
        # Position i in the compressed cache corresponds to kept_indices[i] in the original

        # Average attention
        num_layers = len(attentions)
        avg_attn = torch.zeros(curr_kv_len)
        for l in range(num_layers):
            avg_attn += attentions[l][0, :, 0, :].float().mean(dim=0).cpu()
        avg_attn /= num_layers

        # Map compressed positions back to original positions for needle analysis
        # Use layer 0 as reference (all layers have same compressed_len but different positions)
        # For a rough view, let's compute attention to positions that WERE needle tokens
        # across all layers

        a_needle_avg = 0
        a_instr_avg = 0
        a_quest_avg = 0
        for l in range(num_layers):
            avec = attentions[l][0, :, 0, :].float().mean(dim=0).cpu()
            kept = layer_kept[l]  # (H, n_kept) - but attention is averaged over query heads...
            # Simpler: use head 0's kept indices as representative for avg
            kept_sorted = sorted(kept[0].tolist())
            for ci, orig_pos in enumerate(kept_sorted):
                if ci < len(avec) - step - 1:  # only compressed prefill positions
                    if ns <= orig_pos < ne:
                        a_needle_avg += avec[ci].item()
                    if orig_pos < instr_end:
                        a_instr_avg += avec[ci].item()
                    if q_start <= orig_pos < seq_len:
                        a_quest_avg += avec[ci].item()
        a_needle_avg = a_needle_avg / num_layers * 100
        a_instr_avg = a_instr_avg / num_layers * 100
        a_quest_avg = a_quest_avg / num_layers * 100

        # L7H22 specifically
        a7 = attentions[7][0, 22, 0, :].float().cpu()
        kept7 = sorted(layer_kept[7][5].tolist())  # KV head 5
        l7_needle = 0
        for ci, orig_pos in enumerate(kept7):
            if ci < len(a7) - step - 1:
                if ns <= orig_pos < ne:
                    l7_needle += a7[ci].item()
        l7_needle *= 100

        # L13H14
        a13 = attentions[13][0, 14, 0, :].float().cpu()
        kept13 = sorted(layer_kept[13][3].tolist())  # KV head 3
        l13_needle = 0
        for ci, orig_pos in enumerate(kept13):
            if ci < len(a13) - step - 1:
                if ns <= orig_pos < ne:
                    l13_needle += a13[ci].item()
        l13_needle *= 100

        print(f"  {step:>4} {tok_text:>15} | {a_needle_avg:>13.3f}% {a_instr_avg:>9.2f}% {a_quest_avg:>9.2f}% | {l7_needle:>11.3f}% {l13_needle:>12.3f}%")

        # For digit tokens, show detailed breakdown
        if tok_text.strip() in "0123456789":
            # Show which needle positions are available and their attention
            print(f"        L7H22 attention to kept needle tokens:")
            for ci, orig_pos in enumerate(kept7):
                if ci < len(a7) - step - 1 and ns <= orig_pos < ne:
                    cat = "ctx"
                    if ks <= orig_pos < ke: cat = "KEY"
                    elif vs <= orig_pos < ve: cat = "VAL"
                    pct = a7[ci].item() * 100
                    bar = "█" * int(pct * 5)
                    print(f"          orig={orig_pos:>4} [{cat:>3}] '{decoded[orig_pos].strip()[:10]:>10}' {pct:>7.3f}% {bar}")
            # Show what it attends to INSTEAD (top-5 positions)
            top5 = a7[:compressed_len].topk(min(5, compressed_len))
            print(f"        L7H22 top-5 attended positions (compressed):")
            for rank, (val, idx) in enumerate(zip(top5.values, top5.indices)):
                ci = idx.item()
                if ci < len(kept7):
                    orig = kept7[ci]
                    orig_tok = decoded[orig].strip()[:12] if orig < len(decoded) else "?"
                    print(f"          comp={ci:>4} → orig={orig:>4} '{orig_tok:>12}' {val.item()*100:>7.3f}%")

        next_token = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        tok_id = gen_tokens_snap[-1][0]
        if tok_id == tokenizer.eos_token_id:
            break

    snap_gen_text = "".join(t[1] for t in gen_tokens_snap)
    print(f"\n  SnapKV generation: {snap_gen_text[:120]}")
    print(f"  Correct answer: {target_value}")
    print(f"  Match: {target_value in snap_gen_text}")

    # ═══════════════════════════════════════════════════════════
    # [3] COMPARISON SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("[3] COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n  Full KV output:  {full_gen_text[:80]}")
    print(f"  SnapKV output:   {snap_gen_text[:80]}")
    print(f"  Correct value:   {target_value}")

    del model, attentions, past_kv_snap, out
    torch.cuda.empty_cache()

    print("\nDONE")


if __name__ == "__main__":
    main()
