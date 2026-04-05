#!/usr/bin/env python3
"""Attention Ray Tracing: Propagate importance through attention chains.

Idea: Instead of only scoring tokens by question→token attention,
trace chains like:
  question → needle_first_digit → (reverse) period '.' → all_digits

Algorithm:
1. Start with SnapKV scores (question last-W attention per position)
2. For each position with score > threshold, look at what IT attended to
   during prefill (its attention row when it was the query)
3. Add those positions' scores proportionally
4. Compare original vs propagated scores for value digits

This is like "ray tracing" in rendering: shoot rays from the camera (question),
they hit surfaces (tokens), and bounce to illuminate other surfaces (more tokens).
"""

import torch
import random
import torch.nn.functional as F
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
        "decoded": decoded,
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
    print("ATTENTION RAY TRACING: Importance propagation through attention chains")
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

    info = find_needle_tokens(tokenizer, input_ids, target_key, target_value)
    decoded = info["decoded"]
    ns, ne = info["needle_start"], info["needle_end"]
    ks, ke = info["key_start"], info["key_end"]
    vs, ve = info["val_start"], info["val_end"]

    print(f"\n  Seq len: {seq_len}")
    print(f"  Needle: pos {ns}-{ne}, KEY: {ks}-{ke}, VAL: {vs}-{ve}")
    print(f"  Value digits: {[decoded[i].strip() for i in range(vs, ve)]}")

    # ─── Run prefill with attention ───
    print(f"\n  Running prefill with output_attentions=True...")
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, use_cache=False)

    num_layers = len(out.attentions)
    num_heads = out.attentions[0].shape[1]
    num_kv_heads = 8
    heads_per_group = num_heads // num_kv_heads

    # ─── SnapKV-style scoring: last W tokens' attention ───
    W = 64  # typical SnapKV window
    print(f"\n  SnapKV window W={W}")

    print("\n" + "=" * 80)
    print("[1] SNAPKV BASELINE SCORES vs RAY-TRACED SCORES")
    print("    Per layer, per KV head")
    print("=" * 80)

    for layer_idx in range(num_layers):
        attn = out.attentions[layer_idx][0].float()  # (32, seq, seq)

        # SnapKV scoring: average attention from last W query positions
        # Per KV head: average across query heads in the group AND last W positions
        snapkv_scores = torch.zeros(num_kv_heads, seq_len)
        for kv_h in range(num_kv_heads):
            h_start = kv_h * heads_per_group
            h_end = h_start + heads_per_group
            # attention from last W tokens, averaged over query heads in group
            window_attn = attn[h_start:h_end, -W:, :].mean(dim=0).mean(dim=0)  # (seq_len,)
            snapkv_scores[kv_h] = window_attn.cpu()

        # ─── Ray tracing: propagate importance ───
        # For each KV head, compute propagated scores
        raytraced_scores = snapkv_scores.clone()

        for kv_h in range(num_kv_heads):
            h_start = kv_h * heads_per_group
            h_end = h_start + heads_per_group

            # Get full prefill attention for this KV group (avg over query heads)
            group_attn = attn[h_start:h_end, :, :].mean(dim=0)  # (seq, seq)

            # Two directions of importance propagation:
            #
            # FORWARD (who did I attend to?):
            #   If position i is important AND i attended to j during prefill → j is also important
            #   boost_fwd(j) = sum_i [importance(i) * attn(i→j)]
            #
            # BACKWARD (who attended to me?):
            #   If position i is important AND j attended to i during prefill → j is also important
            #   (j's info was read by i, so j carries relevant content)
            #   boost_bwd(j) = sum_i [importance(i) * attn(j→i)]
            #
            # Combined: both directions carry relevance signals

            alpha_fwd = 1.0
            alpha_bwd = 1.0

            importance = snapkv_scores[kv_h].to(group_attn.device)  # (seq,)

            # Forward: importance flows along attention direction
            # group_attn[i, j] = how much position i attended to j
            # boost_fwd[j] = sum_i importance[i] * attn[i, j]
            boost_fwd = (importance.unsqueeze(0) * group_attn).sum(dim=0)  # (seq,)

            # Backward: importance flows against attention direction
            # If i is important and j attended to i → j is related
            # boost_bwd[j] = sum_i importance[i] * attn[j, i]
            # = sum_i importance[i] * attn.T[i, j]
            # = (importance @ attn.T)[j]  but we need attn[j, i] for each j
            # attn.T[i, j] = attn[j, i], so:
            # boost_bwd[j] = sum_i importance[i] * attn[j, i] = (attn @ importance)[j]
            boost_bwd = torch.matmul(group_attn, importance)  # (seq,)

            raytraced_scores[kv_h] += (alpha_fwd * boost_fwd + alpha_bwd * boost_bwd).cpu()

        # Show results for value digits
        any_improvement = False
        for kv_h in range(num_kv_heads):
            snap_val = [snapkv_scores[kv_h, vi].item() * 100 for vi in range(vs, ve)]
            ray_val = [raytraced_scores[kv_h, vi].item() * 100 for vi in range(vs, ve)]
            snap_total = sum(snap_val)
            ray_total = sum(ray_val)

            if ray_total > snap_total * 1.5 and ray_total > 0.1:
                any_improvement = True
                # Compute rank of value digits
                snap_ranks = []
                ray_ranks = []
                for vi in range(vs, ve):
                    snap_rank = (snapkv_scores[kv_h] > snapkv_scores[kv_h, vi]).sum().item() + 1
                    ray_rank = (raytraced_scores[kv_h] > raytraced_scores[kv_h, vi]).sum().item() + 1
                    snap_ranks.append(snap_rank)
                    ray_ranks.append(ray_rank)

                if layer_idx in [7, 9, 13, 15] or ray_total > snap_total * 3:
                    print(f"\n  L{layer_idx:2d} KV{kv_h}:")
                    print(f"    SnapKV scores (×100):  [{', '.join(f'{s:.4f}' for s in snap_val)}]  total={snap_total:.4f}")
                    print(f"    Raytraced scores:      [{', '.join(f'{r:.4f}' for r in ray_val)}]  total={ray_total:.4f}")
                    print(f"    Improvement:           {ray_total/max(snap_total, 1e-6):.1f}×")
                    print(f"    SnapKV ranks:          [{', '.join(f'{r}/{seq_len}' for r in snap_ranks)}]")
                    print(f"    Raytraced ranks:       [{', '.join(f'{r}/{seq_len}' for r in ray_ranks)}]")

                    # Would eviction change? (keep top 50%)
                    n_kept = seq_len // 2
                    snap_kept = set(snapkv_scores[kv_h].topk(n_kept).indices.tolist())
                    ray_kept = set(raytraced_scores[kv_h].topk(n_kept).indices.tolist())
                    snap_val_kept = [vi for vi in range(vs, ve) if vi in snap_kept]
                    ray_val_kept = [vi for vi in range(vs, ve) if vi in ray_kept]
                    print(f"    SnapKV keeps (cr=0.5): {[decoded[v].strip() for v in snap_val_kept]} ({len(snap_val_kept)}/7)")
                    print(f"    Raytrace keeps:        {[decoded[v].strip() for v in ray_val_kept]} ({len(ray_val_kept)}/7)")

    # ─── Multi-hop ray tracing ───
    print("\n" + "=" * 80)
    print("[2] MULTI-HOP RAY TRACING (2 bounces)")
    print("    question → A → B → target")
    print("=" * 80)

    # Focus on key retrieval head layers
    for layer_idx in [7, 9, 13, 15]:
        attn = out.attentions[layer_idx][0].float()

        for kv_h in range(num_kv_heads):
            h_start = kv_h * heads_per_group
            h_end = h_start + heads_per_group
            group_attn = attn[h_start:h_end, :, :].mean(dim=0)  # (seq, seq)

            # SnapKV: 1-hop from question window
            hop0 = group_attn[-W:, :].mean(dim=0)  # (seq,)

            # 1-hop forward: important tokens → what they attended to
            hop1_fwd = torch.matmul(hop0.unsqueeze(0), group_attn).squeeze(0)
            # 1-hop backward: important tokens → who attended to them
            hop1_bwd = torch.matmul(group_attn, hop0)
            hop1 = hop1_fwd + hop1_bwd

            # 2-hop: propagate combined signal again (both directions)
            hop2_fwd = torch.matmul(hop1.unsqueeze(0), group_attn).squeeze(0)
            hop2_bwd = torch.matmul(group_attn, hop1)
            hop2 = hop2_fwd + hop2_bwd

            # Combined score
            combined = hop0 + 0.5 * hop1 + 0.25 * hop2

            # Check value digits
            h0_val = [hop0[vi].item() * 100 for vi in range(vs, ve)]
            h1_val = [hop1[vi].item() * 100 for vi in range(vs, ve)]
            h2_val = [hop2[vi].item() * 100 for vi in range(vs, ve)]
            comb_val = [combined[vi].item() * 100 for vi in range(vs, ve)]

            h0_total = sum(h0_val)
            comb_total = sum(comb_val)

            if comb_total > h0_total * 2 and comb_total > 0.1:
                n_kept = seq_len // 2
                h0_kept = set(hop0.topk(n_kept).indices.tolist())
                comb_kept = set(combined.topk(n_kept).indices.tolist())
                h0_val_kept = [vi for vi in range(vs, ve) if vi in h0_kept]
                comb_val_kept = [vi for vi in range(vs, ve) if vi in comb_kept]

                print(f"\n  L{layer_idx:2d} KV{kv_h}:")
                print(f"    0-hop (SnapKV): [{', '.join(f'{s:.3f}' for s in h0_val)}]  total={h0_total:.3f}")
                print(f"    1-hop:          [{', '.join(f'{s:.3f}' for s in h1_val)}]  total={sum(h1_val):.3f}")
                print(f"    2-hop:          [{', '.join(f'{s:.3f}' for s in h2_val)}]  total={sum(h2_val):.3f}")
                print(f"    Combined:       [{', '.join(f'{s:.3f}' for s in comb_val)}]  total={comb_total:.3f}")
                print(f"    SnapKV keeps (cr=0.5): {[decoded[v].strip() for v in h0_val_kept]} ({len(h0_val_kept)}/7)")
                print(f"    Raytrace keeps:        {[decoded[v].strip() for v in comb_val_kept]} ({len(comb_val_kept)}/7)")

    # ─── Per-head ray tracing for L7H22 specifically ───
    print("\n" + "=" * 80)
    print("[3] L7H22 SPECIFIC: Single query head ray tracing")
    print("    Not averaged across KV group — use H22 alone")
    print("=" * 80)

    attn = out.attentions[7][0].float()
    h22_attn = attn[22]  # (seq, seq) — just query head 22

    # SnapKV: H22's last-W attention
    h22_hop0 = h22_attn[-W:, :].mean(dim=0)
    # 1-hop bidirectional
    h22_hop1_fwd = torch.matmul(h22_hop0.unsqueeze(0), h22_attn).squeeze(0)
    h22_hop1_bwd = torch.matmul(h22_attn, h22_hop0)
    h22_hop1 = h22_hop1_fwd + h22_hop1_bwd
    # 2-hop bidirectional
    h22_hop2_fwd = torch.matmul(h22_hop1.unsqueeze(0), h22_attn).squeeze(0)
    h22_hop2_bwd = torch.matmul(h22_attn, h22_hop1)
    h22_hop2 = h22_hop2_fwd + h22_hop2_bwd
    h22_combined = h22_hop0 + 0.5 * h22_hop1 + 0.25 * h22_hop2

    print(f"\n  L7 H22 (the copy head):")
    for name, scores in [("0-hop (SnapKV)", h22_hop0),
                          ("1-hop", h22_hop1),
                          ("2-hop", h22_hop2),
                          ("Combined", h22_combined)]:
        val_scores = [scores[vi].item() * 100 for vi in range(vs, ve)]
        total = sum(val_scores)
        # Top-10 scored positions
        top10 = scores.topk(10)
        top10_str = ", ".join(f"pos{idx.item()}='{decoded[idx.item()].strip()}'({v.item()*100:.2f}%)"
                             for v, idx in zip(top10.values, top10.indices))
        print(f"\n    {name}:")
        print(f"      Value digits: [{', '.join(f'{s:.4f}' for s in val_scores)}] total={total:.4f}")
        print(f"      Top-10: {top10_str}")

    n_kept = seq_len // 2
    h22_snap_kept = set(h22_hop0.topk(n_kept).indices.tolist())
    h22_ray_kept = set(h22_combined.topk(n_kept).indices.tolist())
    snap_val_kept = [vi for vi in range(vs, ve) if vi in h22_snap_kept]
    ray_val_kept = [vi for vi in range(vs, ve) if vi in h22_ray_kept]

    print(f"\n    SnapKV H22 keeps: {[decoded[v].strip() for v in snap_val_kept]} ({len(snap_val_kept)}/7)")
    print(f"    Raytrace H22 keeps: {[decoded[v].strip() for v in ray_val_kept]} ({len(ray_val_kept)}/7)")

    # ─── Trace the actual chain ───
    print("\n" + "=" * 80)
    print("[4] TRACE THE CHAIN: question → first_digit → period → all_digits")
    print("    Show the actual attention values along the hypothesized chain")
    print("=" * 80)

    # Chain: question_tokens → digit '7' (pos vs) → what does '7' attend to?
    #        question_tokens → key tokens → what do key tokens attend to?
    #        and: who attends to later digits strongly? → period '.' (pos ve)
    period_pos = ve  # '.' is right after last digit

    # For L7H22
    print(f"\n  L7 H22 chain:")
    print(f"    question(last {W}) → digit '7' (pos {vs}): {h22_attn[-W:, vs].mean().item()*100:.4f}%")
    print(f"    question(last {W}) → ':' (pos {vs-2}): {h22_attn[-W:, vs-2].mean().item()*100:.4f}%")
    print(f"    question(last {W}) → ' ' (pos {vs-1}): {h22_attn[-W:, vs-1].mean().item()*100:.4f}%")
    print(f"    question(last {W}) → '.' (pos {period_pos}): {h22_attn[-W:, period_pos].mean().item()*100:.4f}%")

    print(f"\n    digit '7' (query) → previous needle tokens:")
    for i in range(ns, vs):
        a = h22_attn[vs, i].item() * 100
        if a > 0.1:
            tag = "KEY" if ks <= i < ke else "ctx"
            print(f"      pos {i} [{tag}] '{decoded[i].strip()}': {a:.3f}%")

    print(f"\n    '.' (pos {period_pos}, query) → value digits:")
    for vi in range(vs, ve):
        a = h22_attn[period_pos, vi].item() * 100
        print(f"      pos {vi} '{decoded[vi].strip()}': {a:.3f}%")

    # Now check ALL heads at layer 7 for period → value attention
    print(f"\n    All L7 heads: '.' → value digits (total):")
    for h in range(num_heads):
        val_total = sum(attn[h, period_pos, vi].item() * 100 for vi in range(vs, ve))
        if val_total > 5:
            kvh = h // heads_per_group
            per_digit = [attn[h, period_pos, vi].item() * 100 for vi in range(vs, ve)]
            print(f"      H{h:2d} (KV{kvh}): [{', '.join(f'{d:.1f}%' for d in per_digit)}] total={val_total:.1f}%")

    # Check: does the chain work across different heads?
    # question(H22) → digit '7' → (different head at same layer) period → all digits
    print(f"\n    Cross-head chain at L7:")
    print(f"      H22 (question → '7'): {h22_attn[-W:, vs].mean().item()*100:.3f}%")
    print(f"      H25 (KV6, '.' → all val): {sum(attn[25, period_pos, vi].item()*100 for vi in range(vs, ve)):.1f}%")
    print(f"      H18 (KV4, '.' → all val): {sum(attn[18, period_pos, vi].item()*100 for vi in range(vs, ve)):.1f}%")

    del out
    torch.cuda.empty_cache()
    print("\n\nDONE")


if __name__ == "__main__":
    main()
