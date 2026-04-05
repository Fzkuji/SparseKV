#!/usr/bin/env python3
"""Find and analyze cases where KV eviction causes inference failure.

1. Generate multiple NIAH samples (varied needle positions, num distractors)
2. Run with full KV → get ground truth
3. Run with SnapKV/CriticalSnapKV → find failures
4. For failures: capture eviction decisions, attention patterns, retrieval heads
"""

import torch
import random
import json
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvpress import SnapKVPress
from kvpress.presses.scorer_press import ScorerPress

# ─── Eviction logging via monkey-patch ───
_eviction_log = []
_orig_compress = ScorerPress.compress

def _debug_compress(self, module, hidden_states, keys, values, attentions, kwargs):
    scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
    k_len = keys.shape[2]
    n_kept = int(k_len * (1 - self.compression_ratio))
    kept_indices = scores.topk(n_kept, dim=-1).indices  # (B, H, n_kept)
    _eviction_log.append({
        "layer": module.layer_idx,
        "k_len": k_len,
        "n_kept": n_kept,
        "kept_indices": kept_indices[0].cpu(),  # (H, n_kept)
        "scores": scores[0].cpu(),               # (H, k_len)
    })
    return _orig_compress(self, module, hidden_states, keys, values, attentions, kwargs)

ScorerPress.compress = _debug_compress


# ─── Sample generation ───
ADJECTIVES = ["silent", "golden", "dusty", "hollow", "bitter", "crimson",
              "gentle", "fierce", "mystic", "frozen", "swift", "ancient",
              "bright", "dark", "wild", "calm", "bold", "shy", "proud", "humble"]
NOUNS = ["rabbit", "hawk", "maple", "pine", "stream", "tide", "wolf",
         "eagle", "river", "stone", "flame", "shadow", "storm", "leaf",
         "crystal", "thunder", "ocean", "moon", "star", "wind"]


def make_sample(num_distractors, target_key, target_value, needle_position_frac):
    """Generate a NIAH sample with needle at a specific fractional position."""
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
    return prompt, target_pos


def find_needle_tokens(tokenizer, input_ids, target_key, target_value):
    """Find the token positions of needle key and value by reconstructing text positions."""
    tokens = input_ids[0].tolist()
    decoded = [tokenizer.decode([t]) for t in tokens]

    # Build line ranges
    lines = []
    line_start = 0
    for i, tok in enumerate(decoded):
        if "\n" in tok or i == len(decoded) - 1:
            lines.append((line_start, i + 1))
            line_start = i + 1

    # Find needle line
    needle_start = needle_end = None
    key_start = key_end = val_start = val_end = None

    for li, (s, e) in enumerate(lines):
        line_text = "".join(decoded[s:e])
        if target_key in line_text and target_value in line_text:
            needle_start = s
            needle_end = e

            # Reconstruct character offsets to find key and value positions
            char_offset = 0
            tok_char_ranges = []
            for i in range(s, e):
                tok_len = len(decoded[i])
                tok_char_ranges.append((char_offset, char_offset + tok_len, i))
                char_offset += tok_len

            # Find key in line_text
            key_char_start = line_text.find(target_key)
            key_char_end = key_char_start + len(target_key)
            if key_char_start >= 0:
                for cs, ce, ti in tok_char_ranges:
                    if cs < key_char_end and ce > key_char_start:
                        if key_start is None:
                            key_start = ti
                        key_end = ti + 1

            # Find value in line_text
            val_char_start = line_text.find(target_value)
            val_char_end = val_char_start + len(target_value)
            if val_char_start >= 0:
                for cs, ce, ti in tok_char_ranges:
                    if cs < val_char_end and ce > val_char_start:
                        if val_start is None:
                            val_start = ti
                        val_end = ti + 1
            break

    return {
        "needle_start": needle_start,
        "needle_end": needle_end,
        "key_start": key_start,
        "key_end": key_end,
        "val_start": val_start,
        "val_end": val_end,
        "lines": lines,
        "decoded": decoded,
    }


def check_answer(generated_text, target_value):
    """Check if the generated text contains the correct answer."""
    return target_value in generated_text


def run_with_press(model, input_ids, press, max_new_tokens=30):
    """Run generation with a press applied."""
    global _eviction_log
    _eviction_log = []

    with torch.no_grad(), press(model):
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return output, _eviction_log.copy()


def run_full(model, input_ids, max_new_tokens=30):
    """Run generation without compression."""
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return output


def analyze_eviction(eviction_log, needle_info, num_kv_heads=8):
    """Analyze which needle tokens were evicted."""
    if needle_info["needle_start"] is None:
        return None

    ns, ne = needle_info["needle_start"], needle_info["needle_end"]
    ks, ke = needle_info.get("key_start"), needle_info.get("key_end")
    vs, ve = needle_info.get("val_start"), needle_info.get("val_end")

    needle_range = set(range(ns, ne)) if ns is not None and ne is not None else set()
    key_range = set(range(ks, ke)) if ks is not None and ke is not None else set()
    val_range = set(range(vs, ve)) if vs is not None and ve is not None else set()

    results = []
    for entry in eviction_log:
        layer = entry["layer"]
        kept = entry["kept_indices"]  # (H, n_kept)
        scores = entry["scores"]      # (H, k_len)

        # Per-head analysis
        per_head = []
        for h in range(kept.shape[0]):
            kept_set = set(kept[h].tolist())
            needle_kept = needle_range & kept_set
            key_kept = key_range & kept_set
            val_kept = val_range & kept_set

            # Score percentiles for needle tokens
            all_scores = scores[h].float().numpy()
            needle_scores = [all_scores[i] for i in needle_range if i < len(all_scores)]
            key_scores = [all_scores[i] for i in key_range if i < len(all_scores)]
            val_scores = [all_scores[i] for i in val_range if i < len(all_scores)]

            sorted_scores = sorted(all_scores)
            total = len(sorted_scores)

            def percentile(score):
                import bisect
                return bisect.bisect_left(sorted_scores, score) / total * 100

            per_head.append({
                "head": h,
                "needle_kept_frac": len(needle_kept) / max(len(needle_range), 1),
                "key_kept_frac": len(key_kept) / max(len(key_range), 1),
                "val_kept_frac": len(val_kept) / max(len(val_range), 1),
                "avg_needle_percentile": sum(percentile(s) for s in needle_scores) / max(len(needle_scores), 1),
                "avg_key_percentile": sum(percentile(s) for s in key_scores) / max(len(key_scores), 1),
                "avg_val_percentile": sum(percentile(s) for s in val_scores) / max(len(val_scores), 1),
            })

        # Aggregate: any head evicted needle?
        any_head_lost_key = any(ph["key_kept_frac"] < 1.0 for ph in per_head)
        any_head_lost_val = any(ph["val_kept_frac"] < 1.0 for ph in per_head)
        all_heads_lost_val = all(ph["val_kept_frac"] < 1.0 for ph in per_head)

        results.append({
            "layer": layer,
            "any_head_lost_key": any_head_lost_key,
            "any_head_lost_val": any_head_lost_val,
            "all_heads_lost_val": all_heads_lost_val,
            "per_head": per_head,
        })

    return results


def main():
    print("=" * 80)
    print("FAILURE ANALYSIS: CriticalSnapKV / SnapKV on NIAH")
    print("=" * 80)

    # ─── Model loading ───
    model_path = "Qwen/Qwen3-8B"
    print(f"\nLoading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()

    # ─── Test configurations ───
    target_keys = [
        ("brave-falcon", "4829301"),
        ("mystic-thunder", "7156842"),
        ("golden-crystal", "3928174"),
    ]

    configs = []
    for num_dist in [30, 50, 80]:
        for pos_frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
            for key, val in target_keys:
                configs.append({
                    "num_distractors": num_dist,
                    "needle_pos_frac": pos_frac,
                    "target_key": key,
                    "target_value": val,
                })

    compression_ratios = [0.3, 0.5, 0.7]

    print(f"\nTotal samples: {len(configs)}")
    print(f"Compression ratios: {compression_ratios}")
    print(f"Total runs: {len(configs)} × (1 full + {len(compression_ratios)} compressed) = {len(configs) * (1 + len(compression_ratios))}")

    # ─── Run all samples ───
    all_results = []
    failures = []

    for i, cfg in enumerate(configs):
        random.seed(42 + i)  # Different seed per sample for different distractors
        prompt, target_pos = make_sample(
            cfg["num_distractors"], cfg["target_key"], cfg["target_value"], cfg["needle_pos_frac"]
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        seq_len = input_ids.shape[1]

        needle_info = find_needle_tokens(tokenizer, input_ids, cfg["target_key"], cfg["target_value"])

        # Full KV baseline
        full_output = run_full(model, input_ids)
        full_text = tokenizer.decode(full_output[0, seq_len:], skip_special_tokens=True)
        full_correct = check_answer(full_text, cfg["target_value"])

        sample_result = {
            "idx": i,
            "config": cfg,
            "seq_len": seq_len,
            "needle_line_pos": target_pos,
            "needle_token_range": f"[{needle_info['needle_start']}, {needle_info['needle_end']})",
            "full_kv_output": full_text[:100],
            "full_kv_correct": full_correct,
            "compressed_results": {},
        }

        if not full_correct:
            print(f"  [{i:>2}] SKIP: Full KV already wrong (dist={cfg['num_distractors']}, pos={cfg['needle_pos_frac']:.1f}, key={cfg['target_key']})")
            print(f"        Output: {full_text[:80]}")
            sample_result["skip_reason"] = "full_kv_wrong"
            all_results.append(sample_result)
            continue

        # Test with each compression ratio
        for cr in compression_ratios:
            press = SnapKVPress(compression_ratio=cr, window_size=64, kernel_size=5)
            comp_output, eviction_log = run_with_press(model, input_ids, press)
            comp_text = tokenizer.decode(comp_output[0, seq_len:], skip_special_tokens=True)
            comp_correct = check_answer(comp_text, cfg["target_value"])

            eviction_analysis = analyze_eviction(eviction_log, needle_info)

            # Count layers where needle value was fully evicted
            if eviction_analysis:
                layers_val_lost = sum(1 for r in eviction_analysis if r["all_heads_lost_val"])
                layers_key_lost = sum(1 for r in eviction_analysis if r["any_head_lost_key"])
            else:
                layers_val_lost = layers_key_lost = -1

            sample_result["compressed_results"][f"snapkv_{cr}"] = {
                "output": comp_text[:100],
                "correct": comp_correct,
                "layers_val_fully_evicted": layers_val_lost,
                "layers_key_partially_evicted": layers_key_lost,
            }

            if not comp_correct and full_correct:
                failures.append({
                    "sample_idx": i,
                    "config": cfg,
                    "compression_ratio": cr,
                    "full_output": full_text[:100],
                    "comp_output": comp_text[:100],
                    "seq_len": seq_len,
                    "needle_info": {
                        "needle_start": needle_info["needle_start"],
                        "needle_end": needle_info["needle_end"],
                        "key_start": needle_info["key_start"],
                        "key_end": needle_info["key_end"],
                        "val_start": needle_info["val_start"],
                        "val_end": needle_info["val_end"],
                    },
                    "eviction_analysis": eviction_analysis,
                })

        status = "✓" if all(r["correct"] for r in sample_result["compressed_results"].values()) else "✗ FAIL"
        print(f"  [{i:>2}] {status} dist={cfg['num_distractors']:>2}, pos={cfg['needle_pos_frac']:.2f}, key={cfg['target_key']:<18} seq={seq_len:>4} | "
              + " | ".join(f"cr={cr}: {'✓' if sample_result['compressed_results'][f'snapkv_{cr}']['correct'] else '✗'}"
                          for cr in compression_ratios))

        all_results.append(sample_result)

        # Clear CUDA cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()

    # ─── Summary ───
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for cr in compression_ratios:
        key = f"snapkv_{cr}"
        total = sum(1 for r in all_results if key in r["compressed_results"])
        correct = sum(1 for r in all_results if key in r["compressed_results"] and r["compressed_results"][key]["correct"])
        print(f"  SnapKV cr={cr}: {correct}/{total} correct ({correct/max(total,1)*100:.1f}%)")

    skipped = sum(1 for r in all_results if "skip_reason" in r)
    print(f"  Skipped (full KV wrong): {skipped}/{len(all_results)}")

    # ─── Failure analysis ───
    print(f"\n  Total failures: {len(failures)}")

    if failures:
        print("\n" + "=" * 80)
        print("DETAILED FAILURE ANALYSIS")
        print("=" * 80)

        for fi, fail in enumerate(failures):
            cfg = fail["config"]
            ni = fail["needle_info"]
            print(f"\n{'─' * 70}")
            print(f"FAILURE #{fi}: dist={cfg['num_distractors']}, pos={cfg['needle_pos_frac']:.2f}, "
                  f"key={cfg['target_key']}, cr={fail['compression_ratio']}")
            print(f"  seq_len={fail['seq_len']}, needle=[{ni['needle_start']},{ni['needle_end']})")
            print(f"  Full KV:    {fail['full_output']}")
            print(f"  Compressed: {fail['comp_output']}")

            ea = fail["eviction_analysis"]
            if ea is None:
                print("  (no eviction analysis available)")
                continue

            # Which layers lost needle tokens?
            print(f"\n  Per-layer eviction of needle tokens:")
            print(f"  {'Layer':>6} | {'Key kept (per head)':>25} | {'Val kept (per head)':>25} | {'Val percentile':>15}")
            print(f"  {'-'*80}")

            layers_lost_key = []
            layers_lost_val = []

            for layer_result in ea:
                layer = layer_result["layer"]
                ph = layer_result["per_head"]

                key_fracs = [f"{p['key_kept_frac']:.0%}" for p in ph]
                val_fracs = [f"{p['val_kept_frac']:.0%}" for p in ph]
                val_pctls = [p["avg_val_percentile"] for p in ph]
                avg_val_pctl = sum(val_pctls) / len(val_pctls)

                # Only show layers where something interesting happens
                any_key_lost = any(p["key_kept_frac"] < 1.0 for p in ph)
                any_val_lost = any(p["val_kept_frac"] < 1.0 for p in ph)

                if any_key_lost:
                    layers_lost_key.append(layer)
                if any_val_lost:
                    layers_lost_val.append(layer)

                if any_key_lost or any_val_lost:
                    key_str = " ".join(f"{f:>4}" for f in key_fracs)
                    val_str = " ".join(f"{f:>4}" for f in val_fracs)
                    marker = ""
                    if any_key_lost:
                        marker += " ◄KEY LOST"
                    if any_val_lost:
                        marker += " ◄VAL LOST"
                    print(f"  {layer:>6} | {key_str:>25} | {val_str:>25} | {avg_val_pctl:>13.1f}%{marker}")

            print(f"\n  Summary: {len(layers_lost_key)}/{len(ea)} layers lost KEY, {len(layers_lost_val)}/{len(ea)} layers lost VALUE")

            # Retrieval head specific analysis (L7H22 = layer 7, head 22 query / KV head ~2-3 with GQA)
            # With GQA 32→8, query head 22 maps to KV head 22//4 = 5
            retrieval_kv_heads = {
                "L7H22": (7, 5),   # query 22 → kv 5
                "L13H14": (13, 3), # query 14 → kv 3
                "L9H9": (9, 2),    # query 9 → kv 2
            }

            print(f"\n  Retrieval head eviction status:")
            for name, (layer_idx, kv_head) in retrieval_kv_heads.items():
                for layer_result in ea:
                    if layer_result["layer"] == layer_idx:
                        ph = layer_result["per_head"][kv_head]
                        print(f"    {name} (layer {layer_idx}, KV head {kv_head}):")
                        print(f"      Key kept: {ph['key_kept_frac']:.0%}, Val kept: {ph['val_kept_frac']:.0%}")
                        print(f"      Key percentile: {ph['avg_key_percentile']:.1f}%, Val percentile: {ph['avg_val_percentile']:.1f}%")
                        break

    # ─── Aggregate failure patterns ───
    if failures:
        print("\n" + "=" * 80)
        print("FAILURE PATTERN ANALYSIS")
        print("=" * 80)

        # By position
        print("\n  Failures by needle position:")
        pos_counts = defaultdict(lambda: {"total": 0, "fail": 0})
        for r in all_results:
            for cr in compression_ratios:
                key = f"snapkv_{cr}"
                if key in r["compressed_results"]:
                    pos_bin = f"{r['config']['needle_pos_frac']:.2f}"
                    pos_counts[(pos_bin, cr)]["total"] += 1
                    if not r["compressed_results"][key]["correct"]:
                        pos_counts[(pos_bin, cr)]["fail"] += 1

        positions = sorted(set(k[0] for k in pos_counts.keys()))
        print(f"  {'Position':>10}", end="")
        for cr in compression_ratios:
            print(f" | cr={cr:>4}", end="")
        print()
        for pos in positions:
            print(f"  {pos:>10}", end="")
            for cr in compression_ratios:
                d = pos_counts.get((pos, cr), {"total": 0, "fail": 0})
                if d["total"] > 0:
                    rate = d["fail"] / d["total"] * 100
                    print(f" | {d['fail']}/{d['total']} ({rate:>4.0f}%)", end="")
                else:
                    print(f" |   -       ", end="")
            print()

        # By num distractors
        print("\n  Failures by num distractors:")
        dist_counts = defaultdict(lambda: {"total": 0, "fail": 0})
        for r in all_results:
            for cr in compression_ratios:
                key = f"snapkv_{cr}"
                if key in r["compressed_results"]:
                    nd = r["config"]["num_distractors"]
                    dist_counts[(nd, cr)]["total"] += 1
                    if not r["compressed_results"][key]["correct"]:
                        dist_counts[(nd, cr)]["fail"] += 1

        dists = sorted(set(k[0] for k in dist_counts.keys()))
        print(f"  {'Distractors':>12}", end="")
        for cr in compression_ratios:
            print(f" | cr={cr:>4}", end="")
        print()
        for nd in dists:
            print(f"  {nd:>12}", end="")
            for cr in compression_ratios:
                d = dist_counts.get((nd, cr), {"total": 0, "fail": 0})
                if d["total"] > 0:
                    rate = d["fail"] / d["total"] * 100
                    print(f" | {d['fail']}/{d['total']} ({rate:>4.0f}%)", end="")
                else:
                    print(f" |   -       ", end="")
            print()

        # Common eviction pattern in failures
        print("\n  Eviction patterns in failures:")
        val_lost_counts = []
        key_lost_counts = []
        for fail in failures:
            if fail["eviction_analysis"]:
                vl = sum(1 for r in fail["eviction_analysis"] if r["any_head_lost_val"])
                kl = sum(1 for r in fail["eviction_analysis"] if r["any_head_lost_key"])
                total_layers = len(fail["eviction_analysis"])
                val_lost_counts.append(vl)
                key_lost_counts.append(kl)

        if val_lost_counts:
            print(f"    Layers losing VALUE (any head): avg {sum(val_lost_counts)/len(val_lost_counts):.1f} / {total_layers}")
            print(f"    Layers losing KEY (any head): avg {sum(key_lost_counts)/len(key_lost_counts):.1f} / {total_layers}")

    # ─── Pick ONE representative failure for deep attention analysis ───
    if failures:
        print("\n" + "=" * 80)
        print("DEEP ATTENTION ANALYSIS ON REPRESENTATIVE FAILURE")
        print("=" * 80)

        # Pick the first failure at cr=0.5
        rep_fail = None
        for f in failures:
            if f["compression_ratio"] == 0.5:
                rep_fail = f
                break
        if rep_fail is None:
            rep_fail = failures[0]

        cfg = rep_fail["config"]
        ni = rep_fail["needle_info"]
        print(f"\n  Config: dist={cfg['num_distractors']}, pos={cfg['needle_pos_frac']}, key={cfg['target_key']}")
        print(f"  Full KV:    {rep_fail['full_output']}")
        print(f"  Compressed: {rep_fail['comp_output']}")

        # Regenerate the sample and run attention analysis
        random.seed(42 + rep_fail["sample_idx"])
        prompt, _ = make_sample(cfg["num_distractors"], cfg["target_key"], cfg["target_value"], cfg["needle_pos_frac"])
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        seq_len = input_ids.shape[1]

        print(f"\n  Running forward pass with output_attentions=True...")
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True, return_dict=True)

        attentions = outputs.attentions
        num_layers = len(attentions)

        ns, ne = ni["needle_start"], ni["needle_end"]
        vs, ve = ni["val_start"], ni["val_end"]
        ks, ke = ni["key_start"], ni["key_end"]

        # Retrieval heads: attention to needle from last token
        print(f"\n  Retrieval head attention to needle (from last token):")
        retrieval_heads = [(7, 22), (13, 14), (9, 9), (15, 13)]
        for l, h in retrieval_heads:
            attn = attentions[l][0, h, -1, :].float()
            a_needle = attn[ns:ne].sum().item() * 100
            a_key = attn[ks:ke].sum().item() * 100 if ks and ke else 0
            a_val = attn[vs:ve].sum().item() * 100 if vs and ve else 0
            print(f"    L{l}H{h}: needle={a_needle:.2f}%, key={a_key:.2f}%, value={a_val:.2f}%")

        # Now show eviction: retrieval head KV heads
        retrieval_kv_heads_deep = {
            "L7H22": (7, 5),   # query 22 → kv 5
            "L13H14": (13, 3), # query 14 → kv 3
            "L9H9": (9, 2),    # query 9 → kv 2
        }
        print(f"\n  Retrieval head KV eviction status:")
        ea = rep_fail["eviction_analysis"]
        if ea:
            for name, (layer_idx, kv_head) in retrieval_kv_heads_deep.items():
                for layer_result in ea:
                    if layer_result["layer"] == layer_idx:
                        ph = layer_result["per_head"][kv_head]
                        print(f"    {name} (layer {layer_idx}, KV head {kv_head}):")
                        print(f"      Key kept: {ph['key_kept_frac']:.0%}, Val kept: {ph['val_kept_frac']:.0%}")
                        print(f"      Key percentile: {ph['avg_key_percentile']:.1f}%, Val percentile: {ph['avg_val_percentile']:.1f}%")
                        break

            # Show per-layer needle value scores vs threshold
            print(f"\n  Needle VALUE token scores vs eviction threshold:")
            print(f"  {'Layer':>6} | {'Avg val score':>14} | {'Threshold (kept)':>17} | {'Val kept?':>10}")
            print(f"  {'-'*55}")
            for entry in rep_fail["eviction_analysis"][:36]:
                layer = entry["layer"]
                ph = entry["per_head"]
                # Average across heads
                avg_val_pctl = sum(p["avg_val_percentile"] for p in ph) / len(ph)
                avg_val_kept = sum(p["val_kept_frac"] for p in ph) / len(ph)
                marker = "" if avg_val_kept > 0.5 else " ◄ EVICTED"
                print(f"  {layer:>6} | {avg_val_pctl:>13.1f}% | {(1-rep_fail['compression_ratio'])*100:>16.0f}% | {avg_val_kept:>9.0%}{marker}")

        del attentions, outputs
        torch.cuda.empty_cache()

    print("\n\nDONE")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
