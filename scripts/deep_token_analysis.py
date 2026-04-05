#!/usr/bin/env python3
"""Deep single-sample analysis: which tokens matter, which does SnapKV evict?

Answers:
  Q1. Are needle tokens scored higher than distractors by SnapKV?
  Q2. At 50% compression, is the needle evicted?
  Q3. If we forcibly mask the needle, does the model fail? (ground truth importance)
  Q4. If we mask distractors instead, does the model still succeed?
  Q5. How does v9 compare to baseline?
"""

import torch
import random
import sys
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress import SnapKVPress
from kvpress.presses.scorer_press import ScorerPress

random.seed(42)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Create NIAH sample
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Token classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def classify_tokens(tokenizer, input_ids, target_key, target_value):
    full_text = tokenizer.decode(input_ids[0])
    tokens = input_ids[0].tolist()
    decoded = [tokenizer.decode([t]) for t in tokens]
    seq_len = len(tokens)

    # Build char offset table
    char_offsets = []
    cum = 0
    for t in tokens:
        piece = tokenizer.decode([t])
        char_offsets.append((cum, cum + len(piece)))
        cum += len(piece)

    def char_to_token_range(char_start, char_end):
        tok_start = tok_end = None
        for i, (cs, ce) in enumerate(char_offsets):
            if tok_start is None and ce > char_start:
                tok_start = i
            if cs < char_end:
                tok_end = i + 1
        return tok_start, tok_end

    # Find target needle
    target_str = f"One of the special magic numbers for {target_key} is: {target_value}."
    ts_char = full_text.find(target_str)
    t_start, t_end = char_to_token_range(ts_char, ts_char + len(target_str)) if ts_char >= 0 else (None, None)

    # Find needle key and value sub-ranges within needle
    if ts_char >= 0:
        nk_char = full_text.find(target_key, ts_char)
        nv_char = full_text.find(target_value, ts_char)
        nk_start, nk_end = char_to_token_range(nk_char, nk_char + len(target_key)) if nk_char >= 0 else (None, None)
        nv_start, nv_end = char_to_token_range(nv_char, nv_char + len(target_value)) if nv_char >= 0 else (None, None)
    else:
        nk_start = nk_end = nv_start = nv_end = None

    # Find question
    q_str = f"What is the special magic number for {target_key}"
    qs_char = full_text.find(q_str)
    q_start, q_end = (char_to_token_range(qs_char, qs_char + len(q_str))[0], seq_len) if qs_char >= 0 else (None, None)

    # Classify
    categories = []
    for i in range(seq_len):
        if nk_start is not None and nk_start <= i < nk_end:
            categories.append("needle_key")
        elif nv_start is not None and nv_start <= i < nv_end:
            categories.append("needle_value")
        elif t_start is not None and t_start <= i < t_end:
            categories.append("needle_ctx")  # needle surrounding text
        elif q_start is not None and q_start <= i:
            categories.append("question")
        elif any(c in decoded[i] for c in [".", ",", ":", "\n", "!", "?"]):
            categories.append("punctuation")
        elif (decoded[i].strip().isdigit() or
              (len(decoded[i].strip()) > 2 and sum(c.isdigit() for c in decoded[i]) > len(decoded[i].strip()) // 2)):
            categories.append("distractor_val")
        elif i < 30:
            categories.append("instruction")
        else:
            categories.append("distractor_txt")

    return {
        "categories": categories,
        "decoded": decoded,
        "needle_range": (t_start, t_end),
        "needle_key_range": (nk_start, nk_end),
        "needle_value_range": (nv_start, nv_end),
        "question_range": (q_start, q_end),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. SnapKV scoring capture via monkey-patch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_snapkv_log = []
_orig_compress = ScorerPress.compress

def _capture_compress(self, module, hidden_states, keys, values, attentions, kwargs):
    scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
    k_len = keys.shape[2]
    n_kept = int(k_len * (1 - self.compression_ratio))
    kept = scores.topk(n_kept, dim=-1).indices  # (B, H, n_kept)

    _snapkv_log.append({
        "layer": module.layer_idx,
        "k_len": k_len,
        "n_kept": n_kept,
        "scores": scores[0].cpu().float(),      # (H, L)
        "kept": kept[0].cpu(),                   # (H, n_kept)
    })
    return _orig_compress(self, module, hidden_states, keys, values, attentions, kwargs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Run forward with specific tokens masked
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def forward_masked(model, input_ids, masked_positions):
    """Forward with certain KV positions masked out."""
    seq_len = input_ids.shape[1]
    attn_mask = torch.ones(1, seq_len, device=input_ids.device, dtype=torch.long)
    for p in masked_positions:
        if 0 <= p < seq_len:
            attn_mask[0, p] = 0
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask).logits[0, -1]
    return logits


def generate_answer(model, tokenizer, input_ids, masked_positions=None, max_new=15):
    """Generate tokens, optionally masking certain positions."""
    kwargs = dict(max_new_tokens=max_new, do_sample=False)
    if masked_positions:
        seq_len = input_ids.shape[1]
        attn_mask = torch.ones(1, seq_len, device=input_ids.device, dtype=torch.long)
        for p in masked_positions:
            if 0 <= p < seq_len:
                attn_mask[0, p] = 0
        kwargs["attention_mask"] = attn_mask
    with torch.no_grad():
        out = model.generate(input_ids, **kwargs)
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Main analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze_model(model, tokenizer, input_ids, tinfo, model_name, cr=0.5):
    ids = input_ids.to(model.device)
    seq_len = ids.shape[1]
    cats = tinfo["categories"]
    decoded = tinfo["decoded"]

    cat_pos = defaultdict(list)
    for i, c in enumerate(cats):
        cat_pos[c].append(i)

    needle_set = set(range(*tinfo["needle_range"])) if tinfo["needle_range"][0] else set()
    key_set = set(range(*tinfo["needle_key_range"])) if tinfo["needle_key_range"][0] else set()
    val_set = set(range(*tinfo["needle_value_range"])) if tinfo["needle_value_range"][0] else set()

    print(f"\n{'#'*80}")
    print(f"# MODEL: {model_name}")
    print(f"{'#'*80}")

    # ── A. Full KV (baseline answer) ──────────────────────────
    print("\n[A] FULL KV CACHE (ground truth)")
    answer_full = generate_answer(model, tokenizer, ids)
    with torch.no_grad():
        full_logits = model(ids).logits[0, -1].float()
    top10 = torch.topk(full_logits, 10)
    print(f"  Answer: '{answer_full}'")
    print(f"  Top-10: {[(tokenizer.decode([t]), f'{v:.1f}') for t, v in zip(top10.indices.tolist(), top10.values.tolist())]}")
    correct = "4829301" in answer_full

    # ── B. SnapKV eviction ────────────────────────────────────
    print(f"\n[B] SNAPKV EVICTION (compression_ratio={cr}, keep {(1-cr)*100:.0f}% tokens)")
    ScorerPress.compress = _capture_compress
    _snapkv_log.clear()

    press = SnapKVPress(compression_ratio=cr, window_size=64, kernel_size=5)
    with torch.no_grad():
        from transformers import DynamicCache
        cache = DynamicCache()
        with press(model):
            snap_out = model(ids, past_key_values=cache)
    ScorerPress.compress = _orig_compress

    snap_logits = snap_out.logits[0, -1].float()
    snap_top10 = torch.topk(snap_logits, 10)
    print(f"  Top-10: {[(tokenizer.decode([t]), f'{v:.1f}') for t, v in zip(snap_top10.indices.tolist(), snap_top10.values.tolist())]}")

    num_layers = len(_snapkv_log)
    n_kept = _snapkv_log[0]["n_kept"]
    k_len = _snapkv_log[0]["k_len"]
    print(f"  {num_layers} layers, keeping {n_kept}/{k_len} tokens per layer per head")

    # ── B1. Per-category SnapKV scores ──
    print(f"\n  [B1] SnapKV importance scores by category (mean over layers & heads):")
    all_scores = torch.stack([e["scores"].mean(dim=0) for e in _snapkv_log])  # (L, L)
    avg_scores = all_scores.mean(dim=0)  # (L,)

    ordered_cats = ["needle_key", "needle_value", "needle_ctx", "question",
                    "instruction", "punctuation", "distractor_val", "distractor_txt"]
    for cat in ordered_cats:
        pp = cat_pos.get(cat, [])
        if not pp:
            continue
        sc = avg_scores[pp]
        print(f"    {cat:18s}: mean={sc.mean():.6f}  max={sc.max():.6f}  "
              f"min={sc.min():.6f}  n={len(pp)}")

    # ── B2. Per-layer needle retention (union over heads) ──
    print(f"\n  [B2] Needle token retention per layer (union over heads):")
    print(f"  {'Lay':>4} | {'key kept':>8} {'val kept':>8} {'all kept':>8} | "
          f"{'needle_score':>13} {'distract_score':>14} {'ratio':>7}")
    print(f"  {'-'*75}")

    dist_val_pos = cat_pos.get("distractor_val", [])

    for entry in _snapkv_log:
        layer = entry["layer"]
        # Union of kept tokens across all heads
        kept_union = set()
        for h in range(entry["kept"].shape[0]):
            kept_union.update(entry["kept"][h].tolist())

        kk = "YES" if key_set.issubset(kept_union) else f"NO ({len(key_set & kept_union)}/{len(key_set)})"
        vk = "YES" if val_set.issubset(kept_union) else f"NO ({len(val_set & kept_union)}/{len(val_set)})"
        ak = "YES" if needle_set.issubset(kept_union) else f"NO ({len(needle_set & kept_union)}/{len(needle_set)})"

        # Scores averaged over heads
        s_h_mean = entry["scores"].mean(dim=0)  # (L,)
        ns = s_h_mean[sorted(needle_set)].mean().item() if needle_set else 0
        ds = s_h_mean[dist_val_pos].mean().item() if dist_val_pos else 0
        ratio = ns / (ds + 1e-10)

        print(f"  {layer:>4} | {kk:>8} {vk:>8} {ak:>8} | "
              f"{ns:>13.6f} {ds:>14.6f} {ratio:>7.2f}")

    # ── B3. Per-head analysis for a few interesting layers ──
    print(f"\n  [B3] Per-HEAD needle retention (selected layers):")
    sample_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    for entry in _snapkv_log:
        if entry["layer"] not in sample_layers:
            continue
        layer = entry["layer"]
        n_heads = entry["kept"].shape[0]
        head_status = []
        for h in range(n_heads):
            kept_h = set(entry["kept"][h].tolist())
            key_ok = key_set.issubset(kept_h)
            val_ok = val_set.issubset(kept_h)
            head_status.append("KV" if (key_ok and val_ok) else ("K" if key_ok else ("V" if val_ok else ".")))
        print(f"    Layer {layer:2d}: [{' '.join(head_status)}]  "
              f"(K=key kept, V=val kept, KV=both, .=neither)")

    # ── B4. Score distribution around the needle ──
    print(f"\n  [B4] Scores around the needle (avg over layers & heads):")
    if tinfo["needle_range"][0]:
        ns, ne = tinfo["needle_range"]
        context_start = max(0, ns - 5)
        context_end = min(seq_len, ne + 5)
        print(f"    {'pos':>5} {'token':>20} {'category':>16} {'score':>10} {'rank':>6}")
        print(f"    {'-'*63}")
        sorted_all = avg_scores.argsort(descending=True).tolist()
        for i in range(context_start, context_end):
            tok = decoded[i].replace("\n", "\\n")[:18]
            rank = sorted_all.index(i) + 1
            marker = " <<<" if i in key_set or i in val_set else ""
            print(f"    {i:>5} {tok:>20} {cats[i]:>16} {avg_scores[i]:>10.6f} {rank:>6}{marker}")

    # ── C. Targeted ablations ─────────────────────────────────
    print(f"\n[C] TARGETED ABLATIONS (ground truth token importance)")

    ablations = [
        ("full_kv", []),
        ("mask_needle_key", sorted(key_set)),
        ("mask_needle_value", sorted(val_set)),
        ("mask_whole_needle", sorted(needle_set)),
        ("mask_distractor_vals", cat_pos.get("distractor_val", [])),
        ("mask_all_punctuation", cat_pos.get("punctuation", [])),
        ("mask_random_50pct", sorted(random.sample(range(seq_len), seq_len // 2))),
    ]

    print(f"  {'Condition':>25} | {'#masked':>7} | {'Answer':>30} | {'Correct?':>8} | Top-3")
    print(f"  {'-'*110}")

    for abl_name, masked in ablations:
        if abl_name == "full_kv":
            ans = answer_full
            logits = full_logits
        else:
            ans = generate_answer(model, tokenizer, ids, masked_positions=masked)
            logits = forward_masked(model, ids, masked)

        top3 = torch.topk(logits, 3)
        top3_str = ", ".join(f"'{tokenizer.decode([t])}'" for t in top3.indices.tolist())
        ok = "4829301" in ans
        print(f"  {abl_name:>25} | {len(masked):>7} | {ans[:28]:>30} | {'YES' if ok else 'NO':>8} | {top3_str}")

    # ── D. Needle ranking among all tokens ────────────────────
    print(f"\n[D] WHERE DOES THE NEEDLE RANK?")
    print(f"  (SnapKV score ranking out of {seq_len} tokens)")

    # Average across layers, show per-head
    avg_per_head = torch.stack([e["scores"] for e in _snapkv_log]).mean(dim=0)  # (H, L)
    n_heads = avg_per_head.shape[0]

    if tinfo["needle_key_range"][0]:
        key_tokens = list(range(*tinfo["needle_key_range"]))
        val_tokens = list(range(*tinfo["needle_value_range"])) if tinfo["needle_value_range"][0] else []

        print(f"\n  Needle KEY tokens ({len(key_tokens)} tokens, positions {key_tokens}):")
        for h in range(min(8, n_heads)):
            ranks = []
            for p in key_tokens:
                r = (avg_per_head[h] > avg_per_head[h, p]).sum().item() + 1
                ranks.append(r)
            avg_rank = sum(ranks) / len(ranks)
            pct = avg_rank / seq_len * 100
            print(f"    Head {h}: avg rank = {avg_rank:.0f}/{seq_len} (top {pct:.1f}%)")

        if val_tokens:
            print(f"\n  Needle VALUE tokens ({len(val_tokens)} tokens, positions {val_tokens}):")
            for h in range(min(8, n_heads)):
                ranks = []
                for p in val_tokens:
                    r = (avg_per_head[h] > avg_per_head[h, p]).sum().item() + 1
                    ranks.append(r)
                avg_rank = sum(ranks) / len(ranks)
                pct = avg_rank / seq_len * 100
                print(f"    Head {h}: avg rank = {avg_rank:.0f}/{seq_len} (top {pct:.1f}%)")

    # ── E. What tokens get highest SnapKV scores? ─────────────
    print(f"\n[E] TOP-30 HIGHEST-SCORED TOKENS (SnapKV, avg over layers & heads)")
    top30_idx = avg_scores.topk(30).indices.tolist()
    print(f"  {'Rank':>4} {'Pos':>5} {'Token':>20} {'Category':>16} {'Score':>10}")
    print(f"  {'-'*60}")
    for rank, idx in enumerate(top30_idx):
        tok = decoded[idx].replace("\n", "\\n")[:18]
        marker = " <<<" if idx in needle_set else ""
        print(f"  {rank+1:>4} {idx:>5} {tok:>20} {cats[idx]:>16} {avg_scores[idx]:>10.6f}{marker}")

    # ── F. Score percentile summary ───────────────────────────
    print(f"\n[F] SCORE PERCENTILE SUMMARY (where each category falls in the global ranking)")
    sorted_scores = avg_scores.sort(descending=True).values
    for cat in ordered_cats:
        pp = cat_pos.get(cat, [])
        if not pp:
            continue
        cat_scores = avg_scores[pp]
        # What percentile is the median score of this category?
        median_score = cat_scores.median().item()
        percentile = (sorted_scores > median_score).sum().item() / seq_len * 100
        print(f"    {cat:18s}: median at top {percentile:.1f}% (n={len(pp)})")

    return answer_full


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9_model", default="/home/zichuanfu2/SparseKV/output/qwen3_sparsekv_v9/merged")
    parser.add_argument("--baseline_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--num_distractors", type=int, default=50)
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--only", choices=["baseline", "v9"], default=None,
                        help="Only analyze one model")
    args = parser.parse_args()

    print("=" * 80)
    print("DEEP TOKEN IMPORTANCE ANALYSIS")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.baseline_model)

    prompt, target_pos = make_sample(args.num_distractors)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    print(f"Sequence length: {seq_len} tokens")
    print(f"Target needle at line ~{target_pos}")
    print(f"Compression ratio: {args.compression_ratio} (keep {(1-args.compression_ratio)*100:.0f}%)")

    tinfo = classify_tokens(tokenizer, input_ids, "brave-falcon", "4829301")

    # Print classification summary
    cat_counts = defaultdict(int)
    for c in tinfo["categories"]:
        cat_counts[c] += 1
    print(f"\nToken categories:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:18s}: {cnt:>4} tokens ({cnt/seq_len*100:.1f}%)")

    # Print needle tokens in detail
    nr = tinfo["needle_range"]
    if nr[0]:
        print(f"\nNeedle line tokens [{nr[0]}, {nr[1]}):")
        for i in range(nr[0], nr[1]):
            tok = tinfo["decoded"][i].replace("\n", "\\n")
            print(f"  {i:>4}: [{tinfo['categories'][i]:>13}] '{tok}'")

    # Analyze models
    models_to_run = []
    if args.only != "v9":
        models_to_run.append(("baseline", args.baseline_model))
    if args.only != "baseline":
        models_to_run.append(("v9", args.v9_model))

    for name, path in models_to_run:
        print(f"\n\nLoading {name}: {path}")
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        )
        model.eval()

        analyze_model(model, tokenizer, input_ids, tinfo, name, args.compression_ratio)

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
