#!/usr/bin/env python3
"""Compare attention patterns between baseline and v9 models."""

import torch
import random
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(42)


def make_multikey2_input(tokenizer, num_distractors=200,
                         target_key="brave-falcon", target_value="4829301"):
    adjectives = ["silent", "golden", "dusty", "hollow", "bitter", "crimson",
                  "gentle", "fierce", "mystic", "frozen", "swift", "ancient",
                  "bright", "dark", "wild", "calm", "bold", "shy", "proud",
                  "humble", "vast", "tiny", "deep", "high"]
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


def find_token_positions(tokenizer, input_ids, search_str):
    """Find token position range for a string."""
    full_text = tokenizer.decode(input_ids[0])
    char_start = full_text.find(search_str)
    if char_start < 0:
        return None, None

    cumlen = 0
    tok_start = None
    tok_end = None
    for i, tid in enumerate(input_ids[0]):
        piece = tokenizer.decode([tid])
        if cumlen <= char_start < cumlen + len(piece):
            tok_start = i
        if cumlen < char_start + len(search_str) <= cumlen + len(piece):
            tok_end = i + 1
            break
        cumlen += len(piece)

    return tok_start, tok_end


def classify_tokens(tokenizer, input_ids, target_key, target_value):
    """Classify each token into categories."""
    tokens = input_ids[0].tolist()
    decoded = [tokenizer.decode([t]) for t in tokens]
    seq_len = len(tokens)

    target_str = f"One of the special magic numbers for {target_key} is: {target_value}."
    t_start, t_end = find_token_positions(tokenizer, input_ids, target_str)

    q_str = f"What is the special magic number for {target_key}"
    q_start, q_end = find_token_positions(tokenizer, input_ids, q_str)

    categories = []
    for i in range(seq_len):
        tok = decoded[i]
        if t_start and t_end and t_start <= i < t_end:
            categories.append("target_needle")
        elif q_start and q_end and q_start <= i:
            categories.append("question")
        elif any(c in tok for c in [".", ",", ":", "\n", "!", "?"]):
            categories.append("punctuation")
        elif tok.strip().isdigit() or (len(tok.strip()) > 3 and any(c.isdigit() for c in tok)):
            categories.append("distractor_value")
        elif i < 20:
            categories.append("instruction")
        else:
            categories.append("distractor_text")

    return categories, decoded, t_start, t_end, q_start, q_end


def analyze_attention(model, tokenizer, input_ids, model_name):
    """Get per-layer attention statistics."""
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_attentions=True,
            return_dict=True,
        )

    all_attentions = outputs.attentions
    num_layers = len(all_attentions)
    seq_len = input_ids.shape[1]

    cumulative_attn = torch.zeros(num_layers, seq_len, device=input_ids.device)
    last_token_attn = torch.zeros(num_layers, seq_len, device=input_ids.device)
    last64_attn = torch.zeros(num_layers, seq_len, device=input_ids.device)

    for layer_idx, attn in enumerate(all_attentions):
        # attn: (1, num_heads, L, L)
        attn_sum = attn[0].sum(dim=0).sum(dim=0)  # (L,)
        cumulative_attn[layer_idx] = attn_sum

        last_attn = attn[0, :, -1, :].mean(dim=0)  # (L,)
        last_token_attn[layer_idx] = last_attn

        window = min(64, seq_len)
        last64 = attn[0, :, -window:, :].mean(dim=0).mean(dim=0)  # (L,)
        last64_attn[layer_idx] = last64

    return {
        "cumulative": cumulative_attn.cpu(),
        "last_token": last_token_attn.cpu(),
        "last64": last64_attn.cpu(),
        "num_layers": num_layers,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9_model", default="/home/zichuanfu2/SparseKV/output/qwen3_sparsekv_v9/merged")
    parser.add_argument("--baseline_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--num_distractors", type=int, default=100)
    parser.add_argument("--output_dir", default="./analysis/attention_diff")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.v9_model)

    prompt, target_pos = make_multikey2_input(tokenizer, args.num_distractors)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    print(f"Sequence length: {seq_len}")
    print(f"Target needle at line ~{target_pos}")

    categories, decoded, t_start, t_end, q_start, q_end = classify_tokens(
        tokenizer, input_ids, "brave-falcon", "4829301"
    )
    print(f"Target needle tokens: [{t_start}, {t_end})")
    print(f"Question tokens: [{q_start}, {q_end})")

    cat_counts = {}
    for c in categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print(f"Token categories: {cat_counts}")

    results = {}

    for name, path in [("baseline", args.baseline_model), ("v9", args.v9_model)]:
        print(f"\nLoading {name}: {path}")
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
        model.eval()
        ids = input_ids.to(model.device)
        attn_data = analyze_attention(model, tokenizer, ids, name)
        results[name] = attn_data
        del model
        torch.cuda.empty_cache()

    # Compare attention patterns
    print("\n" + "=" * 80)
    print("ATTENTION COMPARISON: Baseline vs v9")
    print("=" * 80)

    cat_names = ["target_needle", "question", "punctuation", "distractor_value",
                 "distractor_text", "instruction"]

    for attn_type in ["cumulative", "last_token", "last64"]:
        print(f"\n--- {attn_type} attention ---")

        for name in ["baseline", "v9"]:
            attn = results[name][attn_type]
            avg_attn = attn.mean(dim=0).float()

            print(f"\n  {name}:")
            for cat in cat_names:
                indices = [i for i, c in enumerate(categories) if c == cat]
                if not indices:
                    continue
                cat_scores = avg_attn[indices]
                print(f"    {cat:20s}: mean={cat_scores.mean():.6f}, "
                      f"max={cat_scores.max():.6f}, count={len(indices)}")

        print(f"\n  DIFFERENCE (v9 - baseline):")
        baseline_attn = results["baseline"][attn_type].mean(dim=0).float()
        v9_attn = results["v9"][attn_type].mean(dim=0).float()
        diff = v9_attn - baseline_attn

        for cat in cat_names:
            indices = [i for i, c in enumerate(categories) if c == cat]
            if not indices:
                continue
            cat_diff = diff[indices]
            bl_mean = baseline_attn[indices].mean()
            print(f"    {cat:20s}: mean_diff={cat_diff.mean():+.6f}, "
                  f"relative={cat_diff.mean() / (bl_mean + 1e-10):+.1%}")

    # Per-layer analysis for target_needle
    print("\n" + "=" * 80)
    print("PER-LAYER: Target needle attention (cumulative)")
    print("=" * 80)
    target_indices = [i for i, c in enumerate(categories) if c == "target_needle"]
    distractor_indices = [i for i, c in enumerate(categories) if c == "distractor_value"]

    print(f"{'Layer':>6} | {'BL needle':>12} {'v9 needle':>12} {'diff':>10} | "
          f"{'BL distract':>12} {'v9 distract':>12} {'diff':>10} | "
          f"{'needle/dist BL':>15} {'needle/dist v9':>15}")
    print("-" * 120)

    num_layers = results["baseline"]["num_layers"]
    for layer in range(num_layers):
        bl_needle = results["baseline"]["cumulative"][layer][target_indices].float().mean()
        v9_needle = results["v9"]["cumulative"][layer][target_indices].float().mean()
        bl_dist = results["baseline"]["cumulative"][layer][distractor_indices].float().mean()
        v9_dist = results["v9"]["cumulative"][layer][distractor_indices].float().mean()

        ratio_bl = bl_needle / (bl_dist + 1e-10)
        ratio_v9 = v9_needle / (v9_dist + 1e-10)

        print(f"{layer:>6} | {bl_needle:>12.4f} {v9_needle:>12.4f} {v9_needle - bl_needle:>+10.4f} | "
              f"{bl_dist:>12.4f} {v9_dist:>12.4f} {v9_dist - bl_dist:>+10.4f} | "
              f"{ratio_bl:>15.3f} {ratio_v9:>15.3f}")

    # Attention entropy comparison
    print("\n" + "=" * 80)
    print("ATTENTION ENTROPY (per layer, last token attention)")
    print("=" * 80)

    print(f"{'Layer':>6} | {'BL entropy':>12} {'v9 entropy':>12} {'diff':>10} {'change':>10}")
    print("-" * 60)
    for layer in range(num_layers):
        bl_attn = results["baseline"]["last_token"][layer].float()
        v9_attn = results["v9"]["last_token"][layer].float()

        bl_attn_safe = bl_attn.clamp(min=1e-10)
        v9_attn_safe = v9_attn.clamp(min=1e-10)
        bl_entropy = -(bl_attn_safe * bl_attn_safe.log()).sum()
        v9_entropy = -(v9_attn_safe * v9_attn_safe.log()).sum()

        print(f"{layer:>6} | {bl_entropy:>12.2f} {v9_entropy:>12.2f} "
              f"{v9_entropy - bl_entropy:>+10.2f} {(v9_entropy - bl_entropy) / (bl_entropy + 1e-10):>+10.1%}")

    # Top attended tokens analysis
    print("\n" + "=" * 80)
    print("TOP-50 most attended tokens (cumulative, averaged across layers)")
    print("=" * 80)

    for name in ["baseline", "v9"]:
        avg_attn = results[name]["cumulative"].mean(dim=0).float()
        top_indices = avg_attn.topk(50).indices.tolist()
        print(f"\n  {name} top-50:")
        for rank, idx in enumerate(top_indices[:50]):
            cat = categories[idx]
            tok = decoded[idx].replace("\n", "\\n")
            print(f"    {rank+1:3d}. pos={idx:4d} cat={cat:20s} tok='{tok}' score={avg_attn[idx]:.4f}")

    # Save raw data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        "categories": categories,
        "seq_len": seq_len,
        "target_range": [t_start, t_end],
        "question_range": [q_start, q_end],
    }

    for name in ["baseline", "v9"]:
        for attn_type in ["cumulative", "last_token", "last64"]:
            key = f"{name}_{attn_type}_mean"
            save_data[key] = results[name][attn_type].mean(dim=0).float().tolist()

    with open(output_dir / "attention_diff.json", "w") as f:
        json.dump(save_data, f)
    print(f"\nRaw data saved to {output_dir / 'attention_diff.json'}")


if __name__ == "__main__":
    main()
