#!/usr/bin/env python3
"""Analyze which tokens CriticalSnapKV evicts on multikey_2-style inputs."""

import torch
import random
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress import CriticalKVPress, SnapKVPress
from kvpress.presses.scorer_press import ScorerPress

# ── Monkey-patch to capture eviction details ──
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
        "kept_indices": kept_indices[0, 0].sort().values.cpu().tolist(),  # head 0, batch 0
        "scores": scores[0, 0].cpu().tolist(),  # head 0, batch 0
    })

    return _orig_compress(self, module, hidden_states, keys, values, attentions, kwargs)

ScorerPress.compress = _debug_compress


def make_multikey2_input(num_distractors=200, target_key="brave-falcon", target_value="4829301"):
    """Create a multikey_2-style input."""
    adjectives = ["silent", "golden", "dusty", "hollow", "bitter", "crimson", "gentle", "fierce",
                  "mystic", "frozen", "swift", "ancient", "bright", "dark", "wild", "calm",
                  "bold", "shy", "proud", "humble", "vast", "tiny", "deep", "high"]
    nouns = ["rabbit", "hawk", "maple", "pine", "stream", "tide", "wolf", "eagle",
             "river", "stone", "flame", "shadow", "storm", "leaf", "crystal", "thunder",
             "ocean", "moon", "star", "wind", "cloud", "dawn", "frost", "ember"]

    # Generate distractor needles
    needles = []
    for _ in range(num_distractors):
        key = f"{random.choice(adjectives)}-{random.choice(nouns)}"
        value = str(random.randint(1000000, 9999999))
        needles.append(f"One of the special magic numbers for {key} is: {value}.")

    # Insert target needle at random position
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
    return prompt, target_needle, target_pos


def find_needle_token_range(tokenizer, input_ids, target_key, target_value):
    """Find the token positions of the target key and value by decoding token-by-token."""
    tokens = input_ids[0].tolist()
    decoded = [tokenizer.decode([t]) for t in tokens]

    # Search for target_key in the decoded tokens
    key_start = None
    key_text = ""
    for i, d in enumerate(decoded):
        # Look for the start of the key
        if target_key.startswith(d.strip()) or (key_text and target_key.startswith(key_text + d)):
            if key_start is None:
                key_start = i
            key_text += d
            if target_key in key_text:
                key_end = i + 1
                # Now find the value tokens nearby (within 20 tokens after key)
                val_start = None
                val_text = ""
                for j in range(key_end, min(key_end + 20, len(decoded))):
                    chunk = decoded[j].strip()
                    if chunk and any(c.isdigit() for c in chunk):
                        if val_start is None:
                            val_start = j
                        val_text += decoded[j]
                        if target_value in val_text:
                            val_end = j + 1
                            # Expand to full needle line: go back to "One" and forward to "."
                            line_start = key_start
                            for k in range(key_start, max(key_start - 30, 0), -1):
                                if "One" in decoded[k] or "magic" in decoded[k]:
                                    line_start = k
                                    break
                            line_end = val_end
                            for k in range(val_end, min(val_end + 5, len(decoded))):
                                if "." in decoded[k]:
                                    line_end = k + 1
                                    break
                            return line_start, line_end, key_start, key_end, val_start, val_end
                break
        else:
            key_start = None
            key_text = ""

    # Simpler fallback: search decoded text for key string
    full_decoded = tokenizer.decode(input_ids[0])
    char_pos = full_decoded.find(target_key)
    if char_pos >= 0:
        # Map char position to token position
        char_count = 0
        for i, t in enumerate(tokens):
            char_count += len(tokenizer.decode([t]))
            if char_count >= char_pos:
                return max(0, i-15), i+20, i, i+3, i+5, i+8

    return None, None, None, None, None, None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/zichuanfu2/SparseKV/output/qwen3_sparsekv_v9/merged")
    parser.add_argument("--baseline", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--num_distractors", type=int, default=200)
    parser.add_argument("--num_examples", type=int, default=3)
    args = parser.parse_args()

    random.seed(42)

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Analyze both v9 and baseline
    for model_name, model_path in [("v9", args.model), ("baseline", args.baseline)]:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name} ({model_path})")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        random.seed(42)  # Reset seed for consistent examples

        for ex_idx in range(args.num_examples):
            prompt, target_needle, target_pos = make_multikey2_input(args.num_distractors)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            print(f"\n--- Example {ex_idx+1} ---")
            print(f"Sequence length: {seq_len} tokens")
            print(f"Target needle at line ~{target_pos}: {target_needle}")

            # Find needle token positions
            target_key = "brave-falcon"
            target_value = "4829301"
            result = find_needle_token_range(tokenizer, input_ids, target_key, target_value)
            start, end, key_start, key_end, val_start, val_end = result
            if start is not None:
                print(f"Needle line tokens: [{start}, {end})")
                print(f"  Key '{target_key}' at tokens [{key_start}, {key_end})")
                print(f"  Value '{target_value}' at tokens [{val_start}, {val_end})")
                needle_decoded = tokenizer.decode(input_ids[0, start:end])
                print(f"  Decoded: {needle_decoded[:120]}")
            else:
                print("WARNING: Could not locate needle tokens exactly")
                continue

            # Run with CriticalSnapKV
            _eviction_log.clear()

            press = CriticalKVPress(
                press=SnapKVPress(compression_ratio=args.compression_ratio, window_size=64, kernel_size=5)
            )

            with torch.no_grad(), press(model):
                from transformers import DynamicCache
                cache = DynamicCache()
                outputs = model(input_ids, past_key_values=cache)

            # Analyze: how many layers kept the needle tokens?
            # Focus on key+value tokens (the critical part)
            needle_range = set(range(key_start, val_end))
            n_layers = len(_eviction_log)

            layers_keeping_needle = 0
            layers_partial_needle = 0
            needle_scores_by_layer = []

            print(f"\nEviction analysis across {n_layers} layers:")
            print(f"  Total tokens: {_eviction_log[0]['k_len']}, Kept: {_eviction_log[0]['n_kept']}")

            for entry in _eviction_log:
                kept = set(entry["kept_indices"])
                needle_kept = needle_range & kept
                fraction = len(needle_kept) / len(needle_range)

                # Average score for needle vs all tokens
                scores = entry["scores"]
                needle_scores = [scores[i] for i in needle_range if i < len(scores)]
                avg_needle_score = sum(needle_scores) / len(needle_scores) if needle_scores else 0
                avg_all_score = sum(scores) / len(scores)

                needle_scores_by_layer.append({
                    "layer": entry["layer"],
                    "needle_kept_frac": fraction,
                    "avg_needle_score": avg_needle_score,
                    "avg_all_score": avg_all_score,
                    "needle_score_ratio": avg_needle_score / avg_all_score if avg_all_score > 0 else 0,
                })

                if fraction == 1.0:
                    layers_keeping_needle += 1
                elif fraction > 0:
                    layers_partial_needle += 1

            print(f"\n  Layers keeping ALL needle tokens: {layers_keeping_needle}/{n_layers}")
            print(f"  Layers keeping SOME needle tokens: {layers_partial_needle}/{n_layers}")
            print(f"  Layers keeping NO needle tokens: {n_layers - layers_keeping_needle - layers_partial_needle}/{n_layers}")

            # Show per-layer details for a few key layers
            print(f"\n  Per-layer needle retention (layer: kept_fraction, score_ratio):")
            for info in needle_scores_by_layer:
                marker = ""
                if info["needle_kept_frac"] == 0:
                    marker = " *** FULLY EVICTED ***"
                elif info["needle_kept_frac"] < 0.5:
                    marker = " ** mostly evicted **"
                print(f"    Layer {info['layer']:2d}: kept={info['needle_kept_frac']:.1%}, "
                      f"needle_score={info['avg_needle_score']:.4f}, "
                      f"avg_score={info['avg_all_score']:.4f}, "
                      f"ratio={info['needle_score_ratio']:.2f}{marker}")

            # Also generate without compression for comparison
            with torch.no_grad():
                outputs_full = model(input_ids)

            # Compare next-token predictions
            logits_compressed = outputs.logits[0, -1]
            logits_full = outputs_full.logits[0, -1]

            top5_compressed = torch.topk(logits_compressed, 5)
            top5_full = torch.topk(logits_full, 5)

            print(f"\n  Next token predictions:")
            print(f"    Full KV:       {[tokenizer.decode([t]) for t in top5_full.indices.tolist()]}")
            print(f"    CriticalSnapKV: {[tokenizer.decode([t]) for t in top5_compressed.indices.tolist()]}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
