#!/usr/bin/env python3
"""Analyze KVzip score_val: what tokens get high/low scores?

Use the correct pipeline pattern to get score_val, then analyze:
1. Score distribution across token types (numbers, text, template)
2. At cr=0.95, which tokens are kept per head?
3. Are ALL needles' numbers kept, or just the target?
"""

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvpress.presses.kvzip_press import KVzipPress

ADJECTIVES = ["silent", "golden", "dusty", "hollow", "bitter", "crimson",
              "gentle", "fierce", "mystic", "frozen", "swift", "ancient",
              "bright", "dark", "wild", "calm", "bold", "shy", "proud", "humble"]
NOUNS = ["rabbit", "hawk", "maple", "pine", "stream", "tide", "wolf",
         "eagle", "river", "stone", "flame", "shadow", "storm", "leaf",
         "crystal", "thunder", "ocean", "moon", "star", "wind"]


def main():
    target_key = "mystic-thunder"
    target_value = "7156842"

    # Build prompt (same as stress test)
    random.seed(50)
    needles = []
    used_keys = {target_key}
    for _ in range(30):
        while True:
            key = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
            if key not in used_keys:
                used_keys.add(key)
                break
        value = str(random.randint(1000000, 9999999))
        needles.append(f"One of the special magic numbers for {key} is: {value}.")

    target_needle = f"One of the special magic numbers for {target_key} is: {target_value}."
    target_pos = max(0, int(30 * 0.5))
    needles.insert(target_pos, target_needle)

    context = "\n".join(needles)
    prompt = (
        f"A special magic number is hidden within the following text. "
        f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {target_key} mentioned in the provided text?"
    )

    print("=" * 90)
    print("KVZIP SCORE ANALYSIS: What tokens get high/low scores?")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    # Tokenize
    separator = "#" * (len(prompt) + 10)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt + separator}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    context_text, _ = full_text.split(separator)
    context_ids = tokenizer.encode(context_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    context_length = context_ids.shape[1]

    # Decode each token for analysis
    tokens = [tokenizer.decode(context_ids[0, i]) for i in range(context_length)]

    print(f"\nContext length: {context_length} tokens")

    # Find target needle tokens
    target_text = f"One of the special magic numbers for {target_key} is: {target_value}."
    target_token_ids = tokenizer.encode(target_text, add_special_tokens=False)
    # Find target position in context_ids
    target_start = None
    for i in range(context_length - len(target_token_ids)):
        if context_ids[0, i:i+len(target_token_ids)].tolist() == target_token_ids:
            target_start = i
            break

    if target_start:
        print(f"Target needle at positions {target_start}-{target_start+len(target_token_ids)-1}")
        print(f"Target tokens: {[tokenizer.decode(t) for t in target_token_ids]}")
    else:
        print("WARNING: Could not find exact target needle position")
        # Try to find "mystic-thunder" in tokens
        for i in range(context_length - 5):
            chunk = tokenizer.decode(context_ids[0, i:i+10])
            if "mystic" in chunk.lower() and "thunder" in chunk.lower():
                target_start = i
                print(f"  Found approximate target at position {i}: {chunk}")
                break

    # Run KVzip to get scores
    press = KVzipPress(compression_ratio=0.5, layerwise=True)

    # Monkey-patch to save score_val
    saved_scores = {}
    original_compress = press.compress_post
    def patched_compress(model):
        saved_scores['score_val'] = press.score_val.clone()
        original_compress(model)
    press.compress_post = patched_compress

    cache = DynamicCache()
    with torch.no_grad(), press(model):
        model.model(input_ids=context_ids, past_key_values=cache)

    score_val = saved_scores.get('score_val')
    if score_val is None:
        print("ERROR: score_val not captured")
        return

    print(f"\nscore_val shape: {score_val.shape}")
    # shape: [n_layers, 1, n_kv_heads, context_length]

    n_layers, _, n_heads, ctx_len = score_val.shape

    # ═══════════════════════════════════════════════════════
    # Analysis 1: Per-token score (averaged across all layers and heads)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("ANALYSIS 1: Average score per token position")
    print("=" * 90)

    avg_score = score_val.mean(dim=(0, 1, 2))  # [ctx_len]
    print(f"Score range: [{avg_score.min():.6f}, {avg_score.max():.6f}]")
    print(f"Score mean: {avg_score.mean():.6f}, std: {avg_score.std():.6f}")

    # Top 20 tokens by average score
    top_k = 30
    top_indices = torch.topk(avg_score, top_k).indices.tolist()
    print(f"\nTop {top_k} tokens by avg score:")
    for idx in top_indices:
        tok = tokens[idx] if idx < len(tokens) else "?"
        print(f"  pos={idx:4d}  score={avg_score[idx]:.6f}  token='{tok}'")

    # Bottom 20 tokens by average score
    bot_indices = torch.topk(-avg_score, top_k).indices.tolist()
    print(f"\nBottom {top_k} tokens by avg score:")
    for idx in bot_indices:
        tok = tokens[idx] if idx < len(tokens) else "?"
        print(f"  pos={idx:4d}  score={avg_score[idx]:.6f}  token='{tok}'")

    # ═══════════════════════════════════════════════════════
    # Analysis 2: Score of target needle vs other needles
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("ANALYSIS 2: Target needle vs other needles")
    print("=" * 90)

    # Find all number tokens (7-digit values)
    # Each needle has format: "... for KEY is: VALUE."
    # Find all "is:" positions and the 7 digit tokens after them
    needle_scores = []
    for i in range(context_length - 10):
        # Look for digit sequences (the 7-digit values)
        tok = tokens[i].strip()
        if tok.isdigit() and len(tok) >= 1:
            # Check if this is part of a needle value
            # Look at surrounding context
            context_window = tokenizer.decode(context_ids[0, max(0,i-15):i+10])
            if "magic numbers for" in context_window:
                is_target = False
                if target_start and abs(i - target_start) < 25:
                    is_target = True
                score = avg_score[i].item()
                needle_scores.append({
                    'pos': i,
                    'token': tok,
                    'score': score,
                    'is_target': is_target,
                    'context': context_window[-60:]
                })

    target_digit_scores = [n['score'] for n in needle_scores if n['is_target']]
    other_digit_scores = [n['score'] for n in needle_scores if not n['is_target']]

    print(f"\nTarget needle digit scores (avg): {sum(target_digit_scores)/max(len(target_digit_scores),1):.6f}")
    print(f"Other needles digit scores (avg): {sum(other_digit_scores)/max(len(other_digit_scores),1):.6f}")
    print(f"Target digits: {len(target_digit_scores)}, Other digits: {len(other_digit_scores)}")

    print("\nTarget needle digit tokens:")
    for n in needle_scores:
        if n['is_target']:
            print(f"  pos={n['pos']}  score={n['score']:.6f}  tok='{n['token']}'")

    print("\nSample other needle digit tokens (first 10):")
    for n in [x for x in needle_scores if not x['is_target']][:10]:
        print(f"  pos={n['pos']}  score={n['score']:.6f}  tok='{n['token']}'  ctx=...{n['context'][-40:]}")

    # ═══════════════════════════════════════════════════════
    # Analysis 3: Token type analysis
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("ANALYSIS 3: Score by token type")
    print("=" * 90)

    # Categorize tokens
    digit_scores = []
    word_scores = []
    punct_scores = []
    template_scores = []  # "One", "of", "the", "special", "magic", "numbers", "for", "is"
    template_words = {"one", "of", "the", "special", "magic", "numbers", "for", "is", ":", "."}

    for i in range(context_length):
        tok = tokens[i].strip().lower()
        s = avg_score[i].item()
        if tok.isdigit():
            digit_scores.append(s)
        elif tok in template_words:
            template_scores.append(s)
        elif tok.isalpha() and len(tok) > 1:
            word_scores.append(s)
        elif tok in [':', '.', ',', '\n', '-']:
            punct_scores.append(s)

    print(f"  Digits:    n={len(digit_scores):4d}  avg={sum(digit_scores)/max(len(digit_scores),1):.6f}")
    print(f"  Template:  n={len(template_scores):4d}  avg={sum(template_scores)/max(len(template_scores),1):.6f}")
    print(f"  Words:     n={len(word_scores):4d}  avg={sum(word_scores)/max(len(word_scores),1):.6f}")
    print(f"  Punct:     n={len(punct_scores):4d}  avg={sum(punct_scores)/max(len(punct_scores),1):.6f}")

    # ═══════════════════════════════════════════════════════
    # Analysis 4: Simulate cr=0.95 per-head retention
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("ANALYSIS 4: Simulated per-head retention at various CRs")
    print("=" * 90)

    # Focus on a few key layers
    key_layers = [7, 9, 13, 15]

    for cr in [0.5, 0.7, 0.9, 0.95]:
        print(f"\n  --- CR = {cr} ---")
        for layer_idx in key_layers:
            layer_scores = score_val[layer_idx, 0]  # [n_heads, ctx_len]
            n_total = n_heads * ctx_len
            n_pruned = int(n_total * cr)

            # Global bottom-k across all heads
            flat_scores = layer_scores.reshape(-1)
            _, pruned_indices = torch.topk(-flat_scores, n_pruned)
            kept_mask = torch.ones(n_total, dtype=torch.bool)
            kept_mask[pruned_indices] = False
            kept_mask = kept_mask.reshape(n_heads, ctx_len)

            per_head_kept = kept_mask.sum(dim=1).tolist()
            print(f"  L{layer_idx}: per-head kept = {per_head_kept}  (total kept={sum(per_head_kept)}/{n_total})")

            # Check if target needle digits are kept
            if target_start:
                for h in range(n_heads):
                    target_kept = []
                    target_evicted = []
                    for offset in range(min(20, len(target_token_ids))):
                        pos = target_start + offset
                        if pos < ctx_len:
                            tok = tokens[pos]
                            if kept_mask[h, pos]:
                                target_kept.append((pos, tok.strip()))
                            else:
                                target_evicted.append((pos, tok.strip()))
                    if target_kept or h in [2, 3, 5]:  # show retrieval heads
                        kept_str = ",".join([f"'{t}'" for _, t in target_kept[:8]])
                        evicted_str = ",".join([f"'{t}'" for _, t in target_evicted[:8]])
                        print(f"    KV{h}: target kept=[{kept_str}] evicted=[{evicted_str}]")

    # ═══════════════════════════════════════════════════════
    # Analysis 5: Are ALL needles treated equally?
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("ANALYSIS 5: Per-needle average score (are all needles equal?)")
    print("=" * 90)

    # Find each needle's value position and compute avg score
    # Re-parse needles
    random.seed(50)
    all_needles_info = []
    used_keys2 = {target_key}
    for idx in range(30):
        while True:
            key = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
            if key not in used_keys2:
                used_keys2.add(key)
                break
        value = str(random.randint(1000000, 9999999))
        all_needles_info.append((key, value, False))

    all_needles_info.insert(target_pos, (target_key, target_value, True))

    # For each needle, find its tokens and compute average score
    for key, value, is_target in all_needles_info:
        needle_text = f"magic numbers for {key} is: {value}."
        needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
        # Find in context
        found = False
        for i in range(context_length - len(needle_ids)):
            if context_ids[0, i:i+len(needle_ids)].tolist() == needle_ids:
                needle_avg = avg_score[i:i+len(needle_ids)].mean().item()
                needle_max = avg_score[i:i+len(needle_ids)].max().item()
                marker = " <<<TARGET" if is_target else ""
                print(f"  {key:20s}: avg={needle_avg:.6f} max={needle_max:.6f}  pos={i}{marker}")
                found = True
                break
        if not found:
            # Try partial match
            partial = tokenizer.encode(value, add_special_tokens=False)
            for i in range(context_length - len(partial)):
                if context_ids[0, i:i+len(partial)].tolist() == partial:
                    val_avg = avg_score[i:i+len(partial)].mean().item()
                    marker = " <<<TARGET" if is_target else ""
                    print(f"  {key:20s}: val_avg={val_avg:.6f}  pos={i}{marker}")
                    found = True
                    break
            if not found:
                print(f"  {key:20s}: NOT FOUND")

    print("\n\nDONE")


if __name__ == "__main__":
    main()
