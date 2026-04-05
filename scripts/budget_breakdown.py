#!/usr/bin/env python3
"""Budget breakdown: at cr=0.7 and cr=0.9, which token CATEGORIES get the budget?

Key question: are distractors properly evicted? How much budget goes to each category?
"""

import torch
import random
import math
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half
from kvpress.attention_patch import patch_attention_functions

patch_attention_functions()

ADJECTIVES = ["silent", "golden", "dusty", "hollow", "bitter", "crimson",
              "gentle", "fierce", "mystic", "frozen", "swift", "ancient",
              "bright", "dark", "wild", "calm", "bold", "shy", "proud", "humble"]
NOUNS = ["rabbit", "hawk", "maple", "pine", "stream", "tide", "wolf",
         "eagle", "river", "stone", "flame", "shadow", "storm", "leaf",
         "crystal", "thunder", "ocean", "moon", "star", "wind"]


def build_niah_prompt(num_distractors, target_key, target_value, needle_pos_frac, seed=50):
    random.seed(seed)
    needles = []
    used_keys = {target_key}
    distractor_info = []
    for _ in range(num_distractors):
        while True:
            key = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
            if key not in used_keys:
                used_keys.add(key)
                break
        value = str(random.randint(1000000, 9999999))
        needles.append(f"One of the special magic numbers for {key} is: {value}.")
        distractor_info.append((key, value))
    target_needle = f"One of the special magic numbers for {target_key} is: {target_value}."
    target_pos = max(0, min(num_distractors, int(num_distractors * needle_pos_frac)))
    needles.insert(target_pos, target_needle)
    context = "\n".join(needles)
    return (
        f"A special magic number is hidden within the following text. "
        f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {target_key} mentioned in the provided text?"
    ), distractor_info


def find_token_positions(tokenizer, ctx_ids, search_str):
    ctx_len = ctx_ids.shape[1]
    full_text = tokenizer.decode(ctx_ids[0])
    char_pos = full_text.find(search_str)
    if char_pos < 0:
        return []
    positions = []
    cum = ""
    for i in range(ctx_len):
        prev_len = len(cum)
        cum += tokenizer.decode(ctx_ids[0, i])
        if len(cum) > char_pos and prev_len < char_pos + len(search_str):
            positions.append(i)
    return positions


def find_user_input_positions(tokenizer, context_ids, ctx_len):
    full_text = tokenizer.decode(context_ids[0])
    user_marker = "user\n"
    user_start_char = full_text.find(user_marker)
    if user_start_char < 0:
        return list(range(max(0, int(ctx_len * 0.8)), ctx_len))
    user_start_char += len(user_marker)
    im_end = "<|im_end|>"
    user_end_char = full_text.find(im_end, user_start_char)
    if user_end_char < 0:
        user_end_char = len(full_text)
    positions = []
    cum_text = ""
    for i in range(ctx_len):
        tok_text = tokenizer.decode(context_ids[0, i])
        start_char = len(cum_text)
        cum_text += tok_text
        end_char = len(cum_text)
        if end_char > user_start_char and start_char < user_end_char:
            positions.append(i)
    return positions if positions else list(range(max(0, ctx_len - 50), ctx_len))


def classify_tokens(tokenizer, ctx_ids, ctx_len, target_key, target_value, distractor_info):
    """Classify every token into a category."""
    # category: 0=system, 1=instruction, 2=distractor, 3=needle_key, 4=needle_value,
    #           5=needle_other, 6=question, 7=distractor_key, 8=distractor_value
    category = [0] * ctx_len  # default: system (chat template tokens)

    full_text = tokenizer.decode(ctx_ids[0])

    # Find instruction
    instr = "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.\n"
    instr_pos = find_token_positions(tokenizer, ctx_ids, instr)
    for p in instr_pos:
        category[p] = 1

    # Find question
    question = f"What is the special magic number for {target_key} mentioned in the provided text?"
    q_pos = find_token_positions(tokenizer, ctx_ids, question)
    for p in q_pos:
        category[p] = 6

    # Find target needle
    needle = f"One of the special magic numbers for {target_key} is: {target_value}."
    needle_pos = find_token_positions(tokenizer, ctx_ids, needle)
    key_pos = find_token_positions(tokenizer, ctx_ids, target_key)
    value_pos = find_token_positions(tokenizer, ctx_ids, target_value)
    # Mark needle_other first, then overwrite key/value
    for p in needle_pos:
        category[p] = 5
    # Only mark the FIRST occurrence of target_key (in needle, not in question)
    needle_start = needle_pos[0] if needle_pos else 0
    needle_end = needle_pos[-1] if needle_pos else ctx_len
    for p in key_pos:
        if needle_start <= p <= needle_end:
            category[p] = 3
    for p in value_pos:
        category[p] = 4

    # Find each distractor
    for dk, dv in distractor_info:
        d_needle = f"One of the special magic numbers for {dk} is: {dv}."
        d_pos = find_token_positions(tokenizer, ctx_ids, d_needle)
        dk_pos = find_token_positions(tokenizer, ctx_ids, dk)
        dv_pos = find_token_positions(tokenizer, ctx_ids, dv)
        for p in d_pos:
            if category[p] == 0:  # don't overwrite if already classified
                category[p] = 2  # distractor general
        for p in dk_pos:
            if p in d_pos or (d_pos and d_pos[0] <= p <= d_pos[-1]):
                category[p] = 7  # distractor key
        for p in dv_pos:
            if p in d_pos or (d_pos and d_pos[0] <= p <= d_pos[-1]):
                category[p] = 8  # distractor value

    return category


def compute_scores(model, tokenizer, ctx_ids, n_layers, n_kv_heads, n_q_heads, head_dim, chunk_size=1024):
    n_groups = n_q_heads // n_kv_heads
    ctx_len = ctx_ids.shape[1]
    user_pos = find_user_input_positions(tokenizer, ctx_ids, ctx_len)
    user_pos_t = torch.tensor(user_pos, dtype=torch.long, device=model.device)
    final_scores = torch.zeros(n_layers, 1, n_kv_heads, ctx_len, dtype=torch.float32)

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return
            bsz, seq_len, _ = hidden_states.shape
            if seq_len != ctx_len:
                return
            position_embeddings = kwargs.get("position_embeddings", None)
            with torch.no_grad():
                q = module.q_proj(hidden_states).view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                k = module.k_proj(hidden_states).view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                    k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
                q_grouped = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                scale = math.sqrt(head_dim)
                for hi in range(n_kv_heads):
                    k_h = k[0, hi]
                    q_h = q_grouped[0, hi]
                    user_q = q_h[:, user_pos_t, :]
                    user_logits = torch.matmul(user_q, k_h.T) / scale
                    for ui in range(len(user_pos)):
                        user_logits[:, ui, user_pos[ui]+1:] = float('-inf')
                    user_attn = torch.softmax(user_logits.float(), dim=-1)
                    step1 = user_attn.amax(dim=(0, 1))
                    del user_logits, user_attn, user_q
                    fan_in = torch.zeros(ctx_len, dtype=torch.float32, device=k_h.device)
                    for start in range(0, ctx_len, chunk_size):
                        end = min(start + chunk_size, ctx_len)
                        chunk_q = q_h[:, start:end, :]
                        logits = torch.matmul(chunk_q, k_h.T) / scale
                        for ci in range(end - start):
                            logits[:, ci, start + ci + 1:] = float('-inf')
                        chunk_attn = torch.softmax(logits.float(), dim=-1).amax(dim=0)
                        fan_in += chunk_attn.sum(dim=0)
                        del chunk_q, logits, chunk_attn
                    inv_fanin = 1.0 / (fan_in + 1e-6)
                    inv_fanin = inv_fanin / inv_fanin.sum()
                    step1_weighted = step1 * inv_fanin
                    step2 = torch.zeros(ctx_len, dtype=torch.float32, device=k_h.device)
                    step3 = torch.zeros(ctx_len, dtype=torch.float32, device=k_h.device)
                    for start in range(0, ctx_len, chunk_size):
                        end = min(start + chunk_size, ctx_len)
                        chunk_q = q_h[:, start:end, :]
                        logits = torch.matmul(chunk_q, k_h.T) / scale
                        for ci in range(end - start):
                            logits[:, ci, start + ci + 1:] = float('-inf')
                        chunk_attn = torch.softmax(logits.float(), dim=-1).amax(dim=0)
                        chunk_step2 = torch.matmul(chunk_attn, step1_weighted)
                        step2[start:end] = chunk_step2
                        step3 += torch.matmul(chunk_attn.T, chunk_step2)
                        del chunk_q, logits, chunk_attn, chunk_step2
                    def norm01(x):
                        mn, mx = x.min(), x.max()
                        return (x - mn) / (mx - mn + 1e-10)
                    combined = norm01(step1) + norm01(step2) + norm01(step3)
                    final_scores[layer_idx, 0, hi, :] = combined.cpu()
                    del step1, step2, step3, fan_in, inv_fanin, step1_weighted
                del q, k, q_grouped
        return hook_fn

    hooks = []
    for layer in model.model.layers:
        h = layer.self_attn.register_forward_hook(
            make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True)
        hooks.append(h)
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=cache)
    for h in hooks:
        h.remove()
    del cache
    torch.cuda.empty_cache()
    return final_scores


def main():
    model_path = "Qwen/Qwen3-8B"
    target_key = "mystic-thunder"
    target_value = "7156842"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_q_heads

    for n_dist in [30, 100]:
        print(f"\n{'=' * 80}")
        print(f"  {n_dist} distractors")
        print(f"{'=' * 80}")

        prompt, distractor_info = build_niah_prompt(n_dist, target_key, target_value, 0.5, seed=50)
        separator = "#" * (len(prompt) + 10)
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + separator}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        ctx_text, q_suffix = full_text.split(separator)
        ctx_ids = tokenizer.encode(ctx_text, return_tensors="pt",
                                   add_special_tokens=False).to(model.device)
        ctx_len = ctx_ids.shape[1]
        print(f"  Context: {ctx_len} tokens")

        # Classify tokens
        category = classify_tokens(tokenizer, ctx_ids, ctx_len, target_key, target_value, distractor_info)
        cat_names = {0: 'system', 1: 'instruction', 2: 'distractor_txt', 3: 'needle_key',
                     4: 'needle_value', 5: 'needle_other', 6: 'question',
                     7: 'distractor_key', 8: 'distractor_value'}

        # Count tokens per category
        cat_counts = {}
        for c in range(9):
            cat_counts[c] = sum(1 for x in category if x == c)
        print(f"\n  Token distribution:")
        for c in sorted(cat_counts.keys()):
            if cat_counts[c] > 0:
                print(f"    {cat_names[c]:>20}: {cat_counts[c]:>4} ({cat_counts[c]/ctx_len*100:.1f}%)")

        # Compute scores
        print(f"\n  Computing 3-hop scores...")
        scores = compute_scores(model, tokenizer, ctx_ids, n_layers, n_kv_heads, n_q_heads, head_dim)

        # Average score per category
        avg_scores_per_cat = {}
        for c in range(9):
            positions = [i for i, x in enumerate(category) if x == c]
            if positions:
                cat_scores = scores[:, 0, :, torch.tensor(positions)]  # [n_layers, n_kv_heads, n_pos]
                avg_scores_per_cat[c] = cat_scores.mean().item()

        print(f"\n  Average 3-hop score per category:")
        for c in sorted(avg_scores_per_cat.keys(), key=lambda x: -avg_scores_per_cat[x]):
            print(f"    {cat_names[c]:>20}: {avg_scores_per_cat[c]:.4f}")

        # Budget breakdown at different CRs
        for cr in [0.5, 0.7, 0.9, 0.95]:
            total = n_layers * 1 * n_kv_heads * ctx_len
            n_pruned = int(total * cr)
            flat = scores.reshape(-1)
            sorted_s, _ = flat.sort()
            threshold = sorted_s[min(n_pruned, len(sorted_s)-1)].item()
            kept_mask = (scores[:, 0, :, :] >= threshold)  # [n_layers, n_kv_heads, ctx_len]

            # Per-category: average retention rate
            print(f"\n  CR={cr} (keep {100*(1-cr):.0f}%, threshold={threshold:.4f}):")
            print(f"  {'Category':>20} {'#Tokens':>8} {'AvgRetain':>10} {'Budget%':>8} {'TokenBudget':>12}")

            total_kept = kept_mask.float().sum().item()
            for c in sorted(cat_counts.keys()):
                positions = [i for i, x in enumerate(category) if x == c]
                if not positions:
                    continue
                pos_t = torch.tensor(positions)
                cat_kept = kept_mask[:, :, pos_t].float()
                avg_retain = cat_kept.mean().item()
                total_cat_kept = cat_kept.sum().item()
                budget_pct = total_cat_kept / max(total_kept, 1) * 100
                # "Token budget" = equivalent number of unique token positions kept
                # (avg retention × n_positions)
                equiv_tokens = avg_retain * len(positions)
                print(f"    {cat_names[c]:>20} {len(positions):>8} {avg_retain:>9.1%} {budget_pct:>7.1f}% {equiv_tokens:>11.1f}")

            # Summary: useful vs waste
            useful_cats = [1, 3, 4, 5, 6]  # instruction, needle_key/value/other, question
            waste_cats = [2, 7, 8]  # distractor_txt/key/value
            useful_kept = sum(
                kept_mask[:, :, torch.tensor([i for i, x in enumerate(category) if x == c])].float().sum().item()
                for c in useful_cats if any(x == c for x in category)
            )
            waste_kept = sum(
                kept_mask[:, :, torch.tensor([i for i, x in enumerate(category) if x == c])].float().sum().item()
                for c in waste_cats if any(x == c for x in category)
            )
            system_kept = kept_mask[:, :, torch.tensor([i for i, x in enumerate(category) if x == 0])].float().sum().item() if any(x == 0 for x in category) else 0

            print(f"\n    SUMMARY:")
            print(f"      Useful (instr+needle+question): {useful_kept:.0f} slots ({useful_kept/max(total_kept,1)*100:.1f}%)")
            print(f"      Waste  (distractor all):        {waste_kept:.0f} slots ({waste_kept/max(total_kept,1)*100:.1f}%)")
            print(f"      System (chat template):         {system_kept:.0f} slots ({system_kept/max(total_kept,1)*100:.1f}%)")
            print(f"      Total kept:                     {total_kept:.0f} / {total}")

        torch.cuda.empty_cache()

    print("\n\nDONE")


if __name__ == "__main__":
    main()
