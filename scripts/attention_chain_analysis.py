#!/usr/bin/env python3
"""Detailed attention chain analysis for NIAH task.

Expected chain (per user's hypothesis):
1. Question tokens → sink tokens + needle front ("mystic-thunder")
2. Period at end of needle → needle front ("mystic-thunder")
3. Period → whole needle sentence (including value digits)
→ Graph expansion should propagate: question→needle_front + period→needle_front→whole_sentence

We need to verify each step and find where the chain breaks.
"""

import torch
import random
import math
import json
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half

ADJECTIVES = ["silent", "golden", "dusty", "hollow", "bitter", "crimson",
              "gentle", "fierce", "mystic", "frozen", "swift", "ancient",
              "bright", "dark", "wild", "calm", "bold", "shy", "proud", "humble"]
NOUNS = ["rabbit", "hawk", "maple", "pine", "stream", "tide", "wolf",
         "eagle", "river", "stone", "flame", "shadow", "storm", "leaf",
         "crystal", "thunder", "ocean", "moon", "star", "wind"]


def build_prompt(num_distractors, target_key, target_value, needle_pos_frac, seed=50):
    random.seed(seed)
    needles = []
    used_keys = {target_key}
    for _ in range(num_distractors):
        while True:
            key = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
            if key not in used_keys:
                used_keys.add(key)
                break
        value = str(random.randint(1000000, 9999999))
        needles.append(f"One of the special magic numbers for {key} is: {value}.")
    target_needle = f"One of the special magic numbers for {target_key} is: {target_value}."
    target_pos = max(0, min(num_distractors, int(num_distractors * needle_pos_frac)))
    needles.insert(target_pos, target_needle)
    context = "\n".join(needles)
    return (
        f"A special magic number is hidden within the following text. "
        f"Make sure to memorize it. I will quiz you about the number afterwards.\n"
        f"{context}\n"
        f"What is the special magic number for {target_key} mentioned in the provided text?"
    )


def find_token_groups(tokenizer, ctx_ids, ctx_len, target_key, target_value):
    """Find specific token positions for chain analysis."""
    tokens = [tokenizer.decode(ctx_ids[0, i]) for i in range(ctx_len)]

    groups = {
        'sink': list(range(min(5, ctx_len))),  # first 5 tokens
        'target_key_front': [],    # "mystic" or "myst" tokens
        'target_key_full': [],     # all "mystic-thunder" tokens
        'target_value': [],        # "7156842" digit tokens
        'target_colon': [],        # ":" before the value
        'target_period': [],       # "." at end of target needle
        'target_sentence': [],     # entire target needle sentence
        'distractor_sentences': [],# ranges of distractor sentences
        'distractor_values': [],   # value tokens of distractors
    }

    # Find the target needle sentence by scanning decoded windows
    target_sent_start = None
    target_sent_end = None
    for i in range(ctx_len - 10):
        window = tokenizer.decode(ctx_ids[0, i:i+30])
        if target_key in window and target_value in window:
            # Found the needle area, now find exact boundaries
            # Scan backward to find "One"
            for s in range(max(0, i-5), i+5):
                tok = tokens[s].strip()
                if tok.startswith("One") or tok.startswith("\nOne"):
                    target_sent_start = s
                    break
            if target_sent_start is None:
                target_sent_start = i
            # Scan forward to find the period
            for e in range(i, min(ctx_len, i+35)):
                tok = tokens[e].strip()
                if tok.endswith('.') and e > i + 5:
                    target_sent_end = e
                    break
            if target_sent_end is None:
                target_sent_end = min(ctx_len-1, target_sent_start + 25)
            break

    if target_sent_start is not None:
        groups['target_sentence'] = list(range(target_sent_start, target_sent_end + 1))

    # Find specific token types within the target sentence
    for i in range(ctx_len):
        tok = tokens[i]
        tok_lower = tok.lower().strip()

        # Check if in target sentence range
        if target_sent_start and target_sent_start <= i <= target_sent_end:
            # "mystic" tokens
            if 'myst' in tok_lower or 'ic' in tok_lower and i > 0 and 'myst' in tokens[i-1].lower():
                groups['target_key_front'].append(i)
            # Full key tokens (mystic-thunder)
            if any(part in tok_lower for part in ['myst', 'ic', '-', 'thunder', 'thund']):
                if 'myst' in tok_lower or (i > 0 and any(p in tokens[i-1].lower() for p in ['myst', 'ic', '-'])):
                    groups['target_key_full'].append(i)
                elif 'thunder' in tok_lower or 'thund' in tok_lower:
                    groups['target_key_full'].append(i)
            # Value digits
            if any(c.isdigit() for c in tok) and target_value[:3] in tokenizer.decode(ctx_ids[0, max(0,i-2):i+3]):
                groups['target_value'].append(i)
            # Colon
            if ':' in tok:
                groups['target_colon'].append(i)
            # Period
            if tok.strip().endswith('.'):
                groups['target_period'].append(i)

    # Better approach: decode the target sentence and match tokens
    if target_sent_start is not None:
        sent_text = tokenizer.decode(ctx_ids[0, target_sent_start:target_sent_end+1])
        print(f"\n  Target sentence: '{sent_text}'")
        print(f"  Token positions {target_sent_start}-{target_sent_end}:")
        for i in range(target_sent_start, target_sent_end + 1):
            print(f"    [{i:4d}] '{tokens[i]}'")

    # Also identify a few distractor sentences for comparison
    dist_count = 0
    i = 0
    while i < ctx_len - 5 and dist_count < 3:
        window = tokenizer.decode(ctx_ids[0, i:i+30])
        if "One of the special" in window and target_key not in window:
            dist_start = i
            for e in range(i+5, min(ctx_len, i+35)):
                if tokens[e].strip().endswith('.'):
                    groups['distractor_sentences'].append(list(range(dist_start, e+1)))
                    dist_count += 1
                    i = e + 1
                    break
            else:
                i += 1
        else:
            i += 1

    return groups, tokens


def compute_full_attention(model, context_ids, question_ids):
    """Compute full attention matrices for both prefill and question phases."""
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = context_ids.shape[1]
    q_len = question_ids.shape[1]

    # Storage: per-layer attention matrices
    prefill_attn = {}   # [n_kv_heads, ctx_len, ctx_len] (max over groups)
    question_attn = {}  # [n_kv_heads, q_len, ctx_len]
    phase = {'current': 'prefill'}

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return
            bsz, seq_len, _ = hidden_states.shape
            position_embeddings = kwargs.get("position_embeddings", None)

            with torch.no_grad():
                if phase['current'] == 'prefill' and seq_len == ctx_len:
                    q = module.q_proj(hidden_states).view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                    k = module.k_proj(hidden_states).view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                        q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                        k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
                    q_g = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                    k_e = k.unsqueeze(2)
                    attn = torch.matmul(q_g, k_e.transpose(-2, -1)) / math.sqrt(head_dim)
                    causal = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=attn.device), diagonal=1)
                    attn = torch.softmax(attn + causal[None, None, None], dim=-1)
                    # Max over query groups: [1, n_kv_heads, ctx_len, ctx_len]
                    prefill_attn[layer_idx] = attn.amax(dim=2)[0].float().cpu()

                elif phase['current'] == 'question' and seq_len == q_len:
                    past_kv = kwargs.get("past_key_values", None)
                    if past_kv is None:
                        return
                    k = past_kv.layers[layer_idx].keys[:, :, :ctx_len, :]
                    q = module.q_proj(hidden_states).view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                        q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                    q_g = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                    k_e = k.unsqueeze(2)
                    attn = torch.softmax(torch.matmul(q_g, k_e.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1)
                    # Max over query groups: [n_kv_heads, q_len, ctx_len]
                    question_attn[layer_idx] = attn.amax(dim=2)[0].float().cpu()
        return hook_fn

    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.self_attn.register_forward_hook(
            make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True))

    phase['current'] = 'prefill'
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=context_ids, past_key_values=cache)

    phase['current'] = 'question'
    pos_ids = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        model.model(input_ids=question_ids, past_key_values=cache, position_ids=pos_ids)

    for h in hooks:
        h.remove()
    del cache

    return prefill_attn, question_attn


def analyze_chain(prefill_attn, question_attn, groups, tokens, n_layers, ctx_len, q_len):
    """Analyze each step of the hypothesized attention chain."""

    target_key_front = groups['target_key_front']
    target_key_full = groups['target_key_full']
    target_value = groups['target_value']
    target_period = groups['target_period']
    target_sentence = groups['target_sentence']
    sink = groups['sink']

    print(f"\n  Token groups:")
    print(f"    Sink tokens:        {sink}")
    print(f"    Target key front:   {target_key_front} = {[tokens[i] for i in target_key_front]}")
    print(f"    Target key full:    {target_key_full} = {[tokens[i] for i in target_key_full]}")
    print(f"    Target value:       {target_value} = {[tokens[i] for i in target_value]}")
    print(f"    Target period:      {target_period} = {[tokens[i] for i in target_period]}")
    print(f"    Target sentence:    {target_sentence[0]}-{target_sentence[-1]} ({len(target_sentence)} tokens)")

    # Select representative layers to analyze
    analyze_layers = [0, 3, 7, 11, 15, 19, 23, 27, 31, 35]
    analyze_layers = [l for l in analyze_layers if l < n_layers]

    print(f"\n{'='*100}")
    print(f"STEP 1: Question → Context (which context tokens does the question attend to?)")
    print(f"{'='*100}")

    for l in analyze_layers:
        if l not in question_attn:
            continue
        qa = question_attn[l]  # [n_kv_heads, q_len, ctx_len]

        # Average attention from all question tokens to context
        avg_q2c = qa.mean(dim=1)  # [n_kv_heads, ctx_len]

        # How much attention goes to each group?
        attn_to_sink = avg_q2c[:, sink].sum(dim=1).mean().item()
        attn_to_key_front = avg_q2c[:, target_key_front].sum(dim=1).mean().item() if target_key_front else 0
        attn_to_key_full = avg_q2c[:, target_key_full].sum(dim=1).mean().item() if target_key_full else 0
        attn_to_value = avg_q2c[:, target_value].sum(dim=1).mean().item() if target_value else 0
        attn_to_period = avg_q2c[:, target_period].sum(dim=1).mean().item() if target_period else 0
        attn_to_sentence = avg_q2c[:, target_sentence].sum(dim=1).mean().item() if target_sentence else 0

        # Average attention to a random distractor sentence
        if groups['distractor_sentences']:
            dist_sent = groups['distractor_sentences'][0]
            attn_to_dist = avg_q2c[:, dist_sent].sum(dim=1).mean().item()
        else:
            attn_to_dist = 0

        # Top-10 positions across all heads
        top_vals, top_idxs = avg_q2c.mean(dim=0).topk(10)
        top_tokens = [(idx.item(), tokens[idx.item()], f"{top_vals[i].item():.4f}") for i, idx in enumerate(top_idxs)]

        print(f"\n  Layer {l:2d}: Q→sink={attn_to_sink:.4f}  Q→key_front={attn_to_key_front:.4f}  "
              f"Q→key_full={attn_to_key_full:.4f}  Q→value={attn_to_value:.4f}  "
              f"Q→period={attn_to_period:.4f}  Q→sentence={attn_to_sentence:.4f}  "
              f"Q→distractor={attn_to_dist:.4f}")
        print(f"           Top tokens: {top_tokens[:5]}")

        # Per-head breakdown for this layer (find which heads are "retrieval heads")
        head_attn_to_sentence = avg_q2c[:, target_sentence].sum(dim=1)  # [n_kv_heads]
        best_heads = head_attn_to_sentence.topk(min(3, len(head_attn_to_sentence)))
        head_info = [(best_heads.indices[i].item(), f"{best_heads.values[i].item():.4f}")
                     for i in range(len(best_heads.indices))]
        print(f"           Best retrieval heads: {head_info}")

    print(f"\n{'='*100}")
    print(f"STEP 2: Within context - does period attend to needle front? (prefill attention)")
    print(f"{'='*100}")

    if not target_period or not target_key_front:
        print("  WARNING: Could not find period or key front tokens!")
        return

    period_pos = target_period[-1]  # last period token

    for l in analyze_layers:
        if l not in prefill_attn:
            continue
        pa = prefill_attn[l]  # [n_kv_heads, ctx_len, ctx_len]

        # Period → which tokens?
        period_attn = pa[:, period_pos, :]  # [n_kv_heads, ctx_len]

        attn_p2sink = period_attn[:, sink].sum(dim=1).mean().item()
        attn_p2key = period_attn[:, target_key_front].sum(dim=1).mean().item()
        attn_p2value = period_attn[:, target_value].sum(dim=1).mean().item() if target_value else 0
        attn_p2sentence = period_attn[:, target_sentence].sum(dim=1).mean().item()

        # Top-5 positions that period attends to
        top_vals, top_idxs = period_attn.mean(dim=0).topk(10)
        top_toks = [(idx.item(), tokens[idx.item()].strip(), f"{top_vals[i].item():.4f}")
                    for i, idx in enumerate(top_idxs)]

        print(f"\n  Layer {l:2d}: period→sink={attn_p2sink:.4f}  period→key={attn_p2key:.4f}  "
              f"period→value={attn_p2value:.4f}  period→sentence={attn_p2sentence:.4f}")
        print(f"           Period top targets: {top_toks[:5]}")

    print(f"\n{'='*100}")
    print(f"STEP 3: Does needle front ('mystic') attend to the value digits? (prefill attention)")
    print(f"{'='*100}")

    if not target_key_front:
        print("  WARNING: Could not find key front tokens!")
        return

    key_front_pos = target_key_front[0]  # first "mystic" token

    for l in analyze_layers:
        if l not in prefill_attn:
            continue
        pa = prefill_attn[l]

        # Note: causal attention - "mystic" comes BEFORE the value, so it CAN'T attend to value
        # But value CAN attend to "mystic"
        key_attn = pa[:, key_front_pos, :]  # [n_kv_heads, ctx_len]
        top_vals, top_idxs = key_attn.mean(dim=0).topk(10)
        top_toks = [(idx.item(), tokens[idx.item()].strip(), f"{top_vals[i].item():.4f}")
                    for i, idx in enumerate(top_idxs)]

        print(f"\n  Layer {l:2d}: 'mystic' token [{key_front_pos}] attends to: {top_toks[:5]}")

    print(f"\n{'='*100}")
    print(f"STEP 3b: Do value tokens attend back to 'mystic'? (value → key, causal OK)")
    print(f"{'='*100}")

    if target_value:
        val_pos = target_value[0]
        for l in analyze_layers:
            if l not in prefill_attn:
                continue
            pa = prefill_attn[l]
            val_attn = pa[:, val_pos, :]

            attn_v2key = val_attn[:, target_key_front].sum(dim=1).mean().item()
            attn_v2sink = val_attn[:, sink].sum(dim=1).mean().item()

            top_vals, top_idxs = val_attn.mean(dim=0).topk(10)
            top_toks = [(idx.item(), tokens[idx.item()].strip(), f"{top_vals[i].item():.4f}")
                        for i, idx in enumerate(top_idxs)]

            print(f"\n  Layer {l:2d}: value[{val_pos}]→key={attn_v2key:.4f}  "
                  f"value→sink={attn_v2sink:.4f}")
            print(f"           Value top targets: {top_toks[:5]}")

    print(f"\n{'='*100}")
    print(f"STEP 4: Graph expansion score analysis")
    print(f"{'='*100}")

    for l in analyze_layers:
        if l not in question_attn or l not in prefill_attn:
            continue

        qa = question_attn[l]  # [n_kv_heads, q_len, ctx_len]
        pa = prefill_attn[l]  # [n_kv_heads, ctx_len, ctx_len]

        # Seed = max over question positions
        seed = qa.amax(dim=1)  # [n_kv_heads, ctx_len]

        # Graph expansion
        outgoing = torch.bmm(seed.unsqueeze(1), pa).squeeze(1)  # seed * prefill
        incoming = torch.bmm(pa, seed.unsqueeze(2)).squeeze(2)  # prefill^T * seed

        combined = seed + 1.0 * (outgoing + incoming)

        # Score for target sentence vs others
        target_mask = torch.zeros(ctx_len, dtype=torch.bool)
        target_mask[target_sentence] = True
        other_mask = torch.ones(ctx_len, dtype=torch.bool)
        other_mask[:10] = False  # exclude sink
        other_mask[target_sentence] = False

        # Per-component analysis
        seed_target = seed[:, target_mask].mean().item()
        seed_other = seed[:, other_mask].mean().item()
        out_target = outgoing[:, target_mask].mean().item()
        out_other = outgoing[:, other_mask].mean().item()
        inc_target = incoming[:, target_mask].mean().item()
        inc_other = incoming[:, other_mask].mean().item()
        comb_target = combined[:, target_mask].mean().item()
        comb_other = combined[:, other_mask].mean().item()

        print(f"\n  Layer {l:2d}:")
        print(f"    Seed:     target={seed_target:.6f}  other={seed_other:.6f}  ratio={seed_target/max(seed_other,1e-10):.3f}x")
        print(f"    Outgoing: target={out_target:.6f}  other={out_other:.6f}  ratio={out_target/max(out_other,1e-10):.3f}x")
        print(f"    Incoming: target={inc_target:.6f}  other={inc_other:.6f}  ratio={inc_target/max(inc_other,1e-10):.3f}x")
        print(f"    Combined: target={comb_target:.6f}  other={comb_other:.6f}  ratio={comb_target/max(comb_other,1e-10):.3f}x")

        # Which component dominates?
        if comb_other > 0:
            seed_contrib = (seed_target - seed_other) / max(abs(comb_target - comb_other), 1e-10)
            out_contrib = (out_target - out_other) / max(abs(comb_target - comb_other), 1e-10)
            inc_contrib = (inc_target - inc_other) / max(abs(comb_target - comb_other), 1e-10)
            print(f"    Contribution: seed={seed_contrib:.2f}  outgoing={out_contrib:.2f}  incoming={inc_contrib:.2f}")

    print(f"\n{'='*100}")
    print(f"STEP 5: What does graph expansion actually boost? (Top boosted tokens)")
    print(f"{'='*100}")

    for l in [7, 15, 23, 31]:
        if l >= n_layers or l not in question_attn or l not in prefill_attn:
            continue

        qa = question_attn[l]
        pa = prefill_attn[l]
        seed = qa.amax(dim=1)
        outgoing = torch.bmm(seed.unsqueeze(1), pa).squeeze(1)
        incoming = torch.bmm(pa, seed.unsqueeze(2)).squeeze(2)
        expansion = outgoing + incoming  # just the expansion part

        # Average across heads
        exp_avg = expansion.mean(dim=0)  # [ctx_len]
        seed_avg = seed.mean(dim=0)

        # Top expansion-boosted tokens
        top_vals, top_idxs = exp_avg.topk(20)
        print(f"\n  Layer {l:2d} - Top expansion-boosted tokens:")
        for i, idx in enumerate(top_idxs):
            pos = idx.item()
            in_target = "TARGET" if pos in target_sentence else ""
            in_sink = "SINK" if pos in sink else ""
            print(f"    [{pos:4d}] '{tokens[pos].strip():20s}' expansion={top_vals[i].item():.6f}  "
                  f"seed={seed_avg[pos].item():.6f}  {in_target}{in_sink}")


def main():
    model_path = "Qwen/Qwen3-8B"
    target_key = "mystic-thunder"
    target_value = "7156842"
    n_dist = 30
    seed = 50

    print("=" * 100)
    print("ATTENTION CHAIN ANALYSIS")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    prompt = build_prompt(n_dist, target_key, target_value, 0.5, seed=seed)
    separator = "#" * (len(prompt) + 10)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt + separator}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    ctx_text, q_suffix = full_text.split(separator)
    ctx_ids = tokenizer.encode(ctx_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    q_ids = tokenizer.encode(q_suffix, return_tensors="pt", add_special_tokens=False).to(model.device)
    ctx_len = ctx_ids.shape[1]
    q_len = q_ids.shape[1]

    print(f"\n  Context: {ctx_len} tokens, Question: {q_len} tokens")

    # Find token groups
    groups, tokens = find_token_groups(tokenizer, ctx_ids, ctx_len, target_key, target_value)

    # Compute full attention
    print(f"\n  Computing attention matrices...")
    prefill_attn, question_attn = compute_full_attention(model, ctx_ids, q_ids)
    torch.cuda.empty_cache()

    n_layers = model.config.num_hidden_layers

    # Analyze the chain
    analyze_chain(prefill_attn, question_attn, groups, tokens, n_layers, ctx_len, q_len)

    print("\n\nDONE")


if __name__ == "__main__":
    main()
