#!/usr/bin/env python3
"""Detailed attention chain analysis for NIAH task - v2 with fixed token finding.

Expected chain (per user's hypothesis):
1. Question tokens → sink tokens + needle front ("mystic-thunder")
2. Period at end of needle → needle front ("mystic-thunder")
3. Period → whole needle sentence (including value digits)
→ Graph expansion should propagate importance to whole sentence.

We need to verify each step and find where the chain breaks.
"""

import torch
import random
import math
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
    """Find specific token positions using sliding window decode."""
    tokens = [tokenizer.decode(ctx_ids[0, i]) for i in range(ctx_len)]

    # Print all tokens for debugging
    print(f"\n  All tokens around target area:")

    # Find target needle by scanning decoded text with windows
    target_start = None
    target_end = None

    # First, find the approximate area containing target_key
    for i in range(ctx_len - 5):
        window = tokenizer.decode(ctx_ids[0, i:min(i+40, ctx_len)])
        if target_key in window and target_value in window:
            # Found the area. Now find "One of the special" before it
            for s in range(max(0, i-15), i+3):
                w2 = tokenizer.decode(ctx_ids[0, s:s+5])
                if "One" in w2 and "of" in tokenizer.decode(ctx_ids[0, s:s+8]):
                    target_start = s
                    break
            if target_start is None:
                # Try checking for newline + "One"
                for s in range(max(0, i-15), i+3):
                    if '\n' in tokens[s] or tokens[s].strip() == '':
                        w2 = tokenizer.decode(ctx_ids[0, s:s+8])
                        if "One" in w2:
                            target_start = s
                            break
            if target_start is None:
                target_start = max(0, i - 10)

            # Find the period at the end
            for e in range(i, min(ctx_len, i+40)):
                w3 = tokenizer.decode(ctx_ids[0, i:e+1])
                if target_value in w3 and '.' in tokenizer.decode(ctx_ids[0, e:e+1]):
                    target_end = e
                    break
            if target_end is None:
                for e in range(i+5, min(ctx_len, i+40)):
                    if '.' in tokens[e]:
                        target_end = e
                        break
            if target_end is None:
                target_end = min(ctx_len - 1, target_start + 30)
            break

    if target_start is None:
        print("  FAILED to find target needle!")
        return None, tokens

    # Print the target sentence tokens
    sent_text = tokenizer.decode(ctx_ids[0, target_start:target_end+1])
    print(f"\n  Target sentence found: '{sent_text}'")
    print(f"  Positions {target_start}-{target_end}:")
    for i in range(target_start, target_end + 1):
        print(f"    [{i:4d}] '{tokens[i]}'")

    # Now classify each token in the target sentence
    groups = {
        'sink': list(range(min(5, ctx_len))),
        'target_key_tokens': [],    # tokens that are part of "mystic-thunder"
        'target_value_tokens': [],  # tokens that are digit parts of "7156842"
        'target_colon': [],         # ":"
        'target_period': [],        # "." at end
        'target_is_token': [],      # "is" token
        'target_for_token': [],     # "for" token
        'target_sentence': list(range(target_start, target_end + 1)),
        'distractor_sentences': [],
    }

    # Identify token types within the sentence by building cumulative decode
    cumulative = ""
    key_started = False
    key_ended = False
    value_started = False

    for i in range(target_start, target_end + 1):
        tok = tokens[i]
        cumulative = tokenizer.decode(ctx_ids[0, target_start:i+1])

        # Check if this token is part of the key
        if target_key in cumulative and not key_ended:
            # The key is complete up to here. Mark preceding tokens as key.
            # Find where key starts in cumulative
            if not key_started:
                # Back-track to find which tokens form the key
                for j in range(target_start, i+1):
                    partial = tokenizer.decode(ctx_ids[0, j:i+1])
                    if target_key in partial or any(part in partial for part in target_key.split('-')):
                        if any(part in partial for part in ['mystic', 'myst']):
                            for k in range(j, i+1):
                                groups['target_key_tokens'].append(k)
                            key_started = True
                            key_ended = True
                            break

        # Check for value digits
        if target_value[:4] in tokenizer.decode(ctx_ids[0, max(target_start, i-3):i+1]):
            if any(c.isdigit() for c in tok) and not value_started:
                value_started = True
            if value_started and any(c.isdigit() for c in tok):
                groups['target_value_tokens'].append(i)

        # Check for special tokens
        if ':' in tok:
            groups['target_colon'].append(i)
        if tok.strip() == 'is' or tok.strip() == ' is':
            groups['target_is_token'].append(i)
        if tok.strip() == 'for' or tok.strip() == ' for':
            groups['target_for_token'].append(i)
        if '.' in tok and i >= target_end - 1:
            groups['target_period'].append(i)

    # Fallback: manually scan for key tokens using substring matching
    if not groups['target_key_tokens']:
        for i in range(target_start, target_end + 1):
            tok_lower = tokens[i].lower()
            if any(part in tok_lower for part in ['myst', 'ic', 'thunder', 'thund', '-th']):
                groups['target_key_tokens'].append(i)

    # Fallback: scan for value tokens
    if not groups['target_value_tokens']:
        for i in range(target_start, target_end + 1):
            tok = tokens[i].strip()
            if tok and all(c.isdigit() for c in tok):
                groups['target_value_tokens'].append(i)

    # Find a few distractor sentences for comparison
    dist_count = 0
    for i in range(ctx_len - 10):
        if i >= target_start - 5 and i <= target_end + 5:
            continue
        window = tokenizer.decode(ctx_ids[0, i:min(i+40, ctx_len)])
        if "One of the special" in window and target_key not in window:
            d_start = i
            for e in range(i + 5, min(ctx_len, i + 40)):
                if '.' in tokens[e] and e > i + 8:
                    groups['distractor_sentences'].append(list(range(d_start, e + 1)))
                    dist_count += 1
                    break
            if dist_count >= 3:
                break

    # Also find question tokens in the question (for reference)
    # Question is separate, but we need to know what "mystic-thunder" tokens look like there

    print(f"\n  Token groups:")
    print(f"    Sink:         {groups['sink']}")
    print(f"    Key tokens:   {groups['target_key_tokens']} = {[tokens[i] for i in groups['target_key_tokens']]}")
    print(f"    Value tokens: {groups['target_value_tokens']} = {[tokens[i] for i in groups['target_value_tokens']]}")
    print(f"    Colon:        {groups['target_colon']} = {[tokens[i] for i in groups['target_colon']]}")
    print(f"    Period:       {groups['target_period']} = {[tokens[i] for i in groups['target_period']]}")
    print(f"    Sentence:     {groups['target_sentence'][0]}-{groups['target_sentence'][-1]} ({len(groups['target_sentence'])} tokens)")
    print(f"    Distractors:  {len(groups['distractor_sentences'])} found")

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

    prefill_attn = {}
    question_attn = {}
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
                    # Max over query groups: [n_kv_heads, ctx_len, ctx_len]
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


def analyze_chain(prefill_attn, question_attn, groups, tokens, n_layers, ctx_len, q_len, q_tokens):
    """Analyze each step of the hypothesized attention chain."""

    key_tok = groups['target_key_tokens']
    val_tok = groups['target_value_tokens']
    period_tok = groups['target_period']
    sentence = groups['target_sentence']
    sink = groups['sink']

    if not key_tok:
        print("  ERROR: no key tokens found!")
        return
    if not period_tok:
        print("  ERROR: no period tokens found!")
        return

    analyze_layers = list(range(0, n_layers, 4))  # every 4th layer

    # =========================================================================
    # STEP 1: Question → Context
    # =========================================================================
    print(f"\n{'='*100}")
    print(f"STEP 1: Question → Context")
    print(f"  Question tokens: {q_tokens}")
    print(f"  Expected: question should attend to target key tokens {key_tok}")
    print(f"{'='*100}")

    # Find which layers have retrieval heads
    best_retrieval_layers = []
    for l in range(n_layers):
        if l not in question_attn:
            continue
        qa = question_attn[l]  # [n_kv_heads, q_len, ctx_len]
        # Total attention to target sentence
        attn_to_sent = qa[:, :, sentence].sum(dim=(1, 2)).max().item()
        best_retrieval_layers.append((l, attn_to_sent))

    best_retrieval_layers.sort(key=lambda x: -x[1])
    print(f"\n  Top 10 layers by Q→target_sentence attention:")
    for l, score in best_retrieval_layers[:10]:
        qa = question_attn[l]
        attn_to_key = qa[:, :, key_tok].sum(dim=(1, 2)).max().item() if key_tok else 0
        attn_to_val = qa[:, :, val_tok].sum(dim=(1, 2)).max().item() if val_tok else 0
        attn_to_sink = qa[:, :, sink].sum(dim=(1, 2)).max().item()
        # Which head and which question token?
        per_head_sent = qa[:, :, sentence].sum(dim=(1, 2))  # [n_kv_heads]
        best_head = per_head_sent.argmax().item()
        best_head_score = per_head_sent[best_head].item()
        print(f"    Layer {l:2d}: sent={score:.4f}  key={attn_to_key:.4f}  "
              f"val={attn_to_val:.4f}  sink={attn_to_sink:.4f}  "
              f"best_head=KV{best_head}({best_head_score:.4f})")

    # Detailed per-question-token analysis for best retrieval layer
    if best_retrieval_layers:
        best_l = best_retrieval_layers[0][0]
        qa = question_attn[best_l]
        print(f"\n  Best retrieval layer {best_l} - per question token:")
        for qi in range(q_len):
            per_head = qa[:, qi, :]  # [n_kv_heads, ctx_len]
            top_vals, top_idxs = per_head.mean(dim=0).topk(8)
            top_info = [(idx.item(), tokens[idx.item()].strip()[:15], f"{top_vals[j].item():.4f}")
                        for j, idx in enumerate(top_idxs)]
            in_sent = per_head[:, sentence].sum(dim=1).max().item()
            print(f"    Q[{qi}]='{q_tokens[qi].strip()[:10]:10s}': sent={in_sent:.4f}  top={top_info[:5]}")

    # =========================================================================
    # STEP 2: Period → whole needle sentence (prefill)
    # =========================================================================
    print(f"\n{'='*100}")
    print(f"STEP 2: Period [{period_tok}] → rest of needle sentence (prefill)")
    print(f"  Expected: period attends to key tokens and value tokens")
    print(f"{'='*100}")

    period_pos = period_tok[-1]
    for l in analyze_layers:
        if l not in prefill_attn:
            continue
        pa = prefill_attn[l]  # [n_kv_heads, ctx_len, ctx_len]
        period_attn = pa[:, period_pos, :]  # [n_kv_heads, ctx_len]

        attn_p2key = period_attn[:, key_tok].sum(dim=1).mean().item() if key_tok else 0
        attn_p2val = period_attn[:, val_tok].sum(dim=1).mean().item() if val_tok else 0
        attn_p2sink = period_attn[:, sink].sum(dim=1).mean().item()
        attn_p2sent = period_attn[:, sentence].sum(dim=1).mean().item()

        # Attention to nearby tokens (local attention pattern)
        local_start = max(0, period_pos - 5)
        local_end = period_pos
        attn_local = period_attn[:, local_start:local_end].sum(dim=1).mean().item()

        if l % 8 == 0 or l in [best_retrieval_layers[0][0]] if best_retrieval_layers else False:
            top_vals, top_idxs = period_attn.mean(dim=0).topk(8)
            top_info = [(idx.item(), tokens[idx.item()].strip()[:12], f"{top_vals[j].item():.4f}")
                        for j, idx in enumerate(top_idxs)]
            print(f"  L{l:2d}: p→key={attn_p2key:.4f} p→val={attn_p2val:.4f} p→sink={attn_p2sink:.4f} "
                  f"p→sent={attn_p2sent:.4f} p→local={attn_local:.4f}  top={top_info[:5]}")

    # =========================================================================
    # STEP 3: Value tokens → Key tokens (causal: value comes after key)
    # =========================================================================
    print(f"\n{'='*100}")
    print(f"STEP 3: Value tokens {val_tok} → Key tokens {key_tok} (prefill)")
    print(f"  Expected: value digits attend back to 'mystic-thunder'")
    print(f"{'='*100}")

    if val_tok:
        val_pos = val_tok[0]
        for l in analyze_layers:
            if l not in prefill_attn:
                continue
            pa = prefill_attn[l]
            val_attn = pa[:, val_pos, :]

            attn_v2key = val_attn[:, key_tok].sum(dim=1).mean().item()
            attn_v2sink = val_attn[:, sink].sum(dim=1).mean().item()

            if l % 8 == 0:
                top_vals, top_idxs = val_attn.mean(dim=0).topk(8)
                top_info = [(idx.item(), tokens[idx.item()].strip()[:12], f"{top_vals[j].item():.4f}")
                            for j, idx in enumerate(top_idxs)]
                print(f"  L{l:2d}: v→key={attn_v2key:.4f} v→sink={attn_v2sink:.4f}  top={top_info[:5]}")

    # =========================================================================
    # CRITICAL CHECK: Causal attention direction
    # =========================================================================
    print(f"\n{'='*100}")
    print(f"CRITICAL: Causal attention direction check")
    print(f"  Key tokens: {key_tok} (positions ~{key_tok[0] if key_tok else '?'})")
    print(f"  Value tokens: {val_tok} (positions ~{val_tok[0] if val_tok else '?'})")
    print(f"  Period: {period_tok} (position {period_pos})")
    print(f"{'='*100}")

    if key_tok and val_tok:
        print(f"\n  Token order in sentence: key({key_tok[0]}) < value({val_tok[0]}) < period({period_pos})")
        print(f"  Causal mask allows:")
        print(f"    period → key  YES (period is after key)")
        print(f"    period → value  YES (period is after value)")
        print(f"    value → key  YES (value is after key)")
        print(f"    key → value  NO! (key is before value, can't attend to future)")
        print(f"    key → period  NO! (key is before period)")
        print(f"")
        print(f"  For graph expansion:")
        print(f"    Seed = question → ctx attention (question can see all ctx tokens)")
        print(f"    Outgoing: seed @ prefill_attn = where do seed-tokens attend to?")
        print(f"      If seed highlights 'mystic', outgoing = where 'mystic' attends to")
        print(f"      But 'mystic' comes BEFORE value digits → can't attend to value!")
        print(f"    Incoming: prefill_attn^T @ seed = who attends to seed-tokens?")
        print(f"      If seed highlights 'mystic', incoming = who attends to 'mystic'")
        print(f"      = value tokens and period (they can see 'mystic')")

    # =========================================================================
    # STEP 4: Graph expansion decomposition
    # =========================================================================
    print(f"\n{'='*100}")
    print(f"STEP 4: Graph expansion score decomposition")
    print(f"{'='*100}")

    for l in range(n_layers):
        if l not in question_attn or l not in prefill_attn:
            continue

        qa = question_attn[l]
        pa = prefill_attn[l]
        seed = qa.amax(dim=1)  # [n_kv_heads, ctx_len]
        outgoing = torch.bmm(seed.unsqueeze(1), pa).squeeze(1)
        incoming = torch.bmm(pa, seed.unsqueeze(2)).squeeze(2)
        combined = seed + 1.0 * (outgoing + incoming)

        target_mask = torch.zeros(ctx_len, dtype=torch.bool)
        for idx in sentence:
            target_mask[idx] = True
        other_mask = torch.ones(ctx_len, dtype=torch.bool)
        other_mask[:10] = False
        for idx in sentence:
            other_mask[idx] = False

        s_t = seed[:, target_mask].mean().item()
        s_o = seed[:, other_mask].mean().item()
        o_t = outgoing[:, target_mask].mean().item()
        o_o = outgoing[:, other_mask].mean().item()
        i_t = incoming[:, target_mask].mean().item()
        i_o = incoming[:, other_mask].mean().item()
        c_t = combined[:, target_mask].mean().item()
        c_o = combined[:, other_mask].mean().item()

        if l % 4 == 0:
            print(f"  L{l:2d}: seed(T/O)={s_t:.5f}/{s_o:.5f}={s_t/max(s_o,1e-10):.2f}x  "
                  f"out={o_t:.5f}/{o_o:.5f}={o_t/max(o_o,1e-10):.2f}x  "
                  f"in={i_t:.5f}/{i_o:.5f}={i_t/max(i_o,1e-10):.2f}x  "
                  f"comb={c_t:.5f}/{c_o:.5f}={c_t/max(c_o,1e-10):.2f}x")

    # =========================================================================
    # STEP 5: What tokens get the HIGHEST expansion boost?
    # =========================================================================
    print(f"\n{'='*100}")
    print(f"STEP 5: Top expansion-boosted tokens vs top seed tokens")
    print(f"{'='*100}")

    # Pick the best retrieval layer
    for l_idx in range(min(5, len(best_retrieval_layers))):
        l = best_retrieval_layers[l_idx][0]
        if l not in question_attn or l not in prefill_attn:
            continue

        qa = question_attn[l]
        pa = prefill_attn[l]
        seed = qa.amax(dim=1).mean(dim=0)  # [ctx_len]
        outgoing = torch.bmm(qa.amax(dim=1).unsqueeze(1), pa).squeeze(1).mean(dim=0)
        incoming = torch.bmm(pa, qa.amax(dim=1).unsqueeze(2)).squeeze(2).mean(dim=0)
        expansion = outgoing + incoming

        print(f"\n  Layer {l} (retrieval layer rank {l_idx}):")

        # Top seed tokens
        top_s, top_si = seed.topk(15)
        print(f"    Top SEED tokens:")
        for i, idx in enumerate(top_si):
            pos = idx.item()
            label = "TARGET" if pos in sentence else ("SINK" if pos in sink else "")
            print(f"      [{pos:4d}] '{tokens[pos].strip()[:20]:20s}' seed={top_s[i].item():.5f} {label}")

        # Top expansion tokens
        top_e, top_ei = expansion.topk(15)
        print(f"    Top EXPANSION tokens:")
        for i, idx in enumerate(top_ei):
            pos = idx.item()
            label = "TARGET" if pos in sentence else ("SINK" if pos in sink else "")
            print(f"      [{pos:4d}] '{tokens[pos].strip()[:20]:20s}' exp={top_e[i].item():.5f} {label}")

        # Top combined tokens
        combined = seed + expansion
        top_c, top_ci = combined.topk(15)
        print(f"    Top COMBINED tokens:")
        for i, idx in enumerate(top_ci):
            pos = idx.item()
            label = "TARGET" if pos in sentence else ("SINK" if pos in sink else "")
            print(f"      [{pos:4d}] '{tokens[pos].strip()[:20]:20s}' comb={top_c[i].item():.5f} {label}")

    # =========================================================================
    # STEP 6: Fundamental problem diagnosis
    # =========================================================================
    print(f"\n{'='*100}")
    print(f"STEP 6: Fundamental problem diagnosis")
    print(f"{'='*100}")

    # Check: does the question even know which needle is the target?
    # Compare attention to target sentence vs each distractor
    print(f"\n  Question attention to target vs each distractor sentence:")
    for l in [best_retrieval_layers[i][0] for i in range(min(3, len(best_retrieval_layers)))]:
        if l not in question_attn:
            continue
        qa = question_attn[l]

        target_attn = qa[:, :, sentence].sum(dim=(1, 2)).mean().item()
        print(f"\n    Layer {l}: target_sentence_attn = {target_attn:.4f}")
        for di, dist in enumerate(groups['distractor_sentences'][:5]):
            dist_attn = qa[:, :, dist].sum(dim=(1, 2)).mean().item()
            ratio = target_attn / max(dist_attn, 1e-10)
            print(f"      distractor_{di} ({len(dist)} tok): {dist_attn:.4f}  target/dist ratio={ratio:.2f}x")

    # Check: where do "mystic"/"thunder" tokens in the QUESTION attend?
    # The question contains "mystic-thunder" - does it find the right one in context?
    print(f"\n  Question 'mystic-thunder' tokens attending to context:")
    for qi in range(q_len):
        tok = q_tokens[qi].lower()
        if any(part in tok for part in ['myst', 'thunder', 'thund']):
            for l in [best_retrieval_layers[i][0] for i in range(min(3, len(best_retrieval_layers)))]:
                if l not in question_attn:
                    continue
                qa = question_attn[l]
                q_attn = qa[:, qi, :]  # [n_kv_heads, ctx_len]
                top_vals, top_idxs = q_attn.mean(dim=0).topk(10)
                top_info = [(idx.item(), tokens[idx.item()].strip()[:15], f"{top_vals[j].item():.4f}")
                            for j, idx in enumerate(top_idxs)]
                attn_to_target = q_attn[:, sentence].sum(dim=1).mean().item()
                print(f"    Q[{qi}]='{q_tokens[qi].strip()[:12]}' L{l}: target_sent={attn_to_target:.4f}  top={top_info[:6]}")


def main():
    model_path = "Qwen/Qwen3-8B"
    target_key = "mystic-thunder"
    target_value = "7156842"
    n_dist = 30
    seed = 50

    print("=" * 100)
    print("ATTENTION CHAIN ANALYSIS v2")
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
    q_tokens = [tokenizer.decode(q_ids[0, i]) for i in range(q_len)]

    print(f"\n  Context: {ctx_len} tokens, Question: {q_len} tokens")
    print(f"  Question tokens: {q_tokens}")

    # Find token groups
    groups, tokens = find_token_groups(tokenizer, ctx_ids, ctx_len, target_key, target_value)
    if groups is None:
        print("FAILED: could not find target needle")
        return

    # Compute full attention
    print(f"\n  Computing attention matrices (this uses ~15GB VRAM)...")
    prefill_attn, question_attn = compute_full_attention(model, ctx_ids, q_ids)
    torch.cuda.empty_cache()

    n_layers = model.config.num_hidden_layers
    analyze_chain(prefill_attn, question_attn, groups, tokens, n_layers, ctx_len, q_len, q_tokens)

    print("\n\nDONE")


if __name__ == "__main__":
    main()
