#!/usr/bin/env python3
"""Attention chain analysis v3 - fixed token finding + correct question identification.

KEY INSIGHT: After chat template, the "question" tokens (q_ids) are just the
generation prompt: <|im_end|>\n<|im_start|>assistant\n<think>...
The ACTUAL question "What is the magic number for mystic-thunder..." is in the CONTEXT.

So the real chain is:
  gen_prompt → question_text_in_ctx → target_needle_in_ctx

And for prefill self-attention:
  question_text_in_ctx → target_key ("mystic-thunder") in ctx
  period (.) at end of target needle → whole needle sentence
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


def find_all_groups(tokenizer, ctx_ids, ctx_len, target_key, target_value):
    """Find token positions by decoding cumulative windows - robust approach."""
    tokens = [tokenizer.decode(ctx_ids[0, i]) for i in range(ctx_len)]

    # Step 1: Find EXACT position of target_key and target_value by scanning
    # Decode overlapping windows to find exact token boundaries
    key_positions = []
    value_positions = []
    question_positions = []

    # Scan for target_key substring
    for i in range(ctx_len):
        # Check if target_key starts around position i
        for width in range(3, 20):
            if i + width > ctx_len:
                break
            chunk = tokenizer.decode(ctx_ids[0, i:i+width])
            if target_key in chunk:
                # Found it. Now narrow down which tokens are the key
                # Try shrinking from left
                for left in range(i, i+width):
                    narrow = tokenizer.decode(ctx_ids[0, left:i+width])
                    if target_key in narrow:
                        # Try shrinking from right
                        for right in range(i+width, left, -1):
                            even_narrower = tokenizer.decode(ctx_ids[0, left:right])
                            if target_key in even_narrower:
                                # Check if this is within a needle (not the question)
                                # Look for "is:" after the key
                                after = tokenizer.decode(ctx_ids[0, right:min(right+10, ctx_len)])
                                before = tokenizer.decode(ctx_ids[0, max(0,left-15):left])
                                if "is:" in after or "is :" in after:
                                    # This is the needle occurrence
                                    key_positions = list(range(left, right))
                                elif "What" in before or "mentioned" in after:
                                    # This is the question occurrence
                                    question_positions = list(range(left, right))
                                break
                        break
                break

    # Scan for target_value
    for i in range(ctx_len):
        for width in range(2, 15):
            if i + width > ctx_len:
                break
            chunk = tokenizer.decode(ctx_ids[0, i:i+width])
            if target_value in chunk:
                for left in range(i, i+width):
                    narrow = tokenizer.decode(ctx_ids[0, left:i+width])
                    if target_value in narrow:
                        for right in range(i+width, left, -1):
                            even_narrower = tokenizer.decode(ctx_ids[0, left:right])
                            if target_value in even_narrower:
                                value_positions = list(range(left, right))
                                break
                        break
                break

    # Find the target needle sentence boundaries
    needle_sentence = []
    if key_positions:
        # Scan backward for "One" or newline before key
        sent_start = key_positions[0]
        for s in range(key_positions[0], max(0, key_positions[0]-20), -1):
            chunk = tokenizer.decode(ctx_ids[0, s:key_positions[0]+1])
            if "One of the" in chunk:
                sent_start = s
                break
            if s < key_positions[0] - 2 and '\n' in tokens[s]:
                sent_start = s + 1
                break

        # Scan forward for period after value
        sent_end = value_positions[-1] if value_positions else key_positions[-1] + 15
        for e in range(sent_end, min(ctx_len, sent_end + 5)):
            if '.' in tokens[e]:
                sent_end = e
                break
        needle_sentence = list(range(sent_start, sent_end + 1))

    # Find period token (end of needle sentence)
    period_positions = []
    if needle_sentence:
        for i in needle_sentence[-3:]:
            if '.' in tokens[i]:
                period_positions.append(i)

    # Find question text positions ("What is the special magic number for mystic-thunder...")
    # This is at the END of the context, before the generation prompt
    q_text_start = None
    q_text_end = None
    for i in range(max(0, ctx_len - 40), ctx_len):
        chunk = tokenizer.decode(ctx_ids[0, i:min(i+10, ctx_len)])
        if "What" in chunk and q_text_start is None:
            q_text_start = i
        if q_text_start is not None and '?' in chunk:
            q_text_end = min(i + 5, ctx_len - 1)
            # Narrow to find exact ?
            for j in range(i, min(i+6, ctx_len)):
                if '?' in tokens[j]:
                    q_text_end = j
                    break
            break
    q_text_positions = list(range(q_text_start, q_text_end + 1)) if q_text_start else []

    # Find distractor sentences
    distractor_sentences = []
    distractor_key_positions = []
    i = 0
    while i < ctx_len - 10 and len(distractor_sentences) < 5:
        # Skip target needle area
        if key_positions and abs(i - key_positions[0]) < 25:
            i += 25
            continue
        chunk = tokenizer.decode(ctx_ids[0, i:min(i+15, ctx_len)])
        if "One of the" in chunk:
            d_start = i
            d_end = None
            for e in range(i+8, min(ctx_len, i+35)):
                if '.' in tokens[e]:
                    d_end = e
                    break
            if d_end:
                distractor_sentences.append(list(range(d_start, d_end + 1)))
                # Find the key tokens in this distractor
                for k in range(d_start, d_end):
                    if 'for' in tokens[k].lower():
                        dk_start = k + 1
                        for dk in range(dk_start, min(dk_start+5, d_end)):
                            if 'is' in tokens[dk].lower() or ':' in tokens[dk]:
                                distractor_key_positions.append(list(range(dk_start, dk)))
                                break
                        break
                i = d_end + 1
            else:
                i += 1
        else:
            i += 1

    # Print results
    print(f"\n  === TOKEN GROUPS ===")
    if needle_sentence:
        sent_text = tokenizer.decode(ctx_ids[0, needle_sentence[0]:needle_sentence[-1]+1])
        print(f"  Target needle: '{sent_text.strip()}'")
        print(f"    Positions: {needle_sentence[0]}-{needle_sentence[-1]}")
        for i in needle_sentence:
            role = ""
            if i in key_positions: role = " <-- KEY"
            if i in value_positions: role = " <-- VALUE"
            if i in period_positions: role = " <-- PERIOD"
            print(f"    [{i:4d}] '{tokens[i]}'{role}")

    print(f"\n  Key tokens:    {key_positions} = {[tokens[i] for i in key_positions]}")
    print(f"  Value tokens:  {value_positions} = {[tokens[i] for i in value_positions]}")
    print(f"  Period tokens: {period_positions} = {[tokens[i] for i in period_positions]}")

    if q_text_positions:
        q_text = tokenizer.decode(ctx_ids[0, q_text_positions[0]:q_text_positions[-1]+1])
        print(f"\n  Question text in context: '{q_text.strip()}'")
        print(f"    Positions: {q_text_positions[0]}-{q_text_positions[-1]}")
        # Find mystic-thunder in question text
        q_key_positions = []
        for i in q_text_positions:
            if any(part in tokens[i].lower() for part in ['myst', 'thunder', 'thund', '-th']):
                q_key_positions.append(i)
        print(f"    Key ref in question: {q_key_positions} = {[tokens[i] for i in q_key_positions]}")
    else:
        q_key_positions = []

    print(f"  Distractor sentences: {len(distractor_sentences)} found")

    groups = {
        'sink': list(range(min(5, ctx_len))),
        'needle_key': key_positions,
        'needle_value': value_positions,
        'needle_period': period_positions,
        'needle_sentence': needle_sentence,
        'question_text': q_text_positions,
        'question_key_ref': q_key_positions if q_text_positions else [],
        'distractor_sentences': distractor_sentences,
        'distractor_keys': distractor_key_positions,
    }
    return groups, tokens


def compute_attention(model, context_ids, question_ids):
    """Compute attention matrices."""
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


def analyze(prefill_attn, question_attn, groups, tokens, n_layers, ctx_len, q_len, q_tokens):
    """Full chain analysis."""
    nk = groups['needle_key']
    nv = groups['needle_value']
    np_ = groups['needle_period']
    ns = groups['needle_sentence']
    qt = groups['question_text']
    qk = groups['question_key_ref']
    sink = groups['sink']

    # =========================================================================
    print(f"\n{'='*100}")
    print(f"ANALYSIS A: Gen-prompt → Context (what do gen tokens attend to?)")
    print(f"  Gen tokens: {q_tokens}")
    print(f"{'='*100}")

    for l in range(0, n_layers, 4):
        if l not in question_attn:
            continue
        qa = question_attn[l]  # [n_kv_heads, q_len, ctx_len]
        avg = qa.mean(dim=(0, 1))  # [ctx_len]

        attn_sink = avg[sink].sum().item()
        attn_qt = avg[qt].sum().item() if qt else 0
        attn_ns = avg[ns].sum().item() if ns else 0

        top_vals, top_idxs = avg.topk(5)
        top_info = [(idx.item(), tokens[idx.item()].strip()[:12], f"{top_vals[j].item():.4f}")
                    for j, idx in enumerate(top_idxs)]
        print(f"  L{l:2d}: sink={attn_sink:.4f}  q_text={attn_qt:.4f}  needle={attn_ns:.6f}  top={top_info}")

    # =========================================================================
    print(f"\n{'='*100}")
    print(f"ANALYSIS B: Question text in ctx → Needle (prefill self-attention)")
    print(f"  Question text positions: {qt[0] if qt else '?'}-{qt[-1] if qt else '?'}")
    print(f"  Key 'mystic-thunder' in question: {qk}")
    print(f"{'='*100}")

    if qt and ns:
        # For each question text token, where does it attend?
        print(f"\n  B1: All question_text tokens → context (avg over heads):")
        best_layers_qt = []
        for l in range(n_layers):
            if l not in prefill_attn:
                continue
            pa = prefill_attn[l]  # [n_kv_heads, ctx_len, ctx_len]
            # Average attention from question_text positions to needle_sentence
            qt_to_ns = pa[:, qt, :][:, :, ns].mean().item()
            # Compare to qt→distractors
            qt_to_dist = 0
            for ds in groups['distractor_sentences'][:3]:
                qt_to_dist += pa[:, qt, :][:, :, ds].mean().item()
            qt_to_dist /= max(1, len(groups['distractor_sentences'][:3]))
            best_layers_qt.append((l, qt_to_ns, qt_to_dist))

        best_layers_qt.sort(key=lambda x: -x[1])
        print(f"  Top 10 layers for Q_text → Needle:")
        for l, ns_score, dist_score in best_layers_qt[:10]:
            ratio = ns_score / max(dist_score, 1e-10)
            print(f"    L{l:2d}: Q→needle={ns_score:.6f}  Q→distractor={dist_score:.6f}  ratio={ratio:.2f}x")

        # B2: Specifically, "mystic-thunder" in question → "mystic-thunder" in needle
        if qk and nk:
            print(f"\n  B2: 'mystic-thunder' in question ({qk}) → 'mystic-thunder' in needle ({nk}):")
            for l in range(0, n_layers, 4):
                if l not in prefill_attn:
                    continue
                pa = prefill_attn[l]
                qk_to_nk = pa[:, qk, :][:, :, nk].mean().item()
                # Also check: question key ref → all other occurrences of similar tokens
                print(f"    L{l:2d}: q_key→n_key = {qk_to_nk:.6f}")

        # B3: "What" token → where?
        if qt:
            what_pos = qt[0]  # "What" is typically the first question token
            print(f"\n  B3: 'What' token [{what_pos}]='{tokens[what_pos].strip()}' → top targets:")
            for l in [best_layers_qt[i][0] for i in range(min(3, len(best_layers_qt)))]:
                if l not in prefill_attn:
                    continue
                pa = prefill_attn[l]
                what_attn = pa[:, what_pos, :].mean(dim=0)  # [ctx_len]
                top_vals, top_idxs = what_attn.topk(10)
                top_info = [(idx.item(), tokens[idx.item()].strip()[:15], f"{top_vals[j].item():.4f}")
                            for j, idx in enumerate(top_idxs)]
                in_needle = what_attn[ns].sum().item()
                print(f"      L{l:2d}: needle={in_needle:.4f}  top={top_info[:6]}")

    # =========================================================================
    print(f"\n{'='*100}")
    print(f"ANALYSIS C: Period → rest of needle sentence (prefill)")
    print(f"  Period at {np_}")
    print(f"{'='*100}")

    if np_ and ns:
        pp = np_[-1]
        print(f"\n  C1: Period [{pp}] attention distribution:")
        for l in range(0, n_layers, 4):
            if l not in prefill_attn:
                continue
            pa = prefill_attn[l]
            p_attn = pa[:, pp, :].mean(dim=0)  # avg over heads

            p_to_key = p_attn[nk].sum().item() if nk else 0
            p_to_val = p_attn[nv].sum().item() if nv else 0
            p_to_sink = p_attn[sink].sum().item()
            p_to_ns = p_attn[ns].sum().item()
            # Local (5 tokens before period)
            local = list(range(max(0, pp-5), pp))
            p_to_local = p_attn[local].sum().item()

            top_vals, top_idxs = p_attn.topk(8)
            top_info = [(idx.item(), tokens[idx.item()].strip()[:12], f"{top_vals[j].item():.4f}")
                        for j, idx in enumerate(top_idxs)]
            print(f"    L{l:2d}: p→key={p_to_key:.4f} p→val={p_to_val:.4f} p→sink={p_to_sink:.4f} "
                  f"p→sent={p_to_ns:.4f} p→local={p_to_local:.4f}  top={top_info[:5]}")

    # =========================================================================
    print(f"\n{'='*100}")
    print(f"ANALYSIS D: Value digits → Key tokens (prefill, causal OK)")
    print(f"{'='*100}")

    if nv and nk:
        for vp in nv[:2]:  # first 2 value digit tokens
            print(f"\n  D1: Value [{vp}]='{tokens[vp].strip()}' → Key {nk}:")
            for l in range(0, n_layers, 8):
                if l not in prefill_attn:
                    continue
                pa = prefill_attn[l]
                v_attn = pa[:, vp, :].mean(dim=0)
                v_to_key = v_attn[nk].sum().item()
                v_to_sink = v_attn[sink].sum().item()
                top_vals, top_idxs = v_attn.topk(5)
                top_info = [(idx.item(), tokens[idx.item()].strip()[:12], f"{top_vals[j].item():.4f}")
                            for j, idx in enumerate(top_idxs)]
                print(f"      L{l:2d}: v→key={v_to_key:.4f} v→sink={v_to_sink:.4f}  top={top_info}")

    # =========================================================================
    print(f"\n{'='*100}")
    print(f"ANALYSIS E: Causal direction problem for graph expansion")
    print(f"{'='*100}")

    if nk and nv and np_:
        print(f"  Token order: key({nk[0]}) → value({nv[0]}) → period({np_[-1]})")
        print(f"  Causal mask:")
        print(f"    key → value: BLOCKED (key comes first)")
        print(f"    key → period: BLOCKED")
        print(f"    value → key: OK")
        print(f"    period → key: OK")
        print(f"    period → value: OK")
        print(f"")
        print(f"  Graph expansion implications:")
        print(f"    If seed (gen→ctx) highlights 'mystic' key tokens:")
        print(f"      OUTGOING (seed @ prefill): where 'mystic' attends = tokens BEFORE it (causal)")
        print(f"        → 'mystic' can only see 'One of the special magic numbers for' and sink")
        print(f"        → CANNOT reach value digits or period!")
        print(f"      INCOMING (prefill^T @ seed): who attends TO 'mystic' = tokens AFTER it")
        print(f"        → value, period, and all later tokens can attend to 'mystic'")
        print(f"        → This gives high incoming scores to 'mystic' itself, not to value")
        print(f"")
        print(f"  *** This means outgoing expansion from key tokens CANNOT reach value tokens! ***")
        print(f"  *** The 'mystic→value' information flow is through HIDDEN STATES, not attention ***")

    # =========================================================================
    print(f"\n{'='*100}")
    print(f"ANALYSIS F: Score comparison - tracing seed vs ideal")
    print(f"{'='*100}")

    # Compute per-layer tracing scores and compare needle vs others
    for l in range(n_layers):
        if l not in question_attn:
            continue
        qa = question_attn[l]
        seed = qa.amax(dim=1)  # [n_kv_heads, ctx_len]

        if l % 4 != 0:
            continue

        # Avg over heads
        avg_seed = seed.mean(dim=0)  # [ctx_len]

        ns_score = avg_seed[ns].mean().item() if ns else 0
        # Compute scores for different parts of needle
        nk_score = avg_seed[nk].mean().item() if nk else 0
        nv_score = avg_seed[nv].mean().item() if nv else 0
        np_score = avg_seed[np_].mean().item() if np_ else 0

        # Other (non-sink, non-needle)
        other_mask = torch.ones(ctx_len, dtype=torch.bool)
        other_mask[:10] = False
        for idx in ns:
            other_mask[idx] = False
        for idx in qt:
            other_mask[idx] = False
        other_score = avg_seed[other_mask].mean().item()

        # Question text
        qt_score = avg_seed[qt].mean().item() if qt else 0
        sink_score = avg_seed[sink].mean().item()

        print(f"  L{l:2d}: sink={sink_score:.5f}  q_text={qt_score:.5f}  "
              f"needle={ns_score:.5f}(key={nk_score:.5f} val={nv_score:.5f} period={np_score:.5f})  "
              f"other={other_score:.5f}  ratio={ns_score/max(other_score,1e-10):.2f}x")

    # =========================================================================
    print(f"\n{'='*100}")
    print(f"ANALYSIS G: Which heads are retrieval heads? (per-head question→needle)")
    print(f"{'='*100}")

    n_kv_heads = question_attn[0].shape[0] if 0 in question_attn else 8
    for l in range(n_layers):
        if l not in question_attn:
            continue
        qa = question_attn[l]
        per_head_ns = qa[:, :, ns].sum(dim=(1, 2)) if ns else torch.zeros(n_kv_heads)
        max_head = per_head_ns.argmax().item()
        max_val = per_head_ns[max_head].item()
        if max_val > 0.3:  # Only show significant retrieval heads
            # What does this head attend to specifically?
            head_attn = qa[max_head, :, :].mean(dim=0)  # avg over q positions
            top_vals, top_idxs = head_attn.topk(10)
            top_info = [(idx.item(), tokens[idx.item()].strip()[:12], f"{top_vals[j].item():.4f}")
                        for j, idx in enumerate(top_idxs)]
            in_key = head_attn[nk].sum().item() if nk else 0
            in_val = head_attn[nv].sum().item() if nv else 0
            print(f"  L{l:2d} KV{max_head}: total_ns={max_val:.4f}  key={in_key:.4f}  val={in_val:.4f}  top={top_info[:6]}")

    # =========================================================================
    print(f"\n{'='*100}")
    print(f"SUMMARY: Why attention tracing fails")
    print(f"{'='*100}")

    print(f"""
  1. Gen-prompt tokens are generic (<|im_start|>assistant<think>...</think>).
     They have NO semantic content about "mystic-thunder".
     → Gen→ctx attention is dominated by sink tokens and template tokens.

  2. The actual question "What is ... mystic-thunder ..." is in the CONTEXT.
     The model uses HIDDEN STATES (not attention patterns) from the question text
     to condition the generation prompt's behavior.
     → Attention tracing from gen tokens misses the query semantics entirely.

  3. Even if we could use question_text→needle attention (prefill):
     Causal mask prevents key tokens ("mystic") from attending to value tokens ("7156842").
     Graph expansion OUTGOING from key tokens only reaches earlier tokens.
     INCOMING expansion only tells us who attends to key tokens (useful but indirect).

  4. The model retrieves information through HIDDEN STATE propagation:
     question_text hidden states encode "find mystic-thunder" →
     gen tokens' hidden states inherit this through residual stream →
     gen tokens' KV interactions activate retrieval heads →
     But this is NOT visible in Q→K attention patterns from gen tokens.
""")


def main():
    model_path = "Qwen/Qwen3-8B"
    target_key = "mystic-thunder"
    target_value = "7156842"
    n_dist = 30
    seed = 50

    print("=" * 100)
    print("ATTENTION CHAIN ANALYSIS v3")
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
    print(f"  Question (gen prompt) tokens: {q_tokens}")

    groups, tokens = find_all_groups(tokenizer, ctx_ids, ctx_len, target_key, target_value)

    print(f"\n  Computing attention matrices...")
    prefill_attn, question_attn = compute_attention(model, ctx_ids, q_ids)
    torch.cuda.empty_cache()

    n_layers = model.config.num_hidden_layers
    analyze(prefill_attn, question_attn, groups, tokens, n_layers, ctx_len, q_len, q_tokens)

    print("\n\nDONE")


if __name__ == "__main__":
    main()
