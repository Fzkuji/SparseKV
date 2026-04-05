#!/usr/bin/env python3
"""Compare three views of token importance:
1. Ground truth: which tokens does the model actually attend to during generation?
2. KVzip: which tokens does reconstruction scoring keep?
3. Ours: which tokens does ctx-query attention tracing keep?

Goal: understand the gap and whether our approach can be adjusted.
"""

import torch
import random
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.attention_patch import patch_attention_functions

patch_attention_functions()

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


def find_positions(tokenizer, ctx_ids, ctx_len, target_key, target_value):
    tokens = [tokenizer.decode(ctx_ids[0, i]) for i in range(ctx_len)]
    full_text = tokenizer.decode(ctx_ids[0])

    # Build char→token mapping
    char_to_tok = []
    cum_len = 0
    for i in range(ctx_len):
        tok_text = tokenizer.decode(ctx_ids[0, :i+1])
        new_len = len(tok_text)
        for _ in range(new_len - cum_len):
            char_to_tok.append(i)
        cum_len = new_len

    # Find needle
    pattern = f"for {target_key} is: {target_value}."
    nc = full_text.find(pattern)
    one_c = full_text.rfind("One of the special", 0, nc)
    period_c = full_text.find(".", nc + len(pattern) - 2)

    ns = char_to_tok[one_c]
    ne = char_to_tok[min(period_c, len(char_to_tok)-1)]

    # Key and value positions
    kc = full_text.find(target_key, nc - 30)
    key_s = char_to_tok[kc]
    key_e = char_to_tok[kc + len(target_key) - 1]
    vc = full_text.find(target_value, nc)
    val_s = char_to_tok[vc]
    val_e = char_to_tok[vc + len(target_value) - 1]

    # Question text
    qc = full_text.rfind("What is the special")
    q_start = char_to_tok[qc] if qc >= 0 else ctx_len - 20

    key_pos = list(range(key_s, key_e + 1))
    val_pos = list(range(val_s, val_e + 1))
    needle_sent = list(range(ns, ne + 1))
    q_text = list(range(q_start, ctx_len))

    # Find distractor sentences
    distractors = []
    i = 0
    while i < ctx_len - 10 and len(distractors) < 30:
        if ns - 5 <= i <= ne + 5:
            i = ne + 6
            continue
        chunk = tokenizer.decode(ctx_ids[0, i:min(i+15, ctx_len)])
        if "One of the" in chunk:
            d_start = i
            for e in range(i + 8, min(ctx_len, i + 35)):
                if '.' in tokens[e]:
                    distractors.append(list(range(d_start, e + 1)))
                    i = e + 1
                    break
            else:
                i += 1
        else:
            i += 1

    sent_str = tokenizer.decode(ctx_ids[0, ns:ne+1])
    print(f"  Needle: '{sent_str.strip()}'")
    print(f"    Positions: {ns}-{ne}")
    print(f"    Key: {key_pos} = {[tokens[i] for i in key_pos]}")
    print(f"    Value: {val_pos} = {[tokens[i] for i in val_pos]}")
    print(f"  Question text: {q_start}-{ctx_len-1}")
    print(f"  Distractors: {len(distractors)}")

    return {
        'needle_sentence': needle_sent,
        'needle_key': key_pos,
        'needle_value': val_pos,
        'question_text': q_text,
        'sink': list(range(5)),
        'distractors': distractors,
    }, tokens


# =========================================================================
# 1. Ground truth: generation-time attention
# =========================================================================
def get_generation_attention(model, tokenizer, ctx_ids, q_ids, max_gen=30):
    """Generate answer and capture attention from EACH generated token back to context."""
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = ctx_ids.shape[1]
    q_len = q_ids.shape[1]

    # Storage: per generated token, per layer, attention to context
    gen_attn_to_ctx = []  # list of [n_layers, n_kv_heads, ctx_len] tensors
    capture = {'active': False, 'storage': {}}

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            if not capture['active']:
                return
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return
            bsz, seq_len, _ = hidden_states.shape
            if seq_len != 1:
                return  # only capture during autoregressive generation
            position_embeddings = kwargs.get("position_embeddings", None)
            past_kv = kwargs.get("past_key_values", None)
            if past_kv is None:
                return

            with torch.no_grad():
                k = past_kv.layers[layer_idx].keys[:, :, :ctx_len, :]  # only context keys
                q = module.q_proj(hidden_states).view(bsz, 1, n_q_heads, head_dim).transpose(1, 2)
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                q_g = q.view(bsz, n_kv_heads, n_groups, 1, head_dim)
                k_e = k.unsqueeze(2)
                # We need to compute attention over ALL keys (not just ctx), then extract ctx portion
                # But we're slicing k to ctx_len, so attention is only over context
                # This is an approximation - real attention also includes q_ids and prev gen tokens
                # But the context attention portion is what we care about
                full_k = past_kv.layers[layer_idx].keys
                full_len = full_k.shape[2]
                q_full = q.view(bsz, n_kv_heads, n_groups, 1, head_dim)
                k_full = full_k.unsqueeze(2)
                attn_full = torch.softmax(
                    torch.matmul(q_full, k_full.transpose(-2, -1)) / math.sqrt(head_dim),
                    dim=-1
                )
                # Extract just the context portion attention
                attn_ctx = attn_full[:, :, :, :, :ctx_len]  # [1, n_kv_heads, n_groups, 1, ctx_len]
                # Max over groups, squeeze
                attn_ctx = attn_ctx.amax(dim=2)[0, :, 0, :]  # [n_kv_heads, ctx_len]
                capture['storage'][layer_idx] = attn_ctx.float().cpu()
        return hook_fn

    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.self_attn.register_forward_hook(
            make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True))

    # Prefill context
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=cache)

    # Forward question tokens
    pos = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=q_ids, past_key_values=cache, position_ids=pos, num_logits_to_keep=1)

    # Generate and capture attention
    gen_tokens = []
    gen_texts = []
    eos = model.generation_config.eos_token_id
    if not isinstance(eos, list):
        eos = [eos]

    nxt = out.logits[0, -1].argmax()
    gen_tokens.append(nxt)
    cp = ctx_len + q_len

    for i in range(max_gen):
        capture['active'] = True
        capture['storage'] = {}
        with torch.no_grad():
            out = model(input_ids=gen_tokens[-1].unsqueeze(0).unsqueeze(0),
                       past_key_values=cache,
                       position_ids=torch.tensor([[cp + i]], device=model.device))
        capture['active'] = False

        # Store this step's attention
        if capture['storage']:
            step_attn = torch.stack([capture['storage'].get(l, torch.zeros(n_kv_heads, ctx_len))
                                    for l in range(n_layers)])  # [n_layers, n_kv_heads, ctx_len]
            gen_attn_to_ctx.append(step_attn)
            gen_texts.append(tokenizer.decode(gen_tokens[-1]))

        nxt = out.logits[0, -1].argmax()
        gen_tokens.append(nxt)
        if nxt.item() in eos:
            break

    for h in hooks:
        h.remove()

    full_gen = tokenizer.decode(torch.stack(gen_tokens), skip_special_tokens=True)
    del cache
    return gen_attn_to_ctx, gen_texts, full_gen


# =========================================================================
# 2. KVzip scores
# =========================================================================
def get_kvzip_scores(model, tokenizer, ctx_ids):
    press = KVzipPress(compression_ratio=0.5, layerwise=True)
    saved = {}
    orig = press.compress_post
    def patched(m):
        saved['score_val'] = press.score_val.clone()
        orig(m)
    press.compress_post = patched
    cache = DynamicCache()
    with torch.no_grad(), press(model):
        model.model(input_ids=ctx_ids, past_key_values=cache)
    return saved.get('score_val')


# =========================================================================
# 3. Ctx-query attention scores
# =========================================================================
def get_ctx_query_scores(model, ctx_ids, q_text_positions):
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = ctx_ids.shape[1]

    prefill_attn = {}

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
                q_g = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                k_e = k.unsqueeze(2)
                attn = torch.matmul(q_g, k_e.transpose(-2, -1)) / math.sqrt(head_dim)
                causal = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=attn.device), diagonal=1)
                attn = torch.softmax(attn + causal[None, None, None], dim=-1)
                prefill_attn[layer_idx] = attn.amax(dim=2)[0].float().cpu()
        return hook_fn

    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.self_attn.register_forward_hook(
            make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True))

    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=cache)

    for h in hooks:
        h.remove()
    del cache

    q_pos = torch.tensor(q_text_positions)
    scores = torch.zeros(n_layers, 1, n_kv_heads, ctx_len, dtype=torch.float32)
    for l in range(n_layers):
        if l in prefill_attn:
            scores[l, 0] = prefill_attn[l][:, q_pos, :].amax(dim=1)
    return scores


# =========================================================================
# Analysis
# =========================================================================
def analyze_and_compare(gen_attn, gen_texts, full_gen, kvzip_scores, ctx_query_scores,
                        groups, tokens, ctx_len, n_layers, target_value):
    ns = groups['needle_sentence']
    nk = groups['needle_key']
    nv = groups['needle_value']
    qt = groups['question_text']
    sink = groups['sink']
    distractors = groups['distractors']

    print(f"\n  Generated: '{full_gen[:80]}'")
    print(f"  Contains target value: {'YES' if target_value in full_gen else 'NO'}")

    # =====================================================================
    print(f"\n{'='*100}")
    print(f"PART 1: Ground truth - which tokens does the model look at during generation?")
    print(f"{'='*100}")

    # Aggregate attention across all generation steps
    if gen_attn:
        all_gen_attn = torch.stack(gen_attn)  # [n_steps, n_layers, n_kv_heads, ctx_len]
        # Max across steps: most important attention at any generation step
        max_gen_attn = all_gen_attn.amax(dim=0)  # [n_layers, n_kv_heads, ctx_len]
        # Also sum across steps
        sum_gen_attn = all_gen_attn.sum(dim=0)

        # Per-step analysis
        print(f"\n  Per generation step (avg over layers & heads):")
        for step_i, (step_attn, step_text) in enumerate(zip(gen_attn, gen_texts)):
            avg = step_attn.mean(dim=(0, 1))  # [ctx_len]
            attn_sink = avg[sink].sum().item()
            attn_needle = avg[ns].sum().item() if ns else 0
            attn_key = avg[nk].sum().item() if nk else 0
            attn_val = avg[nv].sum().item() if nv else 0
            attn_qt = avg[qt].sum().item() if qt else 0

            top_vals, top_idxs = avg.topk(8)
            top_info = [(idx.item(), tokens[idx.item()].strip()[:12], f"{top_vals[j].item():.4f}")
                        for j, idx in enumerate(top_idxs)]
            print(f"    Step {step_i:2d} '{step_text.strip()[:8]:8s}': "
                  f"sink={attn_sink:.4f} needle={attn_needle:.4f}(key={attn_key:.4f} val={attn_val:.4f}) "
                  f"q_text={attn_qt:.4f}  top={top_info[:5]}")

        # Aggregate: which context positions matter most?
        print(f"\n  Aggregate (max over steps, avg over heads):")
        agg = max_gen_attn.mean(dim=1)  # [n_layers, ctx_len] → avg over heads
        agg_all = agg.mean(dim=0)  # [ctx_len] avg over layers too

        # Top 30 most attended positions
        top_vals, top_idxs = agg_all.topk(30)
        print(f"  Top 30 most attended context tokens (during generation):")
        for i, idx in enumerate(top_idxs):
            pos = idx.item()
            label = ""
            if pos in nk: label = "NEEDLE_KEY"
            elif pos in nv: label = "NEEDLE_VAL"
            elif pos in ns: label = "NEEDLE"
            elif pos in qt: label = "QUESTION"
            elif pos in sink: label = "SINK"
            else:
                for di, d in enumerate(distractors):
                    if pos in d:
                        label = f"DIST_{di}"
                        break
            print(f"    [{pos:4d}] '{tokens[pos].strip()[:20]:20s}' attn={top_vals[i].item():.5f}  {label}")

        # Distribution: what fraction of attention goes where?
        print(f"\n  Attention budget (max-over-steps, avg-over-layers-heads):")
        total = agg_all.sum().item()
        print(f"    Sink:          {agg_all[sink].sum().item()/total*100:.1f}%")
        print(f"    Question text: {agg_all[qt].sum().item()/total*100:.1f}%")
        print(f"    Needle key:    {agg_all[nk].sum().item()/total*100:.1f}%")
        print(f"    Needle value:  {agg_all[nv].sum().item()/total*100:.1f}%")
        print(f"    Needle total:  {agg_all[ns].sum().item()/total*100:.1f}%")
        dist_total = sum(agg_all[d].sum().item() for d in distractors)
        print(f"    Distractors:   {dist_total/total*100:.1f}%")
        print(f"    Other:         {(total - agg_all[sink].sum().item() - agg_all[qt].sum().item() - agg_all[ns].sum().item() - dist_total)/total*100:.1f}%")

        # Per-layer: which layers focus on the needle?
        print(f"\n  Per-layer needle attention (max-over-steps):")
        for l in range(0, n_layers, 4):
            layer_avg = max_gen_attn[l].mean(dim=0)
            n_attn = layer_avg[ns].sum().item()
            d_attn = sum(layer_avg[d].sum().item() for d in distractors[:3]) / 3
            s_attn = layer_avg[sink].sum().item()
            ratio = n_attn / max(d_attn, 1e-10)
            print(f"    L{l:2d}: needle={n_attn:.5f}  avg_dist={d_attn:.5f}  sink={s_attn:.5f}  n/d={ratio:.2f}x")

    # =====================================================================
    print(f"\n{'='*100}")
    print(f"PART 2: KVzip score distribution")
    print(f"{'='*100}")

    if kvzip_scores is not None:
        kz = kvzip_scores.float()  # [n_layers, 1, n_kv_heads, ctx_len]
        kz_avg = kz.mean(dim=(0, 1, 2))  # [ctx_len]

        top_vals, top_idxs = kz_avg.topk(30)
        print(f"  Top 30 KVzip-scored tokens (avg over layers & heads):")
        for i, idx in enumerate(top_idxs):
            pos = idx.item()
            label = ""
            if pos in nk: label = "NEEDLE_KEY"
            elif pos in nv: label = "NEEDLE_VAL"
            elif pos in ns: label = "NEEDLE"
            elif pos in qt: label = "QUESTION"
            elif pos in sink: label = "SINK"
            print(f"    [{pos:4d}] '{tokens[pos].strip()[:20]:20s}' score={top_vals[i].item():.5f}  {label}")

        print(f"\n  KVzip score budget:")
        total = kz_avg.sum().item()
        print(f"    Sink:          {kz_avg[sink].sum().item()/total*100:.1f}%")
        print(f"    Question text: {kz_avg[qt].sum().item()/total*100:.1f}%")
        print(f"    Needle key:    {kz_avg[nk].sum().item()/total*100:.1f}%")
        print(f"    Needle value:  {kz_avg[nv].sum().item()/total*100:.1f}%")
        print(f"    Needle total:  {kz_avg[ns].sum().item()/total*100:.1f}%")
        dist_total = sum(kz_avg[d].sum().item() for d in distractors)
        print(f"    Distractors:   {dist_total/total*100:.1f}%")

    # =====================================================================
    print(f"\n{'='*100}")
    print(f"PART 3: Ctx-query attention score distribution")
    print(f"{'='*100}")

    cq = ctx_query_scores.float()
    cq_avg = cq.mean(dim=(0, 1, 2))

    top_vals, top_idxs = cq_avg.topk(30)
    print(f"  Top 30 ctx-query scored tokens:")
    for i, idx in enumerate(top_idxs):
        pos = idx.item()
        label = ""
        if pos in nk: label = "NEEDLE_KEY"
        elif pos in nv: label = "NEEDLE_VAL"
        elif pos in ns: label = "NEEDLE"
        elif pos in qt: label = "QUESTION"
        elif pos in sink: label = "SINK"
        print(f"    [{pos:4d}] '{tokens[pos].strip()[:20]:20s}' score={top_vals[i].item():.5f}  {label}")

    print(f"\n  Ctx-query score budget:")
    total = cq_avg.sum().item()
    print(f"    Sink:          {cq_avg[sink].sum().item()/total*100:.1f}%")
    print(f"    Question text: {cq_avg[qt].sum().item()/total*100:.1f}%")
    print(f"    Needle key:    {cq_avg[nk].sum().item()/total*100:.1f}%")
    print(f"    Needle value:  {cq_avg[nv].sum().item()/total*100:.1f}%")
    print(f"    Needle total:  {cq_avg[ns].sum().item()/total*100:.1f}%")
    dist_total = sum(cq_avg[d].sum().item() for d in distractors)
    print(f"    Distractors:   {dist_total/total*100:.1f}%")

    # =====================================================================
    print(f"\n{'='*100}")
    print(f"PART 4: Head-by-head comparison at key layers")
    print(f"{'='*100}")

    # For each layer, compare per-head scores
    n_kv_heads = kvzip_scores.shape[2] if kvzip_scores is not None else 8
    for l in [7, 15, 19, 23, 27, 31]:
        if l >= n_layers:
            continue
        print(f"\n  Layer {l}:")
        print(f"  {'Head':>6s}  {'GenAttn→needle':>16s}  {'KVzip→needle':>14s}  {'CtxQ→needle':>14s}  "
              f"{'GenAttn→dist':>14s}  {'KVzip→dist':>12s}  {'CtxQ→dist':>12s}")

        for h in range(n_kv_heads):
            # Gen attention
            if gen_attn:
                ga = max_gen_attn[l, h]
                ga_n = ga[ns].sum().item()
                ga_d = sum(ga[d].sum().item() for d in distractors[:5]) / 5
            else:
                ga_n, ga_d = 0, 0

            # KVzip
            if kvzip_scores is not None:
                kz_h = kvzip_scores[l, 0, h].float()
                kz_n = kz_h[ns].sum().item()
                kz_d = sum(kz_h[d].sum().item() for d in distractors[:5]) / 5
            else:
                kz_n, kz_d = 0, 0

            # Ctx-query
            cq_h = ctx_query_scores[l, 0, h].float()
            cq_n = cq_h[ns].sum().item()
            cq_d = sum(cq_h[d].sum().item() for d in distractors[:5]) / 5

            print(f"  KV{h:>3d}  {ga_n:>16.5f}  {kz_n:>14.5f}  {cq_n:>14.5f}  "
                  f"{ga_d:>14.5f}  {kz_d:>12.5f}  {cq_d:>12.5f}")

    # =====================================================================
    print(f"\n{'='*100}")
    print(f"PART 5: Overlap analysis - if we keep top K tokens, how many needle tokens are kept?")
    print(f"{'='*100}")

    for keep_frac in [0.1, 0.2, 0.3, 0.5]:
        n_keep = int(ctx_len * keep_frac)
        print(f"\n  Keeping top {keep_frac*100:.0f}% = {n_keep} tokens per head (avg over layers & heads):")

        methods = [("GenAttn", max_gen_attn if gen_attn else None),
                   ("KVzip", kvzip_scores[:, 0] if kvzip_scores is not None else None),
                   ("CtxQuery", ctx_query_scores[:, 0])]

        for name, scores_tensor in methods:
            if scores_tensor is None:
                continue

            needle_kept_total = 0
            value_kept_total = 0
            key_kept_total = 0
            count = 0

            for l in range(n_layers):
                for h in range(n_kv_heads):
                    s = scores_tensor[l, h].float()
                    _, top_k = s.topk(n_keep)
                    top_set = set(top_k.tolist())
                    needle_kept = len(top_set & set(ns))
                    value_kept = len(top_set & set(nv))
                    key_kept = len(top_set & set(nk))
                    needle_kept_total += needle_kept
                    value_kept_total += value_kept
                    key_kept_total += key_kept
                    count += 1

            avg_needle = needle_kept_total / count
            avg_value = value_kept_total / count
            avg_key = key_kept_total / count
            print(f"    {name:10s}: needle={avg_needle:.1f}/{len(ns)} "
                  f"(key={avg_key:.1f}/{len(nk)} val={avg_value:.1f}/{len(nv)})")


def main():
    model_path = "Qwen/Qwen3-8B"
    target_key = "mystic-thunder"
    target_value = "7156842"
    n_dist = 30
    seed = 50

    print("=" * 100)
    print("TOKEN IMPORTANCE COMPARISON: Ground Truth vs KVzip vs Ctx-Query")
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

    print(f"\n  Context: {ctx_len} tokens")

    groups, tokens = find_positions(tokenizer, ctx_ids, ctx_len, target_key, target_value)
    n_layers = model.config.num_hidden_layers

    # 1. Ground truth generation attention
    print(f"\n  [1/3] Getting ground truth generation attention...")
    gen_attn, gen_texts, full_gen = get_generation_attention(model, tokenizer, ctx_ids, q_ids)
    torch.cuda.empty_cache()

    # 2. KVzip scores
    print(f"  [2/3] Getting KVzip scores...")
    kvzip_scores = get_kvzip_scores(model, tokenizer, ctx_ids)
    torch.cuda.empty_cache()

    # 3. Ctx-query scores
    print(f"  [3/3] Getting ctx-query scores...")
    ctx_query_scores = get_ctx_query_scores(model, ctx_ids, groups['question_text'])
    torch.cuda.empty_cache()

    # Analysis
    analyze_and_compare(gen_attn, gen_texts, full_gen, kvzip_scores, ctx_query_scores,
                        groups, tokens, ctx_len, n_layers, target_value)

    print("\n\nDONE")


if __name__ == "__main__":
    main()
