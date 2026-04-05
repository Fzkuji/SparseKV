#!/usr/bin/env python3
"""3-hop query-aware attention tracing with inverse fan-in weighting.

Memory-efficient version: uses KV cache keys after prefill, only stores
Q projections at user input positions per layer. All heavy computation
happens post-prefill, one layer at a time.
"""

import torch
import random
import math
import json
import re
import gc
import time
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half
from kvpress import KVzipPress, SnapKVPress
from kvpress.attention_patch import patch_attention_functions
from transformers import pipeline as hf_pipeline

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


def compute_threehop_scores(model, tokenizer, context_ids, question_ids=None,
                            fanin_temp=1.0, chunk_size=1024):
    """Compute 3-hop attention tracing scores.

    Post-prefill approach:
    1. Prefill with hooks that capture user-input Q (with RoPE) per layer
    2. After prefill, use KV cache K to compute attention products per layer
    """
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_q_heads = model.config.num_attention_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    ctx_len = context_ids.shape[1]

    user_pos = find_user_input_positions(tokenizer, context_ids, ctx_len)
    user_pos_t = torch.tensor(user_pos, dtype=torch.long, device=model.device)
    n_user = len(user_pos)
    print(f"  User input: {n_user} tokens (pos {min(user_pos)}-{max(user_pos)})")

    final_scores = torch.zeros(n_layers, 1, n_kv_heads, ctx_len, dtype=torch.float32)

    # Phase 1: Prefill and capture Q at user_input positions (with RoPE)
    user_q_storage = {}  # layer -> [n_q_heads, n_user, head_dim] on CPU

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
                q = module.q_proj(hidden_states[:, user_pos_t, :])  # [1, n_user, hidden]
                q = q.view(bsz, n_user, n_q_heads, head_dim).transpose(1, 2)
                # [1, n_q_heads, n_user, head_dim]
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    cos_u = cos[:, user_pos_t, :]  # [1, n_user, head_dim]
                    sin_u = sin[:, user_pos_t, :]
                    q = (q * cos_u.unsqueeze(1)) + (rotate_half(q) * sin_u.unsqueeze(1))
                user_q_storage[layer_idx] = q[0].cpu()  # [n_q_heads, n_user, head_dim]
        return hook_fn

    hooks = []
    for layer in model.model.layers:
        h = layer.self_attn.register_forward_hook(
            make_hook(int(layer.self_attn.layer_idx)), with_kwargs=True)
        hooks.append(h)

    cache = DynamicCache()
    t0 = time.time()
    with torch.no_grad():
        model.model(input_ids=context_ids, past_key_values=cache)
    for h in hooks:
        h.remove()
    print(f"  Prefill: {time.time()-t0:.1f}s")

    # Phase 2: Compute 3-hop scores per layer using cached K
    t0 = time.time()
    scale = math.sqrt(head_dim)

    for li in range(n_layers):
        if li not in user_q_storage:
            continue

        # Get K from cache (already has RoPE applied)
        k_all = cache.layers[li].keys[0]  # [n_kv_heads, ctx_len, head_dim]
        user_q = user_q_storage[li].to(model.device)  # [n_q_heads, n_user, head_dim]

        # Group Q by KV head
        user_q_grouped = user_q.view(n_kv_heads, n_groups, n_user, head_dim)

        for hi in range(n_kv_heads):
            k_h = k_all[hi]  # [ctx_len, head_dim]

            # ═══ Step 1: OUTGOING from user input ═══
            uq = user_q_grouped[hi]  # [n_groups, n_user, head_dim]
            # Attention: [n_groups, n_user, ctx_len]
            user_attn_logits = torch.matmul(uq, k_h.T) / scale
            # Causal mask
            for ui in range(n_user):
                user_attn_logits[:, ui, user_pos[ui]+1:] = float('-inf')
            user_attn = torch.softmax(user_attn_logits.float(), dim=-1)
            step1 = user_attn.amax(dim=(0, 1))  # [ctx_len]
            del user_attn, user_attn_logits, uq

            # ═══ Fan-in approximation ═══
            # K norms × sqrt(position) as proxy for how universally attended a token is
            k_norms = k_h.float().norm(dim=-1)
            pos_scale = torch.arange(1, ctx_len + 1, device=k_h.device, dtype=torch.float32).sqrt()
            fan_in = k_norms * pos_scale
            inv_fanin = 1.0 / (fan_in + 1e-6).pow(fanin_temp)
            inv_fanin = inv_fanin / inv_fanin.sum()
            step1_weighted = step1 * inv_fanin

            # ═══ Step 2 + Step 3: need attention from ALL positions ═══
            # Process in chunks to avoid L×L matrix
            step2 = torch.zeros(ctx_len, dtype=torch.float32, device=k_h.device)
            step3 = torch.zeros(ctx_len, dtype=torch.float32, device=k_h.device)

            # We need Q for ALL positions (not just user input) for step 2/3
            # But we only stored user Q. Solution: use K as approximate Q
            # (self-attention has Q≈K due to shared representations)
            # Actually, better: use k_h as queries (K-to-K attention)
            # This is equivalent to asking "which keys attend to step1-weighted keys"

            for start in range(0, ctx_len, chunk_size):
                end = min(start + chunk_size, ctx_len)
                chunk_k = k_h[start:end]  # [chunk, d]
                # Use K as Q (approximate): attn[i,j] ≈ softmax(k_i · k_j / sqrt(d))
                logits = torch.matmul(chunk_k, k_h.T) / scale  # [chunk, ctx_len]
                # Causal mask
                for ci in range(end - start):
                    logits[ci, start + ci + 1:] = float('-inf')
                chunk_attn = torch.softmax(logits.float(), dim=-1)  # [chunk, ctx_len]

                # Step 2: each row's weighted sum with step1_weighted
                chunk_step2 = torch.matmul(chunk_attn, step1_weighted)  # [chunk]
                step2[start:end] = chunk_step2

                # Step 3: transpose contribution
                step3 += torch.matmul(chunk_attn.T, chunk_step2)  # [ctx_len]

                del chunk_k, logits, chunk_attn, chunk_step2

            # ═══ Combine ═══
            def norm01(x):
                mn, mx = x.min(), x.max()
                return (x - mn) / (mx - mn + 1e-10)

            combined = norm01(step1) + norm01(step2) + norm01(step3)
            final_scores[li, 0, hi, :] = combined.cpu()

            del step1, step2, step3, fan_in, inv_fanin, step1_weighted, k_norms, pos_scale

        del user_q, user_q_grouped, k_all
        user_q_storage.pop(li)

    print(f"  3-hop scoring: {time.time()-t0:.1f}s")
    return final_scores, cache


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


# ─── Compression ───

def apply_global_compression(model, score_val, cr, ctx_len):
    n_layers, bsz, n_kv_heads, seq_len = score_val.shape
    total = n_layers * bsz * n_kv_heads * seq_len
    n_pruned = int(total * cr)
    if n_pruned <= 0:
        for layer in model.model.layers:
            layer.self_attn.masked_key_indices = None
        return
    flat_scores = score_val.reshape(-1)
    _, prune_idx = torch.topk(-flat_scores, min(n_pruned, flat_scores.numel()))
    layer_size = bsz * n_kv_heads * seq_len
    for layer in model.model.layers:
        li = int(layer.self_attn.layer_idx)
        layer_start = li * layer_size
        layer_end = layer_start + layer_size
        layer_mask = (prune_idx >= layer_start) & (prune_idx < layer_end)
        layer_indices = prune_idx[layer_mask] - layer_start
        if len(layer_indices) == 0:
            layer.self_attn.masked_key_indices = None
            continue
        bi = layer_indices // (n_kv_heads * seq_len)
        remainder = layer_indices % (n_kv_heads * seq_len)
        hi = remainder // seq_len
        si = remainder % seq_len
        layer.self_attn.masked_key_indices = (bi, hi, si)


def apply_perhead_compression(model, score_val, cr, ctx_len):
    n_layers, bsz, n_kv_heads, seq_len = score_val.shape
    n_keep = max(1, int(seq_len * (1 - cr)))
    for layer in model.model.layers:
        module = layer.self_attn
        li = int(module.layer_idx)
        scores = score_val[li]
        _, top_idx = scores.topk(n_keep, dim=-1)
        keep_mask = torch.zeros(bsz, n_kv_heads, seq_len, dtype=torch.bool)
        keep_mask.scatter_(-1, top_idx, True)
        prune_mask = ~keep_mask
        indices = prune_mask.nonzero(as_tuple=False)
        if len(indices) == 0:
            module.masked_key_indices = None
            continue
        module.masked_key_indices = (indices[:, 0], indices[:, 1], indices[:, 2])


def clear_compression(model):
    for layer in model.model.layers:
        layer.self_attn.masked_key_indices = None


def generate_with_scores(model, tokenizer, ctx_ids, q_ids, score_val, cr,
                         allocation='global', max_new=60):
    ctx_len = ctx_ids.shape[1]
    q_len = q_ids.shape[1]
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ctx_ids, past_key_values=cache)
    if cr > 0:
        if allocation == 'global':
            apply_global_compression(model, score_val, cr, ctx_len)
        else:
            apply_perhead_compression(model, score_val, cr, ctx_len)
    pos = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=q_ids, past_key_values=cache, position_ids=pos, num_logits_to_keep=1)
    gen = [out.logits[0, -1].argmax()]
    eos = model.generation_config.eos_token_id
    if not isinstance(eos, list):
        eos = [eos]
    cp = ctx_len + q_len
    for i in range(max_new - 1):
        with torch.no_grad():
            out = model(input_ids=gen[-1].unsqueeze(0).unsqueeze(0),
                       past_key_values=cache,
                       position_ids=torch.tensor([[cp + i]], device=model.device))
        nxt = out.logits[0, -1].argmax()
        gen.append(nxt)
        if nxt.item() in eos:
            break
    text = tokenizer.decode(torch.stack(gen), skip_special_tokens=True)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    clear_compression(model)
    del cache
    return text


# ─── NIAH Test ───

def test_niah(model, tokenizer):
    target_key = "mystic-thunder"
    target_value = "7156842"

    print("\n" + "=" * 80)
    print("TEST 1: NIAH (Needle In A Haystack)")
    print("=" * 80)

    for n_dist in [30, 100]:
        print(f"\n{'─' * 70}")
        print(f"  {n_dist} distractors, needle at 50%")
        print(f"{'─' * 70}")

        prompt = build_niah_prompt(n_dist, target_key, target_value, 0.5, seed=50)
        separator = "#" * (len(prompt) + 10)
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + separator}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        ctx_text, q_suffix = full_text.split(separator)
        ctx_ids = tokenizer.encode(ctx_text, return_tensors="pt",
                                   add_special_tokens=False).to(model.device)
        q_ids = tokenizer.encode(q_suffix, return_tensors="pt",
                                 add_special_tokens=False).to(model.device)
        ctx_len = ctx_ids.shape[1]
        print(f"  Context: {ctx_len} tokens, Question: {q_ids.shape[1]} tokens")

        print(f"\n  Computing 3-hop scores...")
        scores, _ = compute_threehop_scores(model, tokenizer, ctx_ids, q_ids)
        torch.cuda.empty_cache()

        analyze_niah_scores(tokenizer, scores, ctx_ids, target_key, target_value)

        print(f"\n  Generation results:")
        for cr in [0.5, 0.7, 0.9]:
            if cr == 0.5:
                full = generate_with_scores(model, tokenizer, ctx_ids, q_ids,
                                           torch.ones_like(scores), 0.0)
                ok_full = target_value in full
                print(f"    Full KV:           {'OK' if ok_full else 'FAIL'}  {full[:80]}")
                torch.cuda.empty_cache()

            gen = generate_with_scores(model, tokenizer, ctx_ids, q_ids,
                                      scores, cr, allocation='global')
            ok = target_value in gen
            print(f"    3hop-global cr={cr}: {'OK' if ok else 'FAIL'}  {gen[:80]}")
            torch.cuda.empty_cache()

            gen = generate_with_scores(model, tokenizer, ctx_ids, q_ids,
                                      scores, cr, allocation='perhead')
            ok = target_value in gen
            print(f"    3hop-perh   cr={cr}: {'OK' if ok else 'FAIL'}  {gen[:80]}")
            torch.cuda.empty_cache()

        del scores
        gc.collect()
        torch.cuda.empty_cache()


def analyze_niah_scores(tokenizer, score_val, ctx_ids, target_key, target_value):
    ctx_len = ctx_ids.shape[1]
    full_text = tokenizer.decode(ctx_ids[0])
    target_str = f"for {target_key} is: {target_value}."
    char_pos = full_text.find(target_str)
    if char_pos < 0:
        print("  Cannot find target needle for analysis")
        return
    cum = ""
    target_start = target_end = None
    for i in range(ctx_len):
        cum += tokenizer.decode(ctx_ids[0, i])
        if target_start is None and len(cum) > char_pos:
            target_start = max(0, i - 2)
        if target_end is None and len(cum) > char_pos + len(target_str):
            target_end = i + 1
            break
    if target_start is None:
        return

    print(f"\n  Score analysis (target at pos {target_start}-{target_end}):")
    print(f"  {'Layer':>6} {'Target':>10} {'Other':>10} {'Ratio':>8}")
    n_layers = score_val.shape[0]
    for li in range(0, n_layers, 4):
        layer_scores = score_val[li, 0].mean(dim=0)
        target_score = layer_scores[target_start:target_end].mean().item()
        mask = torch.ones(ctx_len, dtype=torch.bool)
        mask[:4] = False
        mask[target_start:target_end] = False
        other_score = layer_scores[mask].mean().item()
        ratio = target_score / max(other_score, 1e-10)
        print(f"  L{li:>4}: {target_score:>10.4f} {other_score:>10.4f} {ratio:>7.3f}x")


# ─── HotpotQA Test ───

def test_hotpotqa(model, tokenizer, pipe, N=10):
    from datasets import load_dataset

    print("\n" + "=" * 80)
    print("TEST 2: LongBench HotpotQA")
    print("=" * 80)

    ds = load_dataset("Xnhyacinth/LongBench", "hotpotqa", split="test")
    samples = list(ds.select(range(min(N, len(ds)))))

    def f1_score(pred, ref):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            return 0.0
        prec = len(common) / len(pred_tokens) if pred_tokens else 0
        rec = len(common) / len(ref_tokens) if ref_tokens else 0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    results = {'3hop_global_07': [], '3hop_perh_07': [],
               'snapkv_07': [], 'kvzip_07': [], 'fullkv': []}

    for i, sample in enumerate(samples):
        prompt = f"{sample['context']}\n\nQuestion: {sample['question']}\nAnswer:"
        max_tokens = 128
        ans_raw = sample["answers"]
        if isinstance(ans_raw, str):
            try: ans_list = json.loads(ans_raw)
            except: ans_list = [ans_raw]
        else:
            ans_list = list(ans_raw)

        # Full KV
        try:
            r = pipe(prompt, max_new_tokens=max_tokens)
            pred = re.sub(r'<think>.*?</think>', '', r["answer"].strip(), flags=re.DOTALL).strip()
        except Exception as e:
            pred = f"ERROR: {e}"
        fk_f1 = max(f1_score(pred, a) for a in ans_list)
        results['fullkv'].append({'pred': pred, 'f1': fk_f1})
        print(f"  Sample {i}: FullKV f1={fk_f1:.3f}  {pred[:60]}")

        # SnapKV 0.7
        try:
            r = pipe(prompt, press=SnapKVPress(compression_ratio=0.7), max_new_tokens=max_tokens)
            pred = re.sub(r'<think>.*?</think>', '', r["answer"].strip(), flags=re.DOTALL).strip()
        except Exception as e:
            pred = f"ERROR: {e}"
        sk_f1 = max(f1_score(pred, a) for a in ans_list)
        results['snapkv_07'].append({'pred': pred, 'f1': sk_f1})
        print(f"           SnapKV0.7 f1={sk_f1:.3f}  {pred[:60]}")

        # KVzip 0.7
        try:
            r = pipe(prompt, press=KVzipPress(compression_ratio=0.7), max_new_tokens=max_tokens)
            pred = re.sub(r'<think>.*?</think>', '', r["answer"].strip(), flags=re.DOTALL).strip()
        except Exception as e:
            pred = f"ERROR: {e}"
        kz_f1 = max(f1_score(pred, a) for a in ans_list)
        results['kvzip_07'].append({'pred': pred, 'f1': kz_f1})
        print(f"           KVzip0.7  f1={kz_f1:.3f}  {pred[:60]}")

        # 3-hop
        separator = "#" * (len(prompt) + 10)
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + separator}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        ctx_text, q_suffix = full_text.split(separator)
        ctx_ids = tokenizer.encode(ctx_text, return_tensors="pt",
                                   add_special_tokens=False).to(model.device)
        q_ids = tokenizer.encode(q_suffix, return_tensors="pt",
                                 add_special_tokens=False).to(model.device)

        try:
            scores, _ = compute_threehop_scores(model, tokenizer, ctx_ids, q_ids,
                                                 chunk_size=2048)
            torch.cuda.empty_cache()

            gen = generate_with_scores(model, tokenizer, ctx_ids, q_ids,
                                      scores, 0.7, allocation='global', max_new=max_tokens)
            g_f1 = max(f1_score(gen, a) for a in ans_list)
            results['3hop_global_07'].append({'pred': gen, 'f1': g_f1})
            print(f"           3hop-G0.7 f1={g_f1:.3f}  {gen[:60]}")
            torch.cuda.empty_cache()

            gen = generate_with_scores(model, tokenizer, ctx_ids, q_ids,
                                      scores, 0.7, allocation='perhead', max_new=max_tokens)
            p_f1 = max(f1_score(gen, a) for a in ans_list)
            results['3hop_perh_07'].append({'pred': gen, 'f1': p_f1})
            print(f"           3hop-P0.7 f1={p_f1:.3f}  {gen[:60]}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"           3hop ERROR: {e}")
            results['3hop_global_07'].append({'pred': str(e), 'f1': 0.0})
            results['3hop_perh_07'].append({'pred': str(e), 'f1': 0.0})
            torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()
        print()

    print(f"\n{'=' * 60}")
    print("HOTPOTQA SUMMARY (avg F1)")
    print(f"{'=' * 60}")
    for name, res in results.items():
        avg = sum(r['f1'] for r in res) / len(res) if res else 0
        print(f"  {name:20s}: {avg:.4f}  ({len(res)} samples)")
    return results


# ─── Main ───

def main():
    model_path = "Qwen/Qwen3-8B"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    # Use same model for pipeline (saves GPU memory)
    print("Loading pipeline (shared model)...")
    pipe = hf_pipeline(
        "kv-press-text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )

    test_niah(model, tokenizer)
    gc.collect()
    torch.cuda.empty_cache()

    test_hotpotqa(model, tokenizer, pipe, N=10)
    print("\n\nDONE")


if __name__ == "__main__":
    main()
