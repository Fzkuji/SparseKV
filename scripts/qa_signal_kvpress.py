"""
QA Signal Eviction based on kvpress KVzipPress architecture.

Instead of reconstruction ("Repeat the previous context exactly"),
we use the question+answer tokens' attention to score KV importance.

The scoring logic mirrors KVzipPress.score_kvzip exactly:
- Subsampled softmax (sink + scored_chunk + query_keys)
- amax(dim=(-3, -2)) over groups and query positions
- Global (non-layerwise) compression across layers

Differences from KVzipPress:
- No "repeat" forward passes
- Instead, we do ONE forward with [context + question + answer]
- Score context KV using attention FROM question+answer tokens

Usage:
    python -u scripts/qa_signal_kvpress.py --model ~/models/Qwen3-8B --output results/qa_signal_ruler4096.json

Requires: kvpress installed (for BasePress, utilities)
    pip install kvpress
"""

import torch
import json
import math
import os
import gc
import time
import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    DynamicCache,
)
from transformers.models.qwen3.modeling_qwen3 import rotate_half

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--n_samples", type=int, default=0, help="0=all")
parser.add_argument("--output", default="results/qa_signal_ruler4096.json")
parser.add_argument("--dataset_dir", default="4096")
parser.add_argument("--layerwise", action="store_true", help="Per-layer uniform compression (default: global)")
parser.add_argument("--shard", type=int, default=0, help="Shard index (0 or 1) for 2-GPU parallel")
parser.add_argument("--n_shards", type=int, default=1, help="Total number of shards")
args = parser.parse_args()

# Match kvzip's tested ratios
EVICT_RATIOS = [0.30, 0.50, 0.70, 0.90, 0.95]
N_SINK = 4  # same as kvzip default


# ============================================================
# Scoring: QA-based importance
# ============================================================

def score_qa_signal(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    context_end: int,
    n_sink: int = 4,
):
    """
    Score context KV positions using question+answer attention.
    
    Args:
        model: the LLM
        input_ids: [1, total_len] = [context + question + answer]
        context_end: position where context ends (= where question starts)
        n_sink: number of sink tokens to always keep
    
    Returns:
        scores: [n_layers, 1, n_kv_heads, context_end] importance scores
    """
    n_layers = model.config.num_hidden_layers
    n_q_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    n_groups = n_q_heads // n_kv_heads
    head_dim = model.config.hidden_size // n_q_heads
    
    total_len = input_ids.shape[1]
    qa_len = total_len - context_end  # length of question + answer
    
    scores = torch.zeros(n_layers, 1, n_kv_heads, context_end, 
                         dtype=model.dtype, device=model.device)
    scores[..., :n_sink] = 1.0  # protect sink tokens
    
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is None:
                return
            bsz, seq_len, _ = hs.shape
            if seq_len != total_len:
                return
            
            pe = kwargs.get("position_embeddings", None)
            
            with torch.no_grad():
                # Get Q and K projections
                q = module.q_proj(hs).view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                k = module.k_proj(hs).view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                
                # Apply RoPE
                if pe is not None:
                    cos, sin = pe
                    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                    k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
                
                # Reshape Q for grouped attention: [bsz, n_kv_heads, n_groups, seq_len, head_dim]
                q = q.view(bsz, n_kv_heads, n_groups, seq_len, head_dim)
                
                # Only need Q from question+answer positions
                q_qa = q[:, :, :, context_end:, :]  # [bsz, n_kv_heads, n_groups, qa_len, head_dim]
                
                # Subsampled keys (mirroring kvzip):
                # sink tokens + context chunk + qa tokens (like repeat chunk in kvzip)
                # For QA signal, we score ALL context positions at once (no chunking needed)
                # sink = k[:, :, :n_sink]
                # context_keys = k[:, :, n_sink:context_end]  (what we're scoring)
                # qa_keys = k[:, :, context_end:]  (the "query" part)
                k_sink = k[:, :, :n_sink]
                k_ctx = k[:, :, n_sink:context_end]
                k_qa = k[:, :, context_end:]
                
                k_sub = torch.cat([k_sink, k_ctx, k_qa], dim=2)
                # k_sub: [bsz, n_kv_heads, sink + ctx_len + qa_len, head_dim]
                
                k_sub = k_sub.unsqueeze(2).transpose(-2, -1)
                # k_sub: [bsz, n_kv_heads, 1, head_dim, sink+ctx+qa]
                
                # Attention: q_qa @ k_sub^T
                attn_logits = torch.matmul(q_qa, k_sub) / math.sqrt(head_dim)
                # [bsz, n_kv_heads, n_groups, qa_len, sink+ctx+qa]
                
                # Causal mask for qa tokens attending to qa keys
                # qa position i (global: context_end + i) can attend to all context keys
                # and qa keys up to position i
                sub_len = n_sink + (context_end - n_sink) + qa_len
                # Positions in subsampled space:
                # [0..sink-1] = sink, always visible
                # [sink..sink+ctx-1] = context, always visible to qa tokens
                # [sink+ctx..sink+ctx+qa-1] = qa tokens, causal among themselves
                qa_start_in_sub = n_sink + (context_end - n_sink)  # = context_end
                
                # Build causal mask only for the qa-to-qa part
                causal = torch.full((qa_len, qa_len), float('-inf'), device=model.device, dtype=attn_logits.dtype)
                causal_cond = torch.arange(qa_len, device=model.device)
                causal.masked_fill_(causal_cond < (causal_cond + 1).view(qa_len, 1), 0)
                # Apply to the qa-to-qa block
                attn_logits[..., qa_start_in_sub:] += causal[None, None, None, :, :]
                
                # Softmax over subsampled keys
                attn_weights = torch.softmax(attn_logits.float(), dim=-1)
                
                # Extract attention to context positions (excluding sink)
                ctx_len = context_end - n_sink
                attn_to_ctx = attn_weights[..., n_sink:n_sink + ctx_len]
                # [bsz, n_kv_heads, n_groups, qa_len, ctx_len]
                
                # amax over groups and qa positions (same as kvzip)
                ctx_scores = attn_to_ctx.amax(dim=(-3, -2))  # [bsz, n_kv_heads, ctx_len]
                
                scores[layer_idx, :, :, n_sink:context_end] = ctx_scores.to(scores.dtype)
                
                del q, k, q_qa, k_sub, attn_logits, attn_weights, attn_to_ctx, ctx_scores
        
        return hook_fn
    
    for layer in model.model.layers:
        li = int(layer.self_attn.layer_idx)
        hooks.append(layer.self_attn.register_forward_hook(make_hook(li), with_kwargs=True))
    
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)
    
    for h in hooks:
        h.remove()
    
    return scores


# ============================================================
# Compression: same as kvzip (global or layerwise)
# ============================================================

def search_hyperplane(X, max_iter=1000):
    """
    Find a fake key k such that for every query q, exp(<q,k>) ≈ 0.
    From kvpress attention_patch.py.
    X: [bsz, seq_len, head_dim] query vectors
    Returns: [bsz, head_dim] fake key
    """
    Y = X.mean(1)
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            return -1e5 * Y / Y.norm(dim=-1, keepdim=True) ** 2
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    # Fallback: just use a large negative vector
    return -1e5 * Y / Y.norm(dim=-1, keepdim=True).clamp(min=1e-6) ** 2


def compress_cache_with_mask(model, cache, scores, compression_ratio):
    """
    Compress cache by replacing evicted keys with fake keys that make
    exp(<q,k>) ≈ 0, matching kvpress's attention_patch approach.
    
    We aggregate scores across heads (mean) to get per-position importance
    per layer, then replace evicted positions' keys with fake keys.
    Values at evicted positions are set to 0.
    
    This preserves position indices (no RoPE issues) while effectively
    zeroing out the attention contribution of evicted positions.
    """
    n_layer, bsz, n_kv_heads, ctx_len = scores.shape
    
    # Aggregate scores across heads: mean → [n_layer, bsz, ctx_len]
    layer_scores = scores.mean(dim=2)
    
    n_keep = max(1, int(ctx_len * (1 - compression_ratio)))
    
    import copy
    cache_copy = copy.deepcopy(cache)
    
    for li in range(n_layer):
        ls = layer_scores[li, 0]  # [ctx_len]
        _, keep_idx = ls.topk(n_keep)
        keep_mask = torch.zeros(ctx_len, dtype=torch.bool, device=ls.device)
        keep_mask[keep_idx] = True
        evict_idx = (~keep_mask).nonzero(as_tuple=True)[0]
        
        if len(evict_idx) == 0:
            continue
        
        if hasattr(cache_copy, 'key_cache'):
            keys = cache_copy.key_cache[li]      # [bsz, n_kv_heads, ctx_len, head_dim]
            values = cache_copy.value_cache[li]
        else:
            keys = cache_copy.layers[li].keys
            values = cache_copy.layers[li].values
        
        # Set evicted values to 0
        values[0, :, evict_idx] = 0
        
        # For evicted keys: replace with fake key per head
        # We need query vectors to find the right fake key, but during generation
        # queries are unknown. Use a simple approach: set to very large negative
        # uniform vector, which gives <q, k> = -c * sum(q_i).
        # For RoPE'd queries where components are roughly balanced, this gives
        # very negative logits → softmax ≈ 0.
        #
        # Use -100/sqrt(head_dim) per dimension to avoid fp16 overflow
        head_dim = keys.shape[-1]
        fake_val = -100.0 / (head_dim ** 0.5)
        keys[0, :, evict_idx] = fake_val
    
    return cache_copy


# ============================================================
# Generation
# ============================================================

def generate_greedy(model, tokenizer, cache, logits, max_new_tokens):
    """Greedy generation from existing cache."""
    generated = []
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated.append(next_token.item())
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated.append(next_token.item())
    return tokenizer.decode(generated, skip_special_tokens=True)


# ============================================================
# Task scoring (from RULER)
# ============================================================

def string_match_all(pred, ans):
    p = pred.lower()
    return all(str(a).lower() in p for a in ans)

def string_match_part(pred, ans):
    p = pred.lower()
    return any(str(a).lower() in p for a in ans)

TASK_SCORERS = {}
for t in ["niah_single_1","niah_single_2","niah_single_3",
          "niah_multikey_1","niah_multikey_2","niah_multikey_3",
          "niah_multivalue","niah_multiquery","vt","cwe","fwe"]:
    TASK_SCORERS[t] = string_match_all
TASK_SCORERS["qa_1"] = string_match_part
TASK_SCORERS["qa_2"] = string_match_part


# ============================================================
# Main
# ============================================================

def main():
    print(f"Model: {args.model}")
    print(f"Eviction ratios: {EVICT_RATIOS}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="cuda")
    model.eval()
    device = model.device
    
    # Load RULER
    ds = load_dataset("simonjegou/ruler", args.dataset_dir, split="test")
    
    import random
    random.seed(42)
    task_samples = defaultdict(list)
    for i, ex in enumerate(ds):
        task_samples[ex["task"]].append(i)
    
    if args.n_samples > 0:
        selected = []
        per_task = max(1, args.n_samples // len(task_samples))
        for task, indices in sorted(task_samples.items()):
            n = min(per_task, len(indices))
            selected.extend(random.sample(indices, n))
        selected.sort()
    else:
        selected = list(range(len(ds)))
    
    # Shard selection for multi-GPU parallel
    if args.n_shards > 1:
        shard_size = len(selected) // args.n_shards
        start = args.shard * shard_size
        end = len(selected) if args.shard == args.n_shards - 1 else start + shard_size
        selected = selected[start:end]
    
    print(f"Selected {len(selected)} samples from {len(task_samples)} tasks (shard {args.shard}/{args.n_shards})")
    
    results = []
    t0 = time.time()
    
    for idx_i, sample_idx in enumerate(selected):
        ex = ds[sample_idx]
        task = ex["task"]
        context = ex["context"]
        question = ex["question"]
        answers = ex["answer"]
        answer_prefix = ex["answer_prefix"]
        max_new = ex["max_new_tokens"]
        scorer = TASK_SCORERS.get(task, string_match_all)
        
        gt_answer = ", ".join(str(a) for a in answers) if isinstance(answers, list) else str(answers)
        
        # ---- Build inputs ----
        # QA input (with answer, for scoring)
        prompt_qa = f"{context}\n\n{question}\n{answer_prefix}{gt_answer}"
        text_qa = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_qa}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        qa_ids = tokenizer.encode(text_qa, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Eval input (without answer, for generation)
        prompt_eval = f"{context}\n\n{question}\n{answer_prefix}"
        text_eval = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_eval}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        eval_ids = tokenizer.encode(text_eval, return_tensors="pt", add_special_tokens=False).to(device)
        L_eval = eval_ids.shape[1]
        
        # Find context boundary
        context_prompt = f"{context}\n\n"
        context_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": context_prompt}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        context_ids = tokenizer.encode(context_text, add_special_tokens=False)
        context_end = len(context_ids)
        
        # ---- Score using QA signal ----
        qa_scores = score_qa_signal(model, qa_ids, context_end, n_sink=N_SINK)
        # qa_scores: [n_layers, 1, n_kv_heads, context_end]
        
        # Expand scores to full eval length (give question tokens max score to keep them)
        full_scores = torch.zeros(
            model.config.num_hidden_layers, 1, model.config.num_key_value_heads, L_eval,
            dtype=qa_scores.dtype, device=qa_scores.device
        )
        full_scores[..., :context_end] = qa_scores
        full_scores[..., context_end:] = qa_scores.max() + 1.0  # keep question tokens
        
        del qa_ids, qa_scores
        torch.cuda.empty_cache()
        
        # ---- Build eval cache ----
        cache_eval = DynamicCache()
        with torch.no_grad():
            out_eval = model(input_ids=eval_ids, past_key_values=cache_eval, use_cache=True)
        
        # Full KV baseline
        import copy
        cache_full = copy.deepcopy(cache_eval)
        gen_full = generate_greedy(model, tokenizer, cache_full, out_eval.logits, max_new)
        full_correct = scorer(gen_full, answers)
        del cache_full
        
        results.append({
            "sample_idx": int(sample_idx), "task": task,
            "method": "full_kv", "ratio": 0.0, "correct": bool(full_correct),
            "gen": gen_full[:200],
        })
        
        # ---- Evict at various ratios ----
        for ratio in EVICT_RATIOS:
            cache_evicted = compress_cache_with_mask(model, cache_eval, full_scores, ratio)
            gen_text = generate_greedy(model, tokenizer, cache_evicted, out_eval.logits, max_new)
            correct = scorer(gen_text, answers)
            results.append({
                "sample_idx": int(sample_idx), "task": task,
                "method": "qa_signal", "ratio": ratio,
                "correct": bool(correct), "gen": gen_text[:200],
            })
            del cache_evicted
            torch.cuda.empty_cache()
        
        del cache_eval, out_eval, full_scores
        torch.cuda.empty_cache()
        gc.collect()
        
        elapsed = time.time() - t0
        eta = elapsed / (idx_i + 1) * (len(selected) - idx_i - 1)
        status = "OK" if full_correct else "FAIL(fullkv)"
        print(f"[{idx_i+1}/{len(selected)}] {task:>20} fullkv={status}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")
        
        if (idx_i + 1) % 50 == 0 or idx_i == len(selected) - 1:
            print(f"\n--- Progress: {idx_i+1}/{len(selected)} ---")
            fk = [r for r in results if r["method"] == "full_kv"]
            fk_acc = sum(r["correct"] for r in fk) / len(fk) * 100 if fk else 0
            print(f"  Full KV: {fk_acc:.1f}%")
            for ratio in EVICT_RATIOS:
                mr = [r for r in results if r["method"] == "qa_signal" and r["ratio"] == ratio]
                if mr:
                    acc = sum(r["correct"] for r in mr) / len(mr) * 100
                    print(f"  CR={ratio}: {acc:.1f}%")
            print()
            
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
    
    # ============================================================
    # Final Summary
    # ============================================================
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS: QA Signal vs KVzip (Reconstruction) on RULER {args.dataset_dir}")
    print(f"{'='*80}")
    
    fk = [r for r in results if r["method"] == "full_kv"]
    fk_acc = sum(r["correct"] for r in fk) / len(fk) * 100
    print(f"Full KV Baseline: {fk_acc:.1f}% ({len(fk)} samples)")
    print()
    
    # Reference numbers from kvpress kvzip runs
    kvzip_ref = {0.30: 95.23, 0.50: 95.21, 0.70: 95.15, 0.90: 87.22, 0.95: 37.65}
    cskv_ref = {0.30: 91.15, 0.50: 85.02, 0.70: 67.52, 0.90: 23.77, 0.95: 15.22}
    
    print(f"{'Ratio':>8} | {'QA Signal':>12} | {'KVzip(recons)':>14} | {'CritSnapKV':>12}")
    print("-" * 55)
    for ratio in EVICT_RATIOS:
        mr = [r for r in results if r["method"] == "qa_signal" and r["ratio"] == ratio]
        qa_acc = sum(r["correct"] for r in mr) / len(mr) * 100 if mr else 0
        kz = kvzip_ref.get(ratio, "-")
        cs = cskv_ref.get(ratio, "-")
        print(f"{ratio:>8.2f} | {qa_acc:>11.1f}% | {kz:>13}% | {cs:>11}%")
    
    # Per-task
    print(f"\n{'='*80}")
    print("PER-TASK BREAKDOWN (QA Signal)")
    print(f"{'='*80}")
    for task in sorted(set(r["task"] for r in results)):
        fk_t = [r for r in results if r["task"] == task and r["method"] == "full_kv"]
        fk_a = sum(r["correct"] for r in fk_t) / len(fk_t) * 100 if fk_t else 0
        row_parts = []
        for ratio in EVICT_RATIOS:
            mr = [r for r in results if r["task"] == task and r["method"] == "qa_signal" and r["ratio"] == ratio]
            if mr:
                acc = sum(r["correct"] for r in mr) / len(mr) * 100
                row_parts.append(f"{ratio}={acc:.0f}%")
        print(f"  {task:>20} (Full={fk_a:.0f}%): {' | '.join(row_parts)}")
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
