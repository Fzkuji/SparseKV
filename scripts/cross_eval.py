"""
Cross-evaluation: Reconstruction vs QA scoring × Reconstruction vs QA eval.

For each sample:
1. Prefill context, get cache
2. Score with Reconstruction signal (kvzip) → importance scores
3. Score with QA signal (attention from Q+A tokens) → importance scores  
4. For each scoring method × each compression ratio:
   a. Apply eviction (masked_key_indices)
   b. Eval QA: generate answer to question → check correctness
   c. Eval Reconstruction: (optional, skip for now)
5. Output per-sample results for visualization

Usage:
    ~/kvpress/.venv/bin/python -u scripts/cross_eval.py
"""
import torch
import json
import os
import gc
import time
import random
import math
from collections import defaultdict
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    pipeline as hf_pipeline,
)
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from kvpress import KVzipPress
from kvpress.presses.base_press import BasePress

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "100"))
RULER_LEN = os.environ.get("RULER_LEN", "4096")
OUTPUT = os.environ.get("OUTPUT", f"results/cross_eval_ruler{RULER_LEN}.json")

RATIOS = [0.30, 0.50, 0.70, 0.90, 0.95]
N_SINK = 4

print(f"Model: {MODEL}")
print(f"Ratios: {RATIOS}")
print(f"N_SAMPLES: {N_SAMPLES}")

# Use kvpress pipeline for consistent eval
pipe = hf_pipeline(
    "kv-press-text-generation",
    model=MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = pipe.tokenizer
model = pipe.model

# Load RULER
ds = load_dataset("simonjegou/ruler", RULER_LEN, split="test")
random.seed(42)
task_samples = defaultdict(list)
for i, ex in enumerate(ds):
    task_samples[ex["task"]].append(i)

selected = []
per_task = max(1, N_SAMPLES // len(task_samples))
for task, indices in sorted(task_samples.items()):
    n = min(per_task, len(indices))
    selected.extend(random.sample(indices, n))
selected.sort()
print(f"Selected {len(selected)} samples from {len(task_samples)} tasks")

# Scoring functions
def string_match_all(pred, ans):
    return all(str(a).lower() in pred.lower() for a in ans)

def string_match_part(pred, ans):
    return any(str(a).lower() in pred.lower() for a in ans)

TASK_SCORERS = {}
for t in ["niah_single_1","niah_single_2","niah_single_3",
          "niah_multikey_1","niah_multikey_2","niah_multikey_3",
          "niah_multivalue","niah_multiquery","vt","cwe","fwe"]:
    TASK_SCORERS[t] = string_match_all
TASK_SCORERS["qa_1"] = string_match_part
TASK_SCORERS["qa_2"] = string_match_part


def compute_qa_scores(model, tokenizer, context, question, answer, n_sink=4, use_answer=True, answer_only=False):
    """
    Compute QA-based importance scores for context KV positions.
    
    Forward pass with [context + question (+ answer)], then use attention
    from question(+answer) tokens to context tokens as importance signal.
    
    Args:
        use_answer: if True, include answer in forward (oracle upper bound)
                   if False, only use question tokens (practical method)
        answer_only: if True, use_answer must be True; only use answer tokens'
                    attention (not question tokens) as scoring signal.
    
    Returns: scores tensor [n_layers, 1, n_kv_heads, context_len]
    """
    if answer_only:
        use_answer = True  # answer_only requires answer in forward
    
    # Build full input
    user_msg = f"{context}\n\n{question}"
    if use_answer:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    else:
        messages = [{"role": "user", "content": user_msg}]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    
    # Also build context-only to find boundary
    ctx_msg = [{"role": "user", "content": context + "####SPLIT####"}]
    ctx_text = tokenizer.apply_chat_template(
        ctx_msg, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    
    # Build context+question boundary (for answer_only mode)
    if answer_only:
        cq_msg = [{"role": "user", "content": user_msg}]
        cq_text = tokenizer.apply_chat_template(
            cq_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        cq_ids = tokenizer.encode(cq_text, return_tensors="pt", add_special_tokens=False)
        cq_len = cq_ids.shape[1]  # context + question length
    ctx_part = ctx_text.split("####SPLIT####")[0]
    ctx_ids = tokenizer.encode(ctx_part, return_tensors="pt", add_special_tokens=False)
    context_len = ctx_ids.shape[1]
    
    full_ids = tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    total_len = full_ids.shape[1]
    qa_len = total_len - context_len
    
    if qa_len <= 0:
        return None
    
    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // n_heads
    n_groups = n_heads // n_kv_heads
    
    scores = torch.zeros(n_layers, 1, n_kv_heads, context_len,
                        dtype=torch.float32, device=model.device)
    
    # Hook into attention to compute QA→context scores from Q,K directly
    # This avoids storing the full [seq, seq] attention matrix (OOM for long seqs)
    # We only compute the qa_tokens × context_tokens block of the attention matrix
    
    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # Access Q, K from the module's cached computation
            # We intercept BEFORE attention by hooking q_proj and k_proj separately
            pass
        return hook_fn
    
    # Better approach: hook into the attention function to intercept Q and K
    # Use a pre-hook on the attention module to capture hidden_states,
    # then compute Q, K, and the partial attention scores ourselves
    
    def make_attn_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # After attention forward, we can access the rotated Q and K
            # But they're not stored... We need to recompute from hidden_states
            # Actually, let's just compute QK scores from the input hidden_states
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return output
            
            bsz, total, hid = hidden_states.shape
            
            # Project to Q, K
            q = module.q_proj(hidden_states)
            k = module.k_proj(hidden_states)
            
            q = q.view(bsz, total, n_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, total, n_kv_heads, head_dim).transpose(1, 2)
            
            # Apply RoPE (need position_ids)
            position_ids = kwargs.get("position_ids", None)
            if position_ids is None:
                position_ids = torch.arange(total, device=hidden_states.device).unsqueeze(0)
            position_embeddings = kwargs.get("position_embeddings", None)
            if position_embeddings is None:
                position_embeddings = module.rotary_emb(k, position_ids)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
            # Only compute QA→context attention scores (not full matrix)
            if answer_only:
                q_qa = q[:, :, cq_len:]  # [bsz, n_heads, answer_len, head_dim]
            else:
                q_qa = q[:, :, context_len:]  # [bsz, n_heads, qa_len, head_dim]
            k_ctx = k[:, :, :context_len]  # [bsz, n_kv_heads, ctx_len, head_dim]
            
            # Expand k for GQA
            k_ctx_expanded = k_ctx.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)
            k_ctx_expanded = k_ctx_expanded.reshape(bsz, n_heads, context_len, head_dim)
            
            # QA queries attending to context keys: [bsz, n_heads, qa_len, ctx_len]
            scale = head_dim ** -0.5
            attn_scores = torch.matmul(q_qa, k_ctx_expanded.transpose(-2, -1)) * scale
            # Softmax over ALL keys (ctx + qa), but we only need relative scores for ctx
            # For importance ranking, raw scores (pre-softmax) work fine
            # Take max over qa positions and group into kv heads
            attn_scores = attn_scores.view(bsz, n_kv_heads, n_groups, qa_len, context_len)
            layer_scores = attn_scores.amax(dim=(2, 3))  # [bsz, n_kv_heads, ctx_len]
            scores[layer_idx, 0] = layer_scores.float()
            
            return output
        return hook_fn
    
    hooks = []
    try:
        for i, layer in enumerate(model.model.layers):
            h = layer.self_attn.register_forward_hook(make_attn_hook(i), with_kwargs=True)
            hooks.append(h)
        
        with torch.no_grad():
            model.model(input_ids=full_ids)  # normal SDPA forward, no output_attentions
    finally:
        for h in hooks:
            h.remove()
    
    return scores


class QASignalPress(BasePress):
    """
    A Press that uses pre-computed QA attention scores for eviction.
    Plugs into kvpress pipeline seamlessly.
    compress() prunes low-scoring KV pairs per layer based on compression_ratio.
    """
    compression_ratio: float = 0.0
    
    def __init__(self, compression_ratio: float, qa_scores: torch.Tensor):
        self.compression_ratio = compression_ratio
        self.qa_scores = qa_scores  # [n_layers, 1, n_kv_heads, ctx_len]
    
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        """Prune KV pairs with lowest QA scores."""
        layer_idx = int(module.layer_idx)
        bsz, n_kv_heads, seq_len, head_dim = keys.shape
        dev = keys.device
        if layer_idx == 0:
            n_keep = max(N_SINK, int(seq_len * (1 - self.compression_ratio)))
            print(f"  [QAPress] layer0: seq_len={seq_len} -> keep={n_keep} (ratio={self.compression_ratio})")
        
        layer_scores = self.qa_scores[layer_idx].to(dev)  # [1, n_kv_heads, ctx_len]
        # Align with actual cache length
        if layer_scores.shape[-1] > seq_len:
            layer_scores = layer_scores[:, :, :seq_len]
        elif layer_scores.shape[-1] < seq_len:
            pad = torch.zeros(1, n_kv_heads, seq_len - layer_scores.shape[-1],
                            device=dev, dtype=layer_scores.dtype)
            layer_scores = torch.cat([layer_scores, pad], dim=-1)
        
        # Number of KV pairs to keep (per head, uniform across heads)
        n_keep = max(N_SINK, int(seq_len * (1 - self.compression_ratio)))
        
        # Always keep sink tokens
        sink_scores = torch.full((bsz, n_kv_heads, N_SINK), float('inf'),
                                device=dev, dtype=layer_scores.dtype)
        layer_scores = layer_scores.clone()
        layer_scores[:, :, :N_SINK] = sink_scores
        
        # Top-k per head
        _, keep_idx = layer_scores.topk(n_keep, dim=-1)  # [bsz, n_kv_heads, n_keep]
        keep_idx = keep_idx.sort(dim=-1).values
        
        # Gather
        keys = keys.gather(2, keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        values = values.gather(2, keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        
        return keys, values


def eval_kvzip(pipe, context, question, answer_prefix, max_new, ratio):
    """Evaluate using kvpress KVzipPress (reconstruction scoring + pipeline eval)."""
    press = KVzipPress(compression_ratio=ratio)
    try:
        out = pipe(context, question=question, answer_prefix=answer_prefix,
                  press=press, max_new_tokens=max_new, do_sample=False)
        return out["answer"]
    except Exception as e:
        return f"ERROR: {e}"


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
    
    # --- Full KV baseline ---
    try:
        out_full = pipe(context, question=question, answer_prefix=answer_prefix,
                       max_new_tokens=max_new, do_sample=False)
        gen_full = out_full["answer"]
    except Exception as e:
        gen_full = f"ERROR: {e}"
    full_correct = scorer(gen_full, answers)
    
    results.append({
        "sample_idx": int(sample_idx), "task": task,
        "scoring": "none", "ratio": 0.0,
        "correct": bool(full_correct), "gen": gen_full[:200],
    })
    
    # --- Compute QA scores (oracle: with answer) ---
    answer_for_scoring = str(answers[0]) if answers else ""
    try:
        qa_scores_oracle = compute_qa_scores(model, tokenizer, context, question, answer_for_scoring, use_answer=True)
    except torch.cuda.OutOfMemoryError:
        print(f"  QA oracle scoring OOM, skipping")
        torch.cuda.empty_cache(); gc.collect()
        qa_scores_oracle = None
    
    # --- Compute QA scores (question-only: no answer) ---
    try:
        qa_scores_qonly = compute_qa_scores(model, tokenizer, context, question, "", use_answer=False)
    except torch.cuda.OutOfMemoryError:
        print(f"  QA question-only scoring OOM, skipping")
        torch.cuda.empty_cache(); gc.collect()
        qa_scores_qonly = None
    
    # --- Compute QA scores (answer-only: only answer tokens' attention) ---
    try:
        qa_scores_aonly = compute_qa_scores(model, tokenizer, context, question, answer_for_scoring, answer_only=True)
    except torch.cuda.OutOfMemoryError:
        print(f"  QA answer-only scoring OOM, skipping")
        torch.cuda.empty_cache(); gc.collect()
        qa_scores_aonly = None
    
    for ratio in RATIOS:
        # --- KVzip (reconstruction scoring) ---
        gen_kvzip = eval_kvzip(pipe, context, question, answer_prefix, max_new, ratio)
        kvzip_correct = scorer(gen_kvzip, answers)
        results.append({
            "sample_idx": int(sample_idx), "task": task,
            "scoring": "reconstruction", "ratio": ratio,
            "correct": bool(kvzip_correct), "gen": gen_kvzip[:200],
        })
        
        # --- QA oracle (question + answer) ---
        if qa_scores_oracle is not None:
            qa_press = QASignalPress(compression_ratio=ratio, qa_scores=qa_scores_oracle)
            try:
                out_qa = pipe(context, question=question, answer_prefix=answer_prefix,
                             press=qa_press, max_new_tokens=max_new, do_sample=False)
                gen_qa = out_qa["answer"]
            except Exception as e:
                gen_qa = f"ERROR: {e}"
        else:
            gen_qa = "ERROR: no qa_scores"
        qa_correct = scorer(gen_qa, answers)
        results.append({
            "sample_idx": int(sample_idx), "task": task,
            "scoring": "qa_oracle", "ratio": ratio,
            "correct": bool(qa_correct), "gen": str(gen_qa)[:200],
        })
        
        # --- Question-only ---
        if qa_scores_qonly is not None:
            qonly_press = QASignalPress(compression_ratio=ratio, qa_scores=qa_scores_qonly)
            try:
                out_qonly = pipe(context, question=question, answer_prefix=answer_prefix,
                               press=qonly_press, max_new_tokens=max_new, do_sample=False)
                gen_qonly = out_qonly["answer"]
            except Exception as e:
                gen_qonly = f"ERROR: {e}"
        else:
            gen_qonly = "ERROR: no qa_scores"
        qonly_correct = scorer(gen_qonly, answers)
        results.append({
            "sample_idx": int(sample_idx), "task": task,
            "scoring": "question_only", "ratio": ratio,
            "correct": bool(qonly_correct), "gen": str(gen_qonly)[:200],
        })
        
        # --- Answer-only ---
        if qa_scores_aonly is not None:
            aonly_press = QASignalPress(compression_ratio=ratio, qa_scores=qa_scores_aonly)
            try:
                out_aonly = pipe(context, question=question, answer_prefix=answer_prefix,
                               press=aonly_press, max_new_tokens=max_new, do_sample=False)
                gen_aonly = out_aonly["answer"]
            except Exception as e:
                gen_aonly = f"ERROR: {e}"
        else:
            gen_aonly = "ERROR: no qa_scores"
        aonly_correct = scorer(gen_aonly, answers)
        results.append({
            "sample_idx": int(sample_idx), "task": task,
            "scoring": "answer_only", "ratio": ratio,
            "correct": bool(aonly_correct), "gen": str(gen_aonly)[:200],
        })
        
        torch.cuda.empty_cache()
        gc.collect()
    
    elapsed = time.time() - t0
    eta = elapsed / (idx_i + 1) * (len(selected) - idx_i - 1)
    status = "OK" if full_correct else "FAIL"
    print(f"[{idx_i+1}/{len(selected)}] {task:>20} fullkv={status}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")
    
    if (idx_i + 1) % 10 == 0:
        print(f"\n--- Progress: {idx_i+1}/{len(selected)} ---")
        fk = [r for r in results if r["scoring"] == "none"]
        fk_acc = sum(r["correct"] for r in fk) / len(fk) * 100
        print(f"  Full KV: {fk_acc:.1f}%")
        for ratio in RATIOS:
            recon = [r for r in results if r["scoring"] == "reconstruction" and r["ratio"] == ratio]
            qo = [r for r in results if r["scoring"] == "qa_oracle" and r["ratio"] == ratio]
            qq = [r for r in results if r["scoring"] == "question_only" and r["ratio"] == ratio]
            ao = [r for r in results if r["scoring"] == "answer_only" and r["ratio"] == ratio]
            r_acc = sum(r["correct"] for r in recon) / len(recon) * 100 if recon else 0
            o_acc = sum(r["correct"] for r in qo) / len(qo) * 100 if qo else 0
            q_acc = sum(r["correct"] for r in qq) / len(qq) * 100 if qq else 0
            a_acc = sum(r["correct"] for r in ao) / len(ao) * 100 if ao else 0
            print(f"  CR={ratio}: Recon={r_acc:.1f}%  Q-only={q_acc:.1f}%  A-only={a_acc:.1f}%  Oracle={o_acc:.1f}%")
        print()
        
        os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)

# Final summary
print(f"\n{'='*60}")
print(f"FINAL: Cross-Eval on RULER {RULER_LEN}")
print(f"{'='*60}")
fk = [r for r in results if r["scoring"] == "none"]
fk_acc = sum(r["correct"] for r in fk) / len(fk) * 100
print(f"Full KV: {fk_acc:.1f}% ({len(fk)} samples)")
print(f"{'Ratio':>8} | {'Recon':>10} | {'Q-only':>10} | {'Oracle':>10}")
print("-" * 50)
for ratio in RATIOS:
    recon = [r for r in results if r["scoring"] == "reconstruction" and r["ratio"] == ratio]
    qo = [r for r in results if r["scoring"] == "qa_oracle" and r["ratio"] == ratio]
    qq = [r for r in results if r["scoring"] == "question_only" and r["ratio"] == ratio]
    r_acc = sum(r["correct"] for r in recon) / len(recon) * 100 if recon else 0
    o_acc = sum(r["correct"] for r in qo) / len(qo) * 100 if qo else 0
    q_acc = sum(r["correct"] for r in qq) / len(qq) * 100 if qq else 0
    print(f"{ratio:>8.2f} | {r_acc:>9.1f}% | {q_acc:>9.1f}% | {o_acc:>9.1f}%")

os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT}")
print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")
