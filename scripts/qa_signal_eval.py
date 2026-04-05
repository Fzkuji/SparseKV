"""
QA Signal Eviction Experiment on RULER 4096.

Compare with existing kvzip (reconstruction-based) results.
QA signal: Prefill [context+question+answer], use attention FROM question+answer tokens
TO context tokens (amax over groups and queries) as importance score.
Then evict at various ratios and re-answer.

This follows Fast KVzip's description: max attention received by each KV feature
under the instruction QA task (Huang et al., 2024).

Usage:
    python -u scripts/qa_signal_eval.py [--model MODEL] [--n_samples N] [--output PATH]
"""
import torch, numpy as np, json, random, copy, os, argparse, time, math, gc
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import rotate_half

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--n_samples", type=int, default=0, help="0 = all samples")
parser.add_argument("--output", default="results/qa_signal_ruler4096.json")
parser.add_argument("--dataset_dir", default="4096")
args = parser.parse_args()

MODEL = args.model
# Match kvzip's ratios: 0.30, 0.50, 0.70, 0.90, 0.95
EVICT_RATIOS = [0.30, 0.50, 0.70, 0.90, 0.95]
SINK = 4
RECENT = 64

print(f"Model: {MODEL}")
print(f"Eviction ratios: {EVICT_RATIOS}")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
model.eval()
device = model.device

n_layers = model.config.num_hidden_layers
n_q_heads = model.config.num_attention_heads
n_kv_heads = model.config.num_key_value_heads
n_groups = n_q_heads // n_kv_heads
head_dim = model.config.hidden_size // n_q_heads
scale = head_dim ** 0.5

# Load RULER 4096
print("Loading RULER dataset...")
ds = load_dataset("simonjegou/ruler", args.dataset_dir, split="test")

# Sample selection: same as kvpress eval (all samples or N per task)
random.seed(42)
task_samples = defaultdict(list)
for i, ex in enumerate(ds):
    task_samples[ex["task"]].append(i)

if args.n_samples > 0:
    selected = []
    n_tasks = len(task_samples)
    per_task = max(1, args.n_samples // n_tasks)
    for task, indices in sorted(task_samples.items()):
        n = min(per_task, len(indices))
        selected.extend(random.sample(indices, n))
    selected.sort()
else:
    selected = list(range(len(ds)))

print(f"Selected {len(selected)} samples from {len(task_samples)} tasks")

# Scoring functions (same as kvpress)
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


def generate_with_cache(cache, logits, max_new_tokens):
    """Generate tokens using existing KV cache, starting from logits."""
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


def compute_qa_scores(input_ids, q_start, n_layers, n_kv_heads, n_groups, n_q_heads, head_dim, scale, device):
    """
    Compute QA importance scores via hook-based attention extraction.
    
    For each KV position j in [0, q_start), compute:
        score[layer, head, j] = max over (groups g, query positions i >= q_start) of attn[g, i, j]
    
    This is the "instruction QA" signal from Fast KVzip:
    - Only attention FROM question+answer tokens TO context tokens matters
    - Aggregation: amax over groups and query positions (not mean, not sum)
    
    We process one KV group at a time to save memory.
    """
    seq_len = input_ids.shape[1]
    scores = torch.zeros(n_layers, n_kv_heads, q_start, device='cpu')
    
    hooks = []
    
    def make_hook(li):
        def hook_fn(module, args, kwargs, output):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is None:
                return
            bsz, slen, _ = hs.shape
            if slen != seq_len:  # only during our scoring forward
                return
            pe = kwargs.get("position_embeddings", None)
            with torch.no_grad():
                q = module.q_proj(hs).view(bsz, slen, n_q_heads, head_dim).transpose(1, 2)
                k = module.k_proj(hs).view(bsz, slen, n_kv_heads, head_dim).transpose(1, 2)
                if pe is not None:
                    cos, sin = pe
                    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                    k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
                
                # Process one KV group at a time
                for g in range(n_kv_heads):
                    qh = slice(g * n_groups, (g + 1) * n_groups)
                    q_g = q[0, qh]  # [n_groups, seq_len, head_dim]
                    k_g = k[0, g:g+1].expand(n_groups, -1, -1)  # [n_groups, seq_len, head_dim]
                    
                    # Only compute attention FROM q+a positions TO context positions
                    # q_g[:, q_start:, :] @ k_g[:, :q_start, :].T
                    q_qa = q_g[:, q_start:, :]  # [n_groups, qa_len, head_dim]
                    k_ctx = k_g[:, :q_start, :]  # [n_groups, ctx_len, head_dim]
                    
                    # Full logits for softmax denominator (causal: each q attends to all k <= its position)
                    # For q positions >= q_start, they can attend to ALL k positions [0, seq_len)
                    # But softmax should be over ALL keys, not just context keys
                    # So we need full row for correct softmax normalization
                    logits_full = torch.matmul(q_qa, k_g.transpose(-1, -2)) / scale
                    # [n_groups, qa_len, seq_len]
                    
                    # Causal mask: q at position (q_start + i) can only attend to k at position <= (q_start + i)
                    qa_len = seq_len - q_start
                    # For each qa position i, mask out k positions > (q_start + i)
                    q_positions = torch.arange(q_start, seq_len, device=device).unsqueeze(1)  # [qa_len, 1]
                    k_positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
                    causal_mask = k_positions > q_positions  # [qa_len, seq_len]
                    logits_full.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
                    
                    attn = torch.softmax(logits_full.float(), dim=-1)  # [n_groups, qa_len, seq_len]
                    
                    # Extract only attention to context positions [0, q_start)
                    attn_to_ctx = attn[:, :, :q_start]  # [n_groups, qa_len, ctx_len]
                    
                    # amax over groups and qa positions
                    max_score = attn_to_ctx.amax(dim=(0, 1))  # [ctx_len]
                    scores[li, g] = max_score.cpu()
                    
                    del logits_full, attn, attn_to_ctx, max_score, q_qa, k_ctx, causal_mask
                del q, k
        return hook_fn
    
    for layer in model.model.layers:
        li = int(layer.self_attn.layer_idx)
        hooks.append(layer.self_attn.register_forward_hook(make_hook(li), with_kwargs=True))
    
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)
    
    for h in hooks:
        h.remove()
    
    return scores


def evict_cache(cache, scores, n_layers, n_kv_heads, kv_len, ratio, device):
    """Evict KV pairs based on scores. Keep sink + recent + top-scored middle tokens."""
    cache_copy = copy.deepcopy(cache)
    
    middle_start = SINK
    middle_end = max(0, kv_len - RECENT)
    n_middle = middle_end - middle_start
    
    if n_middle <= 0:
        return cache_copy, 0.0
    
    n_evict = int(n_middle * ratio)
    n_keep = n_middle - n_evict
    
    if n_keep <= 0:
        n_keep = 1
    
    for li in range(n_layers):
        for h in range(n_kv_heads):
            s = scores[li, h, middle_start:middle_end]
            _, topk_idx = s.topk(min(n_keep, len(s)))
            
            keep_mid = torch.zeros(n_middle, dtype=torch.bool)
            keep_mid[topk_idx] = True
            evict_positions = (~keep_mid).nonzero(as_tuple=True)[0] + middle_start
            evict_positions = evict_positions.to(device)
            
            # Zero out evicted positions
            if hasattr(cache_copy, 'key_cache'):
                cache_copy.key_cache[li][0, h, evict_positions] = 0
                cache_copy.value_cache[li][0, h, evict_positions] = 0
            else:
                cache_copy.layers[li].keys[0, h, evict_positions] = 0
                cache_copy.layers[li].values[0, h, evict_positions] = 0
    
    n_kept = SINK + RECENT + n_keep
    actual_cr = 1.0 - n_kept / kv_len
    return cache_copy, actual_cr


# ============================================================
# Main loop
# ============================================================
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
    
    # ---- Step 1: Build QA scoring input (with ground truth answer) ----
    prompt_with_answer = f"{context}\n\n{question}\n{answer_prefix}{gt_answer}"
    full_text_qa = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_with_answer}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    qa_ids = tokenizer.encode(full_text_qa, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Find where question starts (context-only prompt to find boundary)
    prompt_no_answer = f"{context}\n\n{question}\n{answer_prefix}"
    full_text_eval = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_no_answer}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    eval_ids = tokenizer.encode(full_text_eval, return_tensors="pt", add_special_tokens=False).to(device)
    L_eval = eval_ids.shape[1]
    
    # q_start: where context ends and question begins
    # Approximate by encoding context-only
    context_only_prompt = f"{context}\n\n"
    context_only_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": context_only_prompt}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    context_only_ids = tokenizer.encode(context_only_text, add_special_tokens=False)
    q_start = len(context_only_ids)
    
    # ---- Step 2: Compute QA scores ----
    # Score context positions [0, q_start) using attention from [q_start, L_qa)
    # But we score on eval_ids (without answer) for fair comparison with kvzip
    # Actually no: the QA signal NEEDS the answer to generate answer-aware attention
    # So we score on qa_ids (with answer), but only use scores for [0, L_eval) positions
    
    qa_scores = compute_qa_scores(
        qa_ids, q_start, n_layers, n_kv_heads, n_groups, n_q_heads, head_dim, scale, device
    )
    # qa_scores shape: [n_layers, n_kv_heads, q_start]
    # We need scores for [0, L_eval), pad with zeros for positions [q_start, L_eval)
    full_scores = torch.zeros(n_layers, n_kv_heads, L_eval)
    full_scores[:, :, :q_start] = qa_scores
    # Question+answer_prefix positions [q_start, L_eval) get score 0 → will be evicted first
    # But actually these are part of the prompt, we should keep them
    # Give them max score so they're never evicted
    full_scores[:, :, q_start:L_eval] = full_scores[:, :, :q_start].max() + 1.0
    
    del qa_ids
    torch.cuda.empty_cache()
    
    # ---- Step 3: Build eval cache (without answer) ----
    cache_eval = DynamicCache()
    with torch.no_grad():
        out_eval = model(input_ids=eval_ids, past_key_values=cache_eval, use_cache=True)
    
    # Full KV baseline
    cache_full = copy.deepcopy(cache_eval)
    gen_full = generate_with_cache(cache_full, out_eval.logits, max_new)
    full_correct = scorer(gen_full, answers)
    del cache_full
    
    results.append({
        "sample_idx": int(sample_idx), "task": task,
        "method": "full_kv", "ratio": 0.0, "correct": bool(full_correct),
        "gen": gen_full[:200],
    })
    
    # ---- Step 4: Evict at various ratios ----
    for ratio in EVICT_RATIOS:
        cache_evicted, actual_cr = evict_cache(
            cache_eval, full_scores, n_layers, n_kv_heads, L_eval, ratio, device
        )
        gen_text = generate_with_cache(cache_evicted, out_eval.logits, max_new)
        correct = scorer(gen_text, answers)
        results.append({
            "sample_idx": int(sample_idx), "task": task,
            "method": "qa_signal", "ratio": ratio, "actual_cr": round(actual_cr, 4),
            "correct": bool(correct), "gen": gen_text[:200],
        })
        del cache_evicted
        torch.cuda.empty_cache()
    
    del cache_eval, out_eval, full_scores, qa_scores
    torch.cuda.empty_cache()
    gc.collect()
    
    elapsed = time.time() - t0
    eta = elapsed / (idx_i + 1) * (len(selected) - idx_i - 1)
    status = "OK" if full_correct else "FAIL(fullkv)"
    print(f"[{idx_i+1}/{len(selected)}] {task:>20} fullkv={status}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")
    
    # Progress summary every 50 samples
    if (idx_i + 1) % 50 == 0 or idx_i == len(selected) - 1:
        print(f"\n--- Progress: {idx_i+1}/{len(selected)} ---")
        fk = [r for r in results if r["method"] == "full_kv"]
        fk_acc = sum(r["correct"] for r in fk) / len(fk) * 100 if fk else 0
        print(f"  Full KV: {fk_acc:.1f}%")
        row = []
        for ratio in EVICT_RATIOS:
            mr = [r for r in results if r["method"] == "qa_signal" and r["ratio"] == ratio]
            if mr:
                acc = sum(r["correct"] for r in mr) / len(mr) * 100
                row.append(f"CR={ratio}: {acc:.1f}%")
        print(f"  QA Signal: {' | '.join(row)}")
        print()
        
        # Save intermediate
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

# ============================================================
# Final Summary
# ============================================================
print(f"\n{'='*80}")
print(f"FINAL RESULTS: QA Signal Eviction on RULER {args.dataset_dir}")
print(f"{'='*80}")

fk = [r for r in results if r["method"] == "full_kv"]
fk_acc = sum(r["correct"] for r in fk) / len(fk) * 100
print(f"Full KV Baseline: {fk_acc:.1f}%")
print()

print(f"{'Ratio':>8} | {'QA Signal':>12} | {'KVzip (ref)':>12} | {'CritSnapKV (ref)':>16}")
print("-" * 60)

kvzip_ref = {0.30: 95.23, 0.50: 95.21, 0.70: 95.15, 0.90: 87.22, 0.95: 37.65}
cskv_ref = {0.30: 91.15, 0.50: 85.02, 0.70: 67.52, 0.90: 23.77, 0.95: 15.22}

for ratio in EVICT_RATIOS:
    mr = [r for r in results if r["method"] == "qa_signal" and r["ratio"] == ratio]
    qa_acc = sum(r["correct"] for r in mr) / len(mr) * 100 if mr else 0
    kz = kvzip_ref.get(ratio, "-")
    cs = cskv_ref.get(ratio, "-")
    kz_str = f"{kz:.2f}%" if isinstance(kz, float) else kz
    cs_str = f"{cs:.2f}%" if isinstance(cs, float) else cs
    print(f"{ratio:>8.2f} | {qa_acc:>11.1f}% | {kz_str:>12} | {cs_str:>16}")

# Per-task breakdown
print(f"\n{'='*80}")
print("PER-TASK BREAKDOWN")
print(f"{'='*80}")
for task in sorted(set(r["task"] for r in results)):
    fk_t = [r for r in results if r["task"] == task and r["method"] == "full_kv"]
    fk_a = sum(r["correct"] for r in fk_t) / len(fk_t) * 100 if fk_t else 0
    print(f"\n  {task} (FullKV={fk_a:.0f}%):")
    row = []
    for ratio in EVICT_RATIOS:
        mr = [r for r in results if r["task"] == task and r["method"] == "qa_signal" and r["ratio"] == ratio]
        if mr:
            acc = sum(r["correct"] for r in mr) / len(mr) * 100
            row.append(f"{acc:5.1f}%")
        else:
            row.append("  N/A ")
    print(f"    QA:  {' | '.join(row)}")

# Save final
os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
with open(args.output, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {args.output}")
print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")
print("DONE")
