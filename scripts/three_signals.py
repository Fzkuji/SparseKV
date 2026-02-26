"""
Preliminary experiment: Compare 3 KV importance scoring methods across eviction ratios.
- SnapKV: attention from last W tokens (next-token proxy)
- QA: attention from question tokens (query-aware)
- Recons: total attention received from all positions (reconstruction proxy)

For each sample × method × ratio: evict tokens and check accuracy.
Outputs JSON for visualization with plot_three_signals.py.

Usage:
    python -u scripts/three_signals.py [--model MODEL] [--n_samples N] [--output PATH]
"""
import torch, numpy as np, json, random, copy, os, argparse, time
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half
from kvpress.attention_patch import patch_attention_functions
patch_attention_functions()

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--n_samples", type=int, default=100, help="Samples per task (0=all)")
parser.add_argument("--output", default="results/three_signals.json")
parser.add_argument("--dataset_dir", default="4096", help="RULER data_dir")
args = parser.parse_args()

MODEL = args.model
EVICT_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
SINK = 4
RECENT = 64
SNAP_W = 32

print(f"Model: {MODEL}")
print(f"Eviction ratios: {EVICT_RATIOS}")
print(f"Target samples: {args.n_samples}")

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

# Load RULER
print("Loading RULER dataset...")
ds = load_dataset("simonjegou/ruler", args.dataset_dir, split="test")
random.seed(42)
task_samples = defaultdict(list)
for i, ex in enumerate(ds):
    task_samples[ex["task"]].append(i)

# Select samples: evenly from each task
selected = []
n_tasks = len(task_samples)
per_task = max(1, args.n_samples // n_tasks) if args.n_samples > 0 else None
for task, indices in sorted(task_samples.items()):
    n = min(per_task, len(indices)) if per_task else len(indices)
    selected.extend(random.sample(indices, n))
selected.sort()
print(f"Selected {len(selected)} samples from {n_tasks} tasks ({per_task} per task)")

# Scoring
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


def generate_with_cache(cache, out, max_new_tokens):
    generated = []
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated.append(next_token.item())
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated.append(next_token.item())
    return tokenizer.decode(generated, skip_special_tokens=True)


# Main loop
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
    scorer = TASK_SCORERS[task]

    # Build prompt
    prompt = f"{context}\n\n{question}\n{answer_prefix}"
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(full_text, return_tensors="pt",
                                  add_special_tokens=False).to(device)
    L = prompt_ids.shape[1]

    # Find question boundary
    q_text = f"\n\n{question}\n{answer_prefix}"
    q_only_ids = tokenizer.encode(q_text, add_special_tokens=False)
    q_len = len(q_only_ids)
    q_start = L - q_len

    # Allocate score tensors (CPU)
    snapkv_scores = torch.zeros(n_layers, n_kv_heads, L)
    qa_scores = torch.zeros(n_layers, n_kv_heads, L)
    recons_scores = torch.zeros(n_layers, n_kv_heads, L)

    causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    # Register scoring hooks
    hook_handles = []
    def make_hook(li):
        def hook_fn(module, args, kwargs, output):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is None:
                return
            bsz, sl, _ = hs.shape
            if sl != L:
                return
            pe = kwargs.get("position_embeddings", None) if kwargs else None

            with torch.no_grad():
                q = module.q_proj(hs).view(bsz, sl, n_q_heads, head_dim).transpose(1, 2)
                k = module.k_proj(hs).view(bsz, sl, n_kv_heads, head_dim).transpose(1, 2)
                if pe is not None:
                    cos, sin = pe
                    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                    k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))

                for g in range(n_kv_heads):
                    qh = slice(g * n_groups, (g + 1) * n_groups)
                    q_g = q[0, qh]                                   # [4, L, 128]
                    k_g = k[0, g:g+1].expand(n_groups, -1, -1)      # [4, L, 128]

                    logits_g = torch.matmul(q_g, k_g.transpose(-1, -2)) / scale
                    logits_g.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
                    attn_g = torch.softmax(logits_g.float(), dim=-1)  # [4, L, L]
                    attn_avg = attn_g.mean(dim=0)                     # [L, L]

                    # 1) SnapKV: last W tokens → all positions
                    w = min(SNAP_W, L)
                    snapkv_scores[li, g] = attn_avg[-w:, :].mean(dim=0).cpu()

                    # 2) QA: question tokens → all positions
                    if q_start < L:
                        qa_scores[li, g] = attn_avg[q_start:, :].mean(dim=0).cpu()

                    # 3) Reconstruction: column sum (total attention received)
                    recons_scores[li, g] = attn_avg.sum(dim=0).cpu()

                    del logits_g, attn_g, attn_avg
                del q, k
        return hook_fn

    for layer in model.model.layers:
        li = int(layer.self_attn.layer_idx)
        hook_handles.append(
            layer.self_attn.register_forward_hook(make_hook(li), with_kwargs=True))

    # Forward pass: build cache + compute scores via hooks
    cache = DynamicCache()
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)

    for h in hook_handles:
        h.remove()
    del causal_mask
    torch.cuda.empty_cache()

    # Full KV baseline
    cache_full = copy.deepcopy(cache)
    gen_full = generate_with_cache(cache_full, out, max_new)
    full_correct = scorer(gen_full, answers)
    del cache_full
    results.append({
        "sample_idx": int(sample_idx), "task": task,
        "method": "full_kv", "ratio": 0.0, "correct": bool(full_correct),
        "gen": gen_full[:200],
    })

    # Middle token range
    middle_start = SINK
    middle_end = max(0, L - RECENT)
    n_middle = middle_end - middle_start

    # Test each method × ratio
    methods = {"snapkv": snapkv_scores, "qa": qa_scores, "recons": recons_scores}
    for method_name, scores in methods.items():
        for ratio in EVICT_RATIOS:
            cache_copy = copy.deepcopy(cache)

            if n_middle > 0:
                n_keep = max(1, int(n_middle * (1 - ratio)))

                for li in range(n_layers):
                    for h in range(n_kv_heads):
                        s = scores[li, h, middle_start:middle_end]
                        _, topk_idx = s.topk(min(n_keep, len(s)))

                        # Build eviction mask for middle tokens
                        keep_mid = torch.zeros(n_middle, dtype=torch.bool)
                        keep_mid[topk_idx] = True
                        evict_positions = (~keep_mid).nonzero(as_tuple=True)[0] + middle_start
                        evict_positions = evict_positions.to(device)

                        cache_copy.layers[li].keys[0, h, evict_positions] = 0
                        cache_copy.layers[li].values[0, h, evict_positions] = 0

            gen_text = generate_with_cache(cache_copy, out, max_new)
            correct = scorer(gen_text, answers)

            n_kept = SINK + RECENT + (min(n_keep, n_middle) if n_middle > 0 else 0)
            actual_cr = 1.0 - n_kept / L

            results.append({
                "sample_idx": int(sample_idx), "task": task,
                "method": method_name, "ratio": ratio,
                "actual_cr": round(actual_cr, 4), "correct": bool(correct),
                "gen": gen_text[:200],
            })

            del cache_copy
            torch.cuda.empty_cache()

    del cache
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    eta = elapsed / (idx_i + 1) * (len(selected) - idx_i - 1)
    status = "OK" if full_correct else "FAIL(fullkv)"
    print(f"[{idx_i+1}/{len(selected)}] {task:>20} fullkv={status}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

    # Running summary every 25 samples
    if (idx_i + 1) % 25 == 0:
        print(f"\n--- Progress: {idx_i+1}/{len(selected)} ---")
        for method in ["snapkv", "qa", "recons"]:
            row = []
            for ratio in EVICT_RATIOS:
                mr = [r for r in results
                      if r["method"] == method and r["ratio"] == ratio]
                if mr:
                    acc = sum(r["correct"] for r in mr) / len(mr) * 100
                    row.append(f"{ratio}:{acc:4.0f}%")
            print(f"  {method:>8}: {' | '.join(row)}")
        print()

        # Save intermediate results
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  (intermediate save to {args.output})")

# ============================================================
# Final Summary
# ============================================================
print(f"\n{'='*90}")
print(f"FINAL RESULTS: Accuracy by Method × Eviction Ratio")
print(f"{'='*90}")
print(f"{'Method':>10}", end="")
for ratio in EVICT_RATIOS:
    print(f" {'CR='+str(ratio):>10}", end="")
print()
print("-" * (10 + 11 * len(EVICT_RATIOS)))

for method in ["full_kv", "snapkv", "qa", "recons"]:
    print(f"  {method:>8}", end="")
    if method == "full_kv":
        mr = [r for r in results if r["method"] == "full_kv"]
        acc = sum(r["correct"] for r in mr) / len(mr) * 100
        print(f"  {acc:.1f}% (baseline)")
        continue
    for ratio in EVICT_RATIOS:
        mr = [r for r in results if r["method"] == method and r["ratio"] == ratio]
        if mr:
            acc = sum(r["correct"] for r in mr) / len(mr) * 100
            print(f" {acc:>9.1f}%", end="")
    print()

# Per-task breakdown
print(f"\n{'='*90}")
print(f"PER-TASK BREAKDOWN")
print(f"{'='*90}")

for task in sorted(set(r["task"] for r in results)):
    fk = [r for r in results if r["task"] == task and r["method"] == "full_kv"]
    fk_acc = sum(r["correct"] for r in fk) / len(fk) * 100 if fk else 0

    print(f"\n  {task} (FullKV={fk_acc:.0f}%):")
    for method in ["snapkv", "qa", "recons"]:
        row = []
        for ratio in EVICT_RATIOS:
            mr = [r for r in results
                  if r["task"] == task and r["method"] == method and r["ratio"] == ratio]
            if mr:
                acc = sum(r["correct"] for r in mr) / len(mr) * 100
                row.append(f"{acc:4.0f}%")
            else:
                row.append("  N/A")
        print(f"    {method:>8}: {' | '.join(row)}")

# Save final results
os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
with open(args.output, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {args.output}")
print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")
print("DONE")
