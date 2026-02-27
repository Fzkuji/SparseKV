"""
Evaluate KVzip reconstruction scoring using kvpress's KVzipPress.
Runs each RULER sample at different compression ratios.

Usage:
    python -u scripts/two_signals_kvpress.py [--model MODEL] [--n_samples N] [--output PATH]
"""
import torch, json, random, os, argparse, time
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress import KVPressTextGenerationPipeline
from kvpress.presses import KVzipPress

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--n_samples", type=int, default=100)
parser.add_argument("--output", default="results/kvzip_ruler.json")
parser.add_argument("--dataset_dir", default="4096")
args = parser.parse_args()

MODEL = args.model
EVICT_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

print(f"Model: {MODEL}")
print(f"Eviction ratios: {EVICT_RATIOS}")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda",
                                              attn_implementation="sdpa")
model.eval()

pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)

# Load RULER
print("Loading RULER dataset...")
ds = load_dataset("simonjegou/ruler", args.dataset_dir, split="test")
random.seed(42)
task_samples = defaultdict(list)
for i, ex in enumerate(ds):
    task_samples[ex["task"]].append(i)

selected = []
n_tasks = len(task_samples)
per_task = max(1, args.n_samples // n_tasks) if args.n_samples > 0 else None
for task, indices in sorted(task_samples.items()):
    n = min(per_task, len(indices)) if per_task else len(indices)
    selected.extend(random.sample(indices, n))
selected.sort()
print(f"Selected {len(selected)} samples from {n_tasks} tasks ({per_task} per task)")


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

    # Full KV baseline
    try:
        out = pipe(context, question=question, answer_prefix=answer_prefix,
                   max_new_tokens=max_new, enable_thinking=False)
        gen_full = out["answer"] if isinstance(out, dict) else out
        full_correct = scorer(gen_full, answers)
    except Exception as e:
        print(f"  Full KV error: {e}")
        gen_full = ""
        full_correct = False

    results.append({
        "sample_idx": int(sample_idx), "task": task,
        "method": "full_kv", "ratio": 0.0, "correct": bool(full_correct),
        "gen": str(gen_full)[:200],
    })

    # KVzip at each ratio
    for ratio in EVICT_RATIOS:
        try:
            press = KVzipPress(compression_ratio=ratio)
            out = pipe(context, question=question, answer_prefix=answer_prefix,
                       press=press, max_new_tokens=max_new, enable_thinking=False)
            gen_kvzip = out["answer"] if isinstance(out, dict) else out
            kvzip_correct = scorer(gen_kvzip, answers)
        except Exception as e:
            print(f"  KVzip CR={ratio} error: {e}")
            gen_kvzip = ""
            kvzip_correct = False

        results.append({
            "sample_idx": int(sample_idx), "task": task,
            "method": "kvzip", "ratio": ratio, "correct": bool(kvzip_correct),
            "gen": str(gen_kvzip)[:200],
        })

    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    eta = elapsed / (idx_i + 1) * (len(selected) - idx_i - 1)
    status = "OK" if full_correct else "FAIL(fullkv)"
    print(f"[{idx_i+1}/{len(selected)}] {task:>20} fullkv={status}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

    if (idx_i + 1) % 10 == 0:
        print(f"\n--- Progress: {idx_i+1}/{len(selected)} ---")
        row = []
        for ratio in EVICT_RATIOS:
            mr = [r for r in results if r["method"] == "kvzip" and r["ratio"] == ratio]
            if mr:
                acc = sum(r["correct"] for r in mr) / len(mr) * 100
                row.append(f"{ratio}:{acc:4.0f}%")
        print(f"    kvzip: {' | '.join(row)}")
        print()

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


# Final Summary
print(f"\n{'='*90}")
print(f"FINAL RESULTS")
print(f"{'='*90}")
for method in ["full_kv", "kvzip"]:
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

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
with open(args.output, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {args.output}")
print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")
print("DONE")
