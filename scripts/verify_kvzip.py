"""
Verify kvzip results on AML match the Tencent server results.
Uses kvpress KVPressTextGenerationPipeline with bf16.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 -u scripts/verify_kvzip.py
"""
import torch
import json
import os
import gc
import time
import random
from collections import defaultdict
from datasets import load_dataset
from transformers import pipeline as hf_pipeline
from kvpress import KVzipPress

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "100"))
OUTPUT = os.environ.get("OUTPUT", "results/verify_kvzip_ruler4096.json")

RATIOS = [0.30, 0.50, 0.70, 0.90, 0.95]

print(f"Model: {MODEL}")
print(f"Ratios: {RATIOS}")
print(f"N_SAMPLES: {N_SAMPLES}")

# Use kvpress pipeline with bf16 (fp16 causes overflow in search_hyperplane)
pipe = hf_pipeline(
    "kv-press-text-generation",
    model=MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = pipe.tokenizer
model = pipe.model

# Load RULER
ds = load_dataset("simonjegou/ruler", "4096", split="test")
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

# Scoring
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
    scorer = TASK_SCORERS.get(task, string_match_all)
    
    # Full KV baseline — pipeline with context/question separated
    try:
        out_full = pipe(context, question=question, answer_prefix=answer_prefix,
                       max_new_tokens=max_new, do_sample=False)
        gen_full = out_full["answer"]
    except Exception as e:
        print(f"  FULLKV ERROR: {e}")
        gen_full = f"ERROR: {e}"
    full_correct = scorer(gen_full, answers)
    results.append({
        "sample_idx": int(sample_idx), "task": task,
        "method": "full_kv", "ratio": 0.0, "correct": bool(full_correct),
        "gen": gen_full[:200],
    })
    
    # KVzip at various ratios
    for ratio in RATIOS:
        press = KVzipPress(compression_ratio=ratio)
        try:
            out = pipe(context, question=question, answer_prefix=answer_prefix,
                      press=press, max_new_tokens=max_new, do_sample=False)
            gen_text = out["answer"]
            correct = scorer(gen_text, answers)
        except Exception as e:
            print(f"  ERROR at ratio={ratio}: {e}")
            gen_text = f"ERROR: {e}"
            correct = False
        
        results.append({
            "sample_idx": int(sample_idx), "task": task,
            "method": "kvzip", "ratio": ratio,
            "correct": bool(correct), "gen": str(gen_text)[:200],
        })
        
        torch.cuda.empty_cache()
        gc.collect()
    
    elapsed = time.time() - t0
    eta = elapsed / (idx_i + 1) * (len(selected) - idx_i - 1)
    status = "OK" if full_correct else "FAIL"
    print(f"[{idx_i+1}/{len(selected)}] {task:>20} fullkv={status}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")
    
    if (idx_i + 1) % 20 == 0:
        print(f"\n--- Progress: {idx_i+1}/{len(selected)} ---")
        fk = [r for r in results if r["method"] == "full_kv"]
        fk_acc = sum(r["correct"] for r in fk) / len(fk) * 100
        print(f"  Full KV: {fk_acc:.1f}%")
        for ratio in RATIOS:
            mr = [r for r in results if r["method"] == "kvzip" and r["ratio"] == ratio]
            if mr:
                acc = sum(r["correct"] for r in mr) / len(mr) * 100
                print(f"  KVzip CR={ratio}: {acc:.1f}%")
        print()
        os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)

# Final
print(f"\n{'='*60}")
print(f"FINAL: KVzip on RULER 4096 (AML verification)")
print(f"{'='*60}")
fk = [r for r in results if r["method"] == "full_kv"]
fk_acc = sum(r["correct"] for r in fk) / len(fk) * 100
print(f"Full KV: {fk_acc:.1f}% ({len(fk)} samples)")

# Reference from Tencent server
tencent_ref = {0.30: 95.23, 0.50: 95.21, 0.70: 95.15, 0.90: 87.22, 0.95: 37.65}
print(f"{'Ratio':>8} | {'AML':>8} | {'Tencent(ref)':>12}")
print("-" * 35)
for ratio in RATIOS:
    mr = [r for r in results if r["method"] == "kvzip" and r["ratio"] == ratio]
    acc = sum(r["correct"] for r in mr) / len(mr) * 100 if mr else 0
    ref = tencent_ref.get(ratio, "-")
    print(f"{ratio:>8.2f} | {acc:>7.1f}% | {ref:>11}%")

os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT}")
print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")
