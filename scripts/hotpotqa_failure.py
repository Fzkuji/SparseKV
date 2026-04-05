#!/usr/bin/env python3
"""Analyze KVzip failure cases on LongBench hotpotqa"""

import torch, json, re
from datasets import load_dataset
from transformers import pipeline as hf_pipeline
from kvpress import KVzipPress, SnapKVPress
from kvpress.attention_patch import patch_attention_functions

patch_attention_functions()

model_path = "Qwen/Qwen3-8B"
N = 20

print("Loading model...")
pipe = hf_pipeline(
    "kv-press-text-generation",
    model=model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    model_kwargs={"attn_implementation": "flash_attention_2"},
)

print("Loading hotpotqa...")
ds = load_dataset("Xnhyacinth/LongBench", "hotpotqa", split="test")
samples = list(ds.select(range(min(N, len(ds)))))
print(f"Got {len(samples)} samples")

def f1_score(pred, ref):
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens) if pred_tokens else 0
    rec = len(common) / len(ref_tokens) if ref_tokens else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

def run_method(name, press):
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    results = []
    for i, sample in enumerate(samples):
        prompt = f"{sample['context']}\n\nQuestion: {sample['question']}\nAnswer:"
        max_tokens = sample.get("max_new_tokens", 128) or 128
        try:
            if press is not None:
                r = pipe(prompt, press=press, max_new_tokens=max_tokens)
            else:
                r = pipe(prompt, max_new_tokens=max_tokens)
            pred = r["answer"].strip()
            pred = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL).strip()
        except Exception as e:
            pred = f"ERROR: {e}"

        ans_raw = sample["answers"]
        if isinstance(ans_raw, str):
            try: ans_list = json.loads(ans_raw)
            except: ans_list = [ans_raw]
        else:
            ans_list = list(ans_raw)

        best_f1 = max(f1_score(pred, a) for a in ans_list)
        results.append({"pred": pred, "f1": best_f1, "answers": ans_list})
        print(f"  Sample {i}: f1={best_f1:.3f}  pred={pred[:60]}...")
    return results

# Run all methods
fullkv = run_method("Full KV", None)
kvzip7 = run_method("KVzip 0.7", KVzipPress(compression_ratio=0.7))
kvzip9 = run_method("KVzip 0.9", KVzipPress(compression_ratio=0.9))
snapkv7 = run_method("SnapKV 0.7", SnapKVPress(compression_ratio=0.7))

# Analysis
print(f"\n\n{'#'*70}")
print("FAILURE ANALYSIS: Cases where Full KV correct but KVzip wrong")
print(f"{'#'*70}")

for i in range(N):
    fk = fullkv[i]
    kz7 = kvzip7[i]
    kz9 = kvzip9[i]
    sk7 = snapkv7[i]

    # Show all cases with comparison
    q = samples[i]["question"]
    ctx_len = len(samples[i]["context"])
    ans = fk["answers"][0]

    status = ""
    if fk["f1"] > 0.3 and kz7["f1"] < 0.1:
        status = "★ KVzip0.7 FAIL"
    elif fk["f1"] > 0.3 and kz9["f1"] < 0.1:
        status = "★ KVzip0.9 FAIL"

    print(f"\n--- Sample {i} ({ctx_len} chars) {status} ---")
    print(f"  Q: {q}")
    print(f"  Expected: {ans}")
    print(f"  Full KV  (f1={fk['f1']:.2f}): {fk['pred'][:80]}")
    print(f"  KVzip0.7 (f1={kz7['f1']:.2f}): {kz7['pred'][:80]}")
    print(f"  KVzip0.9 (f1={kz9['f1']:.2f}): {kz9['pred'][:80]}")
    print(f"  SnapKV0.7(f1={sk7['f1']:.2f}): {sk7['pred'][:80]}")

# Summary
print(f"\n\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
avg = lambda r: sum(x["f1"] for x in r) / len(r)
print(f"  Full KV:    {avg(fullkv):.4f}")
print(f"  KVzip 0.7:  {avg(kvzip7):.4f}")
print(f"  KVzip 0.9:  {avg(kvzip9):.4f}")
print(f"  SnapKV 0.7: {avg(snapkv7):.4f}")

# Count failures
fk_ok = sum(1 for x in fullkv if x["f1"] > 0.3)
kz7_fail = sum(1 for i in range(N) if fullkv[i]["f1"] > 0.3 and kvzip7[i]["f1"] < 0.1)
kz9_fail = sum(1 for i in range(N) if fullkv[i]["f1"] > 0.3 and kvzip9[i]["f1"] < 0.1)
sk7_fail = sum(1 for i in range(N) if fullkv[i]["f1"] > 0.3 and snapkv7[i]["f1"] < 0.1)
print(f"\n  Full KV correct (f1>0.3): {fk_ok}/{N}")
print(f"  KVzip0.7 failures (among Full KV correct): {kz7_fail}/{fk_ok}")
print(f"  KVzip0.9 failures (among Full KV correct): {kz9_fail}/{fk_ok}")
print(f"  SnapKV0.7 failures (among Full KV correct): {sk7_fail}/{fk_ok}")

print("\nDONE")
