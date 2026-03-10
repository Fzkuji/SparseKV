#!/usr/bin/env python3
"""
SCBench evaluation script for kvpress.
Output format matches run_all.sh / eval_wrapper.py: {output_dir}/{task}__{model_tag}__{press}__{cr}/metrics.json

Supports both argparse and env vars (argparse takes priority).

Usage:
    python scripts/cross_eval_scbench.py \
        --task scbench_kv --press_name snapkv --compression_ratio 0.5 \
        --model Qwen/Qwen3-8B --model_tag Qwen--Qwen3-8B \
        --output_dir results/phase1_qwen3 --max_seq_length 170000

    # Or via env vars (backward compat):
    SCBENCH_TASK=scbench_kv PRESS_NAME=snapkv COMPRESSION_RATIO=0.5 python scripts/cross_eval_scbench.py
"""
import argparse
import gc
import json
import os
import random
import re
import string
import time
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Args ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=os.environ.get("SCBENCH_TASK", "scbench_kv"))
    p.add_argument("--press_name", default=os.environ.get("PRESS_NAME", "no_press"))
    p.add_argument("--compression_ratio", type=float,
                   default=float(os.environ.get("COMPRESSION_RATIO", "0.0")))
    p.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--model_tag", default=os.environ.get("MODEL_TAG", "Qwen--Qwen3-8B"))
    p.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR", "results/phase1_qwen3"))
    p.add_argument("--max_seq_length", type=int,
                   default=int(os.environ.get("MAX_SEQ_LENGTH", "170000")))
    p.add_argument("--n_samples", type=int,
                   default=int(os.environ.get("N_SAMPLES", "-1")))
    p.add_argument("--max_turns", type=int,
                   default=int(os.environ.get("MAX_TURNS", "-1")))
    return p.parse_args()


args = parse_args()
CR_FMT = f"{args.compression_ratio:.2f}"

# Result directory: {task}__{model_tag}__{press}__{cr}/
RESULT_NAME = f"{args.task}__{args.model_tag}__{args.press_name}__{CR_FMT}"
RESULT_DIR = os.path.join(args.output_dir, RESULT_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)

print(f"Config: MODEL={args.model}, TASK={args.task}, PRESS={args.press_name}, CR={CR_FMT}")
print(f"Result dir: {RESULT_DIR}")

# Check if already done
if os.path.exists(os.path.join(RESULT_DIR, "metrics.json")):
    print(f"Already done! {RESULT_DIR}/metrics.json exists.")
    exit(0)


# ── Max new tokens per task ─────────────────────────────────────────
SCBENCH_MAX_NEW_TOKENS = {
    "scbench_choice_eng": 40, "scbench_qa_eng": 40, "scbench_qa_chn": 40,
    "scbench_kv": 150, "scbench_kv_hard": 150, "scbench_kv_short": 150,
    "scbench_mf": 5, "scbench_mf_mid": 5,
    "scbench_prefix_suffix": 150, "scbench_prefix_suffix_short": 150,
    "scbench_passkey": 15, "scbench_vt": 30, "scbench_many_shot": 10,
    "scbench_summary": 200, "scbench_summary_with_needles": 200,
    "scbench_repoqa": 1024, "scbench_repoqa_and_kv": 1024,
}


# ── Press factory ───────────────────────────────────────────────────
def create_press(press_name, compression_ratio):
    if press_name == "no_press" or compression_ratio == 0.0:
        return None
    from kvpress import (
        SnapKVPress, StreamingLLMPress, KnormPress, RandomPress,
        TOVAPress, ExpectedAttentionPress, ObservedAttentionPress,
        CriticalKVPress,
    )
    try:
        from kvpress import FastKVzipPress
    except ImportError:
        FastKVzipPress = None

    mapping = {
        "snapkv": SnapKVPress,
        "streaming_llm": StreamingLLMPress,
        "critical_snapkv": CriticalKVPress,
        "knorm": KnormPress,
        "random": RandomPress,
        "tova": TOVAPress,
        "expected_attention": ExpectedAttentionPress,
        "observed_attention": ObservedAttentionPress,
    }
    if press_name in mapping:
        return mapping[press_name](compression_ratio=compression_ratio)
    elif press_name == "fastkvzip" and FastKVzipPress is not None:
        return FastKVzipPress(compression_ratio=compression_ratio)
    else:
        raise ValueError(f"Unknown press: {press_name}")


# ── Scoring functions (aligned with FastKVzip / MInference) ─────────
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    def replace_num(text):
        m = {"zero":"0","one":"1","two":"2","three":"3","four":"4",
             "five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
        return re.compile(r"\b(" + "|".join(m.keys()) + r")\b").sub(lambda x: m[x.group()], text)
    return replace_num(white_space_fix(remove_articles(remove_punc(s.lower()))))


def f1_score_single(pred, ref):
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0: return 0.0
    prec = common / len(p) if p else 0
    rec = common / len(r) if r else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


def include_score(pred, ref, normalize=True):
    if normalize: pred, ref = normalize_answer(pred), normalize_answer(ref)
    return 1.0 if ref in pred else 0.0


def include_score_multi(pred, ref, normalize=False):
    refs = ref.split(", ")
    if normalize: pred, refs = normalize_answer(pred), [normalize_answer(r) for r in refs]
    scores = [1.0 if r in pred else 0.0 for r in refs]
    return sum(scores) / len(scores)


def include_score_manyshot(pred, ref):
    if "(" in pred and "(" in ref:
        pa = pred.split("(")[1].split(")")[0] if "(" in pred else pred
        ra = ref.split("(")[1].split(")")[0] if "(" in ref else ref
        return 1.0 if pa == ra else 0.0
    if ref[0] == "(": ref = ref.split(")")[1].strip()
    return include_score(pred, ref)


def exact_match_score(pred, ref, normalize=False):
    if normalize: pred, ref = normalize_answer(pred), normalize_answer(ref)
    return 1.0 if pred == ref else 0.0


def rouge_score_single(pred, ref):
    try:
        from rouge import Rouge
        return Rouge().get_scores([pred], [ref], avg=True)["rouge-l"]["f"]
    except Exception:
        return f1_score_single(pred, ref)


def get_scorer(task_name):
    base = task_name.replace("_short", "").replace("_tiny", "").replace("_mid", "")
    if base in ["scbench_kv", "scbench_kv_hard", "scbench_prefix_suffix", "scbench_passkey"]:
        return lambda p, r: include_score(p, r)
    elif base == "scbench_vt":
        return lambda p, r: include_score_multi(p, r, normalize=False)
    elif base == "scbench_mf":
        return lambda p, r: exact_match_score(p, r, normalize=False)
    elif base == "scbench_many_shot":
        return lambda p, r: include_score_manyshot(p, r)
    elif base == "scbench_qa_eng":
        return lambda p, r: max(f1_score_single(p, r), include_score(p, r))
    elif base == "scbench_choice_eng":
        return lambda p, r: include_score(p.split("\n")[0], r)
    elif "summary" in base:
        return lambda p, r: rouge_score_single(p, r)
    elif "repoqa" in base:
        return lambda p, r: f1_score_single(p, r)
    else:
        return lambda p, r: include_score(p, r)


# ── Model ───────────────────────────────────────────────────────────
from kvpress import KVPressTextGenerationPipeline

_model = AutoModelForCausalLM.from_pretrained(
    args.model, device_map="auto", torch_dtype=torch.bfloat16,
)
_tokenizer = AutoTokenizer.from_pretrained(args.model)
_tokenizer.model_max_length = args.max_seq_length
print(f"Tokenizer max length: {args.max_seq_length}")

pipe = KVPressTextGenerationPipeline(model=_model, tokenizer=_tokenizer)
press = create_press(args.press_name, args.compression_ratio)
print(f"Press: {press}")


# ── Dataset ─────────────────────────────────────────────────────────
def load_scbench_samples():
    print(f"Loading Jang-Hyun/SCBench-preprocessed: {args.task}.parquet ...")
    ds = load_dataset("Jang-Hyun/SCBench-preprocessed",
                      data_files=f"{args.task}.parquet", split="train")
    print(f"Raw dataset size: {len(ds)}")

    random.seed(42)
    indices = list(range(len(ds)))
    if args.n_samples > 0:
        random.shuffle(indices)
        indices = sorted(indices[:args.n_samples])

    scorer = get_scorer(args.task)
    max_new = SCBENCH_MAX_NEW_TOKENS.get(args.task, 50)

    samples = []
    for idx in indices:
        eg = ds[idx]
        context = eg["prompts"][0]
        questions = eg["prompts"][1:]
        answers = []
        for gt in eg["ground_truth"]:
            answers.append(", ".join(str(x) for x in gt) if isinstance(gt, list) else str(gt))
        if args.max_turns > 0:
            questions = questions[:args.max_turns]
            answers = answers[:args.max_turns]
        samples.append(dict(idx=idx, task=args.task, context=context,
                            questions=questions, answers=answers,
                            max_new_tokens=max_new, scorer=scorer,
                            n_turns=len(questions)))

    total_turns = sum(s["n_turns"] for s in samples)
    print(f"Loaded {len(samples)} samples, {total_turns} total turns")
    ctx_lens = [len(_tokenizer.encode(s["context"])) for s in samples[:5]]
    print(f"Context token lengths (first 5): {ctx_lens}")
    return samples


# ── Eval ────────────────────────────────────────────────────────────
def eval_multiturn(pipe, context, questions, max_new, press=None):
    try:
        out = pipe(context, questions=questions, press=press,
                   max_new_tokens=max_new, max_context_length=args.max_seq_length,
                   do_sample=False)
        return out["answers"]
    except Exception as e:
        print(f"  ERROR: {e}")
        return [f"ERROR: {e}"] * len(questions)


# ── Main loop ───────────────────────────────────────────────────────
samples = load_scbench_samples()
predictions = []  # (pred, ref, score) per turn
results_raw = []  # detailed results
t0 = time.time()

for idx_i, sample in enumerate(samples):
    gens = eval_multiturn(pipe, sample["context"], sample["questions"],
                          sample["max_new_tokens"], press=press)
    for turn_idx, (gen, ans) in enumerate(zip(gens, sample["answers"])):
        gen_clean = gen.rstrip("</s>").strip() if gen.endswith("</s>") else gen
        score = sample["scorer"](gen_clean, ans)
        predictions.append({"pred": gen_clean[:500], "ref": str(ans)[:500], "score": score})
        results_raw.append({
            "sample_idx": int(sample["idx"]),
            "task": sample["task"],
            "turn_idx": turn_idx,
            "press": args.press_name,
            "ratio": args.compression_ratio,
            "score": float(score),
            "gen": gen[:300],
            "answer": str(ans)[:300],
        })

    sample_scores = [r["score"] for r in results_raw if r["sample_idx"] == sample["idx"]]
    avg_s = sum(sample_scores) / len(sample_scores) if sample_scores else 0
    elapsed = time.time() - t0
    eta = elapsed / (idx_i + 1) * (len(samples) - (idx_i + 1))
    print(f"[{idx_i+1}/{len(samples)}] {sample['task']} #{sample['idx']} "
          f"turns={sample['n_turns']} avg={avg_s:.2f}  "
          f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

    # Periodic save
    if (idx_i + 1) % 5 == 0 or (idx_i + 1) == len(samples):
        with open(os.path.join(RESULT_DIR, "predictions_raw.json"), "w") as f:
            json.dump(results_raw, f, indent=2, ensure_ascii=False)

# ── Save in run_all.sh-compatible format ────────────────────────────
elapsed = time.time() - t0
all_scores = [p["score"] for p in predictions]
avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

# metrics.json — the key file run_all.sh checks for completion
metrics = {
    "dataset": args.task,
    "model": args.model,
    "press": args.press_name,
    "compression_ratio": args.compression_ratio,
    "score": round(avg_score, 4),
    "num_samples": len(samples),
    "num_turns": sum(s["n_turns"] for s in samples),
    "total_time_minutes": round(elapsed / 60, 2),
}

# Per-turn breakdown
max_t = max((s["n_turns"] for s in samples), default=1)
if max_t > 1:
    per_turn = {}
    for t in range(max_t):
        turn_scores = [r["score"] for r in results_raw if r["turn_idx"] == t]
        if turn_scores:
            per_turn[f"turn_{t}"] = round(sum(turn_scores) / len(turn_scores), 4)
    metrics["per_turn"] = per_turn

with open(os.path.join(RESULT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# predictions.csv — matches eval_wrapper format
import csv
with open(os.path.join(RESULT_DIR, "predictions.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["prediction", "reference", "score"])
    for p in predictions:
        writer.writerow([p["pred"], p["ref"], p["score"]])

# Also save full raw results
with open(os.path.join(RESULT_DIR, "predictions_raw.json"), "w") as f:
    json.dump(results_raw, f, indent=2, ensure_ascii=False)

# Print summary
print(f"\n{'='*60}")
print(f"FINAL: {args.task} | {args.press_name} cr={CR_FMT} | {len(samples)} samples")
print(f"Score: {avg_score:.4f}  Time: {elapsed/60:.1f}m")
print(f"{'='*60}")
if max_t > 1 and "per_turn" in metrics:
    for k, v in metrics["per_turn"].items():
        print(f"  {k}: {v}")
print(f"\nSaved to: {RESULT_DIR}/")
