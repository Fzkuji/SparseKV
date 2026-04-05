"""
QASignalPress: KVzipPress variant using QA attention instead of reconstruction.

Uses kvpress pipeline for correct prefill → compress → generate flow.

Usage:
    python3 -u scripts/qa_signal_press.py --model Qwen/Qwen3-8B --n_samples 100
"""
import torch
import json
import os
import gc
import time
import math
import random
import logging
import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, DynamicCache

from kvpress import KVzipPress
from kvpress.presses.base_press import SUPPORTED_MODELS
from kvpress.attention_patch import patch_attention_functions
from kvpress.utils import extract_keys_and_values, get_prerope_query_states
from transformers.models.qwen3.modeling_qwen3 import rotate_half

patch_attention_functions()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--n_samples", type=int, default=100)
parser.add_argument("--output", default="results/qa_signal_ruler4096.json")
parser.add_argument("--dataset_dir", default="4096")
args = parser.parse_args()

RATIOS = [0.30, 0.50, 0.70, 0.90, 0.95]


@dataclass
class QASignalPress(KVzipPress):
    """
    Like KVzipPress but uses QA attention instead of reconstruction attention.
    """
    gt_answer_text: str = ""
    
    def _perform_kvzip_compression(self, model, tokenizer):
        """
        Override: use QA attention instead of reconstruction.
        Feed answer tokens, collect attention to context, then compress.
        """
        self.context_length = self._context_ids.shape[1]
        ctx_len = self.context_length
        
        # Initialize scores
        self.score_val = torch.zeros(
            (model.config.num_hidden_layers, 1, model.config.num_key_value_heads, ctx_len),
            dtype=model.dtype, device=model.device,
        )
        self.score_val[..., :self.n_sink] = 1.0
        
        # Encode answer
        answer_ids = tokenizer.encode(self.gt_answer_text, return_tensors="pt", add_special_tokens=False)
        answer_ids = answer_ids.to(model.device)
        
        # Score all context positions at once
        self.start_idx = self.prefix_length
        self.end_idx = ctx_len
        
        # Forward answer tokens through cached model
        model(
            input_ids=answer_ids,
            past_key_values=self._cache,
            num_logits_to_keep=1,
        )
        
        # Compress based on scores
        self.compress_post(model)


# ============================================================
# Task scoring
# ============================================================
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


def _decode(model, tokenizer, cache, logits, max_new_tokens):
    """Greedy decode from logits using existing cache."""
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


def main():
    print(f"Model: {args.model}")
    print(f"Ratios: {RATIOS}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    
    ds = load_dataset("simonjegou/ruler", args.dataset_dir, split="test")
    random.seed(42)
    task_samples = defaultdict(list)
    for i, ex in enumerate(ds):
        task_samples[ex["task"]].append(i)
    
    selected = []
    per_task = max(1, args.n_samples // len(task_samples))
    for task, indices in sorted(task_samples.items()):
        n = min(per_task, len(indices))
        selected.extend(random.sample(indices, n))
    selected.sort()
    print(f"Selected {len(selected)} samples from {len(task_samples)} tasks")
    
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
        
        # Build context-only and question-only inputs
        # Context = everything up to the question
        context_text = f"{context}\n\n"
        question_text = f"{question}\n{answer_prefix}"
        
        # Full prompt for chat template
        full_prompt = context_text + question_text
        messages = [{"role": "user", "content": full_prompt}]
        full_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        full_ids = tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        
        # Context-only for chat template (to find boundary)
        ctx_messages = [{"role": "user", "content": context_text}]
        ctx_text = tokenizer.apply_chat_template(ctx_messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ctx_ids = tokenizer.encode(ctx_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        context_len = ctx_ids.shape[1]
        
        # Full KV baseline: prefill + generate
        cache_full = DynamicCache()
        with torch.no_grad():
            out_full = model(input_ids=full_ids, past_key_values=cache_full, use_cache=True)
        gen_full = _decode(model, tokenizer, cache_full, out_full.logits, max_new)
        full_correct = scorer(gen_full, answers)
        del cache_full
        torch.cuda.empty_cache()
        
        results.append({
            "sample_idx": int(sample_idx), "task": task,
            "method": "full_kv", "ratio": 0.0, "correct": bool(full_correct),
            "gen": gen_full[:200],
        })
        
        # QA Signal at various ratios
        for ratio in RATIOS:
            press = QASignalPress(compression_ratio=ratio)
            press.gt_answer_text = gt_answer
            
            try:
                # Step 1: Prefill with press (compression happens on exit)
                cache = DynamicCache()
                with press(model):
                    model.model(input_ids=full_ids, past_key_values=cache)
                # Cache is now compressed (evicted positions masked via attention_patch)
                
                # Step 2: Get logits by re-forwarding last token
                # Pop last KV entry so we can re-forward it
                cache_len = cache.get_seq_length()
                for li in range(len(cache)):
                    cache.layers[li].keys = cache.layers[li].keys[:, :, :cache_len-1]
                    cache.layers[li].values = cache.layers[li].values[:, :, :cache_len-1]
                
                last_token = full_ids[:, -1:]
                with torch.no_grad():
                    out = model(input_ids=last_token, past_key_values=cache, use_cache=True)
                
                gen_text = _decode(model, tokenizer, cache, out.logits, max_new)
                correct = scorer(gen_text, answers)
                
            except Exception as e:
                import traceback
                print(f"  ERROR ratio={ratio}: {e}")
                traceback.print_exc()
                gen_text = f"ERROR: {e}"
                correct = False
            
            results.append({
                "sample_idx": int(sample_idx), "task": task,
                "method": "qa_signal", "ratio": ratio,
                "correct": bool(correct), "gen": str(gen_text)[:200],
            })
            del cache
            torch.cuda.empty_cache()
            gc.collect()
        
        elapsed = time.time() - t0
        eta = elapsed / (idx_i + 1) * (len(selected) - idx_i - 1)
        status = "OK" if full_correct else "FAIL"
        print(f"[{idx_i+1}/{len(selected)}] {task:>20} fullkv={status}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")
        
        if (idx_i + 1) % 20 == 0 or idx_i == len(selected) - 1:
            _print_summary(results, idx_i + 1, len(selected), final=(idx_i == len(selected) - 1))
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
    
    print(f"\nSaved to {args.output}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")


def _print_summary(results, done, total, final=False):
    tencent_kvzip = {0.30: 95.23, 0.50: 95.21, 0.70: 95.15, 0.90: 87.22, 0.95: 37.65}
    header = "FINAL" if final else f"Progress {done}/{total}"
    print(f"\n{'='*60}")
    print(header)
    fk = [r for r in results if r["method"] == "full_kv"]
    fk_acc = sum(r["correct"] for r in fk) / len(fk) * 100
    print(f"Full KV: {fk_acc:.1f}% ({len(fk)} samples)")
    print(f"  Ratio | QA Signal | KVzip(ref)")
    for ratio in RATIOS:
        mr = [r for r in results if r["method"] == "qa_signal" and r["ratio"] == ratio]
        acc = sum(r["correct"] for r in mr) / len(mr) * 100 if mr else 0
        kz = tencent_kvzip.get(ratio, "-")
        print(f"  {ratio:.2f}  | {acc:>8.1f}% | {kz:>9}%")
    
    if final:
        print(f"\nPER-TASK:")
        sep = " | "
        for task in sorted(set(r["task"] for r in results)):
            fk_t = [r for r in results if r["task"] == task and r["method"] == "full_kv"]
            fk_a = sum(r["correct"] for r in fk_t) / len(fk_t) * 100
            parts = []
            for ratio in RATIOS:
                mr = [r for r in results if r["task"] == task and r["method"] == "qa_signal" and r["ratio"] == ratio]
                acc = sum(r["correct"] for r in mr) / len(mr) * 100 if mr else 0
                parts.append(f"{ratio}={acc:.0f}%")
            joined = sep.join(parts)
            print(f"  {task:>20} (Full={fk_a:.0f}%): {joined}")


if __name__ == "__main__":
    main()
