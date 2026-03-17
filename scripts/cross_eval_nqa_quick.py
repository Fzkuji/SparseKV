"""
Quick cross-eval on LongBench NarrativeQA: only cr=0.9, only QA signals (skip recon).
Recon baseline already known: fastkvzip=22.75, no_press=28.91.

Usage:
    MODEL=/home/models/Qwen3-8B python -u scripts/cross_eval_nqa_quick.py
"""
import torch
import json
import os
import gc
import time
import random
import re
import string
from collections import Counter
from datasets import load_dataset
from transformers import (
    pipeline as hf_pipeline,
)
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from kvpress.presses.base_press import BasePress

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-8B"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "200"))
OUTPUT = os.environ.get("OUTPUT", "results/preliminary/cross_eval/cross_eval_nqa_qa_signals_cr090.json")
CR = 0.90
N_SINK = 4

print(f"Model: {MODEL}")
print(f"N_SAMPLES: {N_SAMPLES}")
print(f"CR: {CR}")

# --- F1 metric ---
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_tokens) if pred_tokens else 0
    r = num_same / len(gt_tokens) if gt_tokens else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def best_f1(prediction, answers):
    if isinstance(answers, str):
        answers = [answers]
    return max(f1_score(prediction, a) for a in answers)

# --- Model ---
pipe = hf_pipeline(
    "kv-press-text-generation",
    model=MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = pipe.tokenizer
model = pipe.model

# --- Data ---
ds = load_dataset("THUDM/LongBench", "narrativeqa", split="test", trust_remote_code=True)
print(f"Dataset size: {len(ds)}")
random.seed(42)
selected = sorted(random.sample(range(len(ds)), min(N_SAMPLES, len(ds))))
print(f"Selected {len(selected)} samples")

# --- QA scoring ---
def compute_qa_scores(model, tokenizer, context, question, answer, n_sink=4, use_answer=True, answer_only=False):
    if answer_only:
        use_answer = True
    user_msg = f"{context}\n\n{question}"
    if use_answer:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    else:
        messages = [{"role": "user", "content": user_msg}]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    ctx_msg = [{"role": "user", "content": context + "####SPLIT####"}]
    ctx_text = tokenizer.apply_chat_template(ctx_msg, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ctx_part = ctx_text.split("####SPLIT####")[0]
    ctx_ids = tokenizer.encode(ctx_part, return_tensors="pt", add_special_tokens=False)
    context_len = ctx_ids.shape[1]

    if answer_only:
        cq_msg = [{"role": "user", "content": user_msg}]
        cq_text = tokenizer.apply_chat_template(cq_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        cq_ids = tokenizer.encode(cq_text, return_tensors="pt", add_special_tokens=False)
        cq_len = cq_ids.shape[1]

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

    scores = torch.zeros(n_layers, 1, n_kv_heads, context_len, dtype=torch.float32, device=model.device)

    def make_attn_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                return output
            bsz, total, hid = hidden_states.shape
            q = module.q_proj(hidden_states)
            k = module.k_proj(hidden_states)
            q = q.view(bsz, total, n_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, total, n_kv_heads, head_dim).transpose(1, 2)
            position_ids = kwargs.get("position_ids", None)
            if position_ids is None:
                position_ids = torch.arange(total, device=hidden_states.device).unsqueeze(0)
            position_embeddings = kwargs.get("position_embeddings", None)
            if position_embeddings is None:
                position_embeddings = module.rotary_emb(k, position_ids)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if answer_only:
                q_qa = q[:, :, cq_len:]
            else:
                q_qa = q[:, :, context_len:]
            k_ctx = k[:, :, :context_len]
            k_ctx_expanded = k_ctx.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)
            k_ctx_expanded = k_ctx_expanded.reshape(bsz, n_heads, context_len, head_dim)
            scale = head_dim ** -0.5
            attn_scores = torch.matmul(q_qa, k_ctx_expanded.transpose(-2, -1)) * scale
            attn_scores = attn_scores.view(bsz, n_kv_heads, n_groups, -1, context_len)
            layer_scores = attn_scores.amax(dim=(2, 3))
            scores[layer_idx, 0] = layer_scores.float()
            return output
        return hook_fn

    hooks = []
    try:
        for i, layer in enumerate(model.model.layers):
            h = layer.self_attn.register_forward_hook(make_attn_hook(i), with_kwargs=True)
            hooks.append(h)
        with torch.no_grad():
            model.model(input_ids=full_ids)
    finally:
        for h in hooks:
            h.remove()
    return scores


class QASignalPress(BasePress):
    compression_ratio: float = 0.0
    def __init__(self, compression_ratio, qa_scores):
        self.compression_ratio = compression_ratio
        self.qa_scores = qa_scores
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        layer_idx = int(module.layer_idx)
        bsz, n_kv_heads, seq_len, head_dim = keys.shape
        dev = keys.device
        layer_scores = self.qa_scores[layer_idx].to(dev)
        if layer_scores.shape[-1] > seq_len:
            layer_scores = layer_scores[:, :, :seq_len]
        elif layer_scores.shape[-1] < seq_len:
            pad = torch.zeros(1, n_kv_heads, seq_len - layer_scores.shape[-1], device=dev, dtype=layer_scores.dtype)
            layer_scores = torch.cat([layer_scores, pad], dim=-1)
        n_keep = max(N_SINK, int(seq_len * (1 - self.compression_ratio)))
        layer_scores = layer_scores.clone()
        layer_scores[:, :, :N_SINK] = float('inf')
        _, keep_idx = layer_scores.topk(n_keep, dim=-1)
        keep_idx = keep_idx.sort(dim=-1).values
        keys = keys.gather(2, keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        values = values.gather(2, keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        return keys, values


# --- Main ---
results = []
t0 = time.time()

for idx_i, sample_idx in enumerate(selected):
    ex = ds[sample_idx]
    context = ex["context"]
    question = ex["input"]
    answers = json.loads(ex["answers"]) if isinstance(ex["answers"], str) else ex["answers"]
    answer_for_scoring = str(answers[0]) if answers else ""

    # Compute all three QA scores (one forward each)
    scores_dict = {}
    for name, kwargs in [
        ("qa_oracle", {"use_answer": True, "answer_only": False}),
        ("question_only", {"use_answer": False, "answer_only": False}),
        ("answer_only", {"use_answer": True, "answer_only": True}),
    ]:
        try:
            s = compute_qa_scores(model, tokenizer, context, question, answer_for_scoring, **kwargs)
            scores_dict[name] = s
        except Exception as e:
            print(f"  [{name}] scoring failed: {e}")
            scores_dict[name] = None
        torch.cuda.empty_cache(); gc.collect()

    # Eval each signal at cr=0.9
    for signal_name, qa_scores in scores_dict.items():
        if qa_scores is not None:
            press = QASignalPress(compression_ratio=CR, qa_scores=qa_scores)
            try:
                out = pipe(context, question=question, press=press, max_new_tokens=128, do_sample=False)
                gen = out["answer"]
            except Exception as e:
                gen = f"ERROR: {e}"
        else:
            gen = "ERROR: no scores"
        f1 = best_f1(gen, answers)
        results.append({
            "sample_idx": int(sample_idx), "task": "narrativeqa",
            "scoring": signal_name, "ratio": CR,
            "f1": round(f1 * 100, 2), "gen": gen[:200],
        })

    torch.cuda.empty_cache(); gc.collect()

    elapsed = time.time() - t0
    eta = elapsed / (idx_i + 1) * (len(selected) - idx_i - 1)
    print(f"[{idx_i+1}/{len(selected)}] elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m")

    if (idx_i + 1) % 10 == 0:
        print(f"\n--- Progress ({idx_i+1}/{len(selected)}) ---")
        for s in ["qa_oracle", "question_only", "answer_only"]:
            subset = [r for r in results if r["scoring"] == s]
            if subset:
                avg = sum(r["f1"] for r in subset) / len(subset)
                print(f"  {s:15s} cr={CR}: avg F1 = {avg:.2f}")
        print()

        os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)

# Final save
os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - t0
print(f"\n=== DONE in {elapsed/60:.1f} min ===")
for s in ["qa_oracle", "question_only", "answer_only"]:
    subset = [r for r in results if r["scoring"] == s]
    if subset:
        avg = sum(r["f1"] for r in subset) / len(subset)
        print(f"  {s:15s} cr={CR}: avg F1 = {avg:.2f}")
print(f"\nKnown baselines: no_press=28.91, fastkvzip=22.75")
