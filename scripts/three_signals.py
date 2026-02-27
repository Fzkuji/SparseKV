"""
Compare 2 KV importance scoring methods across eviction ratios on RULER.
- QA: attention from question+answer_prefix tokens to context (max over queries & groups)
- Recons: KVzip-style reconstruction — append "Repeat the previous context exactly",
  do chunked forward passes, collect max attention to each context KV position.

Both use `amax(dim=(-3,-2))` following Fast KVzip (arXiv 2601.17668).

Usage:
    python -u scripts/three_signals.py [--model MODEL] [--n_samples N] [--output PATH]
"""
import torch, numpy as np, json, random, copy, os, argparse, time, math
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--n_samples", type=int, default=100, help="Samples per task (0=all)")
parser.add_argument("--output", default="results/two_signals.json")
parser.add_argument("--dataset_dir", default="4096", help="RULER data_dir")
parser.add_argument("--recons_chunk", type=int, default=2000, help="Reconstruction chunk size")
args = parser.parse_args()

MODEL = args.model
EVICT_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
SINK = 4
RECENT = 64
RECONS_CHUNK = args.recons_chunk

print(f"Model: {MODEL}")
print(f"Eviction ratios: {EVICT_RATIOS}")
print(f"Target samples: {args.n_samples}")
print(f"Reconstruction chunk size: {RECONS_CHUNK}")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda",
                                              attn_implementation="eager")
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

selected = []
n_tasks = len(task_samples)
per_task = max(1, args.n_samples // n_tasks) if args.n_samples > 0 else None
for task, indices in sorted(task_samples.items()):
    n = min(per_task, len(indices)) if per_task else len(indices)
    selected.extend(random.sample(indices, n))
selected.sort()
print(f"Selected {len(selected)} samples from {n_tasks} tasks ({per_task} per task)")


# Scoring functions
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
    """Greedy generation from existing cache."""
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


def compute_attention_scores(input_ids, position_ids=None, past_key_values=None,
                             target_range=None):
    """
    Run a forward pass and collect attention scores for context KV positions.
    
    Args:
        input_ids: [1, seq_len] query tokens
        position_ids: [1, seq_len] position ids (for correct RoPE with cache)
        past_key_values: DynamicCache with context KVs
        target_range: (start, end) — which KV positions to score (in the cache)
    
    Returns:
        scores: [n_layers, n_kv_heads, target_len] — max attention from query tokens
                to each target KV position, max over query positions and groups.
    """
    # We need eager attention to get attention weights
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
        )
    
    t_start, t_end = target_range
    target_len = t_end - t_start
    scores = torch.zeros(n_layers, n_kv_heads, target_len)
    
    for li in range(n_layers):
        # attn_weights shape: [bsz, n_q_heads, q_len, kv_len]
        attn = out.attentions[li]  # [1, n_q_heads, q_len, total_kv_len]
        # Reshape to [bsz, n_kv_heads, n_groups, q_len, kv_len]
        attn = attn.view(1, n_kv_heads, n_groups, attn.shape[2], attn.shape[3])
        # Extract target range
        attn_target = attn[:, :, :, :, t_start:t_end]  # [1, n_kv_heads, n_groups, q_len, target_len]
        # Max over groups and query positions → [1, n_kv_heads, target_len]
        layer_scores = attn_target.amax(dim=(-3, -2))  # max over (groups, q_len)
        scores[li] = layer_scores[0].cpu()
    
    return scores


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

    # Build full prompt (context + question)
    prompt = f"{context}\n\n{question}\n{answer_prefix}"
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(full_text, return_tensors="pt",
                                  add_special_tokens=False).to(device)
    L = prompt_ids.shape[1]

    # Find context boundary (everything before question)
    q_text = f"\n\n{question}\n{answer_prefix}"
    q_only_ids = tokenizer.encode(q_text, add_special_tokens=False)
    q_len = len(q_only_ids)
    ctx_end = L - q_len  # context tokens: [0, ctx_end)

    # ================================================================
    # Step 1: Prefill — get full cache + QA scores via output_attentions
    # ================================================================
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True, output_attentions=True,
                    past_key_values=DynamicCache())

    # Extract QA scores: attention from question tokens to context positions
    # Question tokens are the last q_len tokens of the prompt
    # Target: context KV positions [SINK, ctx_end)
    middle_start = SINK
    middle_end = max(SINK, min(ctx_end, L - RECENT))
    n_middle = middle_end - middle_start

    qa_scores = torch.zeros(n_layers, n_kv_heads, L)
    for li in range(n_layers):
        attn = out.attentions[li]  # [1, n_q_heads, L, L]
        # Question tokens attend to all positions
        # attn[:, :, -q_len:, :] = attention FROM question tokens TO all KV positions
        qa_attn = attn[:, :, -q_len:, :]  # [1, n_q_heads, q_len, L]
        qa_attn = qa_attn.view(1, n_kv_heads, n_groups, q_len, L)
        # Max over groups and query positions
        qa_scores_layer = qa_attn.amax(dim=(-3, -2))  # [1, L]
        qa_scores[li] = qa_scores_layer[0].cpu()

    # Get cache from the output (without attentions stored)
    cache = out.past_key_values
    
    # Free attention tensors
    del out.attentions
    torch.cuda.empty_cache()

    # ================================================================
    # Step 2: Full KV baseline generation
    # ================================================================
    cache_full = copy.deepcopy(cache)
    gen_full = generate_with_cache(cache_full, out, max_new)
    full_correct = scorer(gen_full, answers)
    del cache_full
    results.append({
        "sample_idx": int(sample_idx), "task": task,
        "method": "full_kv", "ratio": 0.0, "correct": bool(full_correct),
        "gen": gen_full[:200],
    })

    # ================================================================
    # Step 3: Reconstruction scoring (KVzip-style)
    # Build "Repeat the previous context exactly" query, chunked forward
    # ================================================================
    recons_scores = torch.zeros(n_layers, n_kv_heads, L)
    
    # Build context token ids for chunking (just the context portion)
    # Context is prompt_ids[0, :ctx_end]
    ctx_ids = prompt_ids[0, :ctx_end]  # [ctx_end]
    
    # Chunk the context
    chunk_size = RECONS_CHUNK
    n_chunks = max(1, (ctx_end + chunk_size - 1) // chunk_size)
    
    for ci in range(n_chunks):
        c_start = ci * chunk_size
        c_end = min((ci + 1) * chunk_size, ctx_end)
        
        # Build repeat query
        if ci == 0:
            repeat_prompt = "\n\nRepeat the previous context exactly."
        else:
            # Include last 8 tokens of previous chunk as hint
            prev_end = ci * chunk_size
            prev_start = max(0, prev_end - 8)
            hint_ids = prompt_ids[0, prev_start:prev_end]
            hint_text = tokenizer.decode(hint_ids)
            repeat_prompt = f"\n\nRepeat the part of the previous context exactly, starting with {hint_text}"
        
        repeat_ids = tokenizer.encode(repeat_prompt, return_tensors="pt",
                                       add_special_tokens=False).to(device)
        
        # The chunk's ground-truth tokens (what should be repeated)
        chunk_ids = prompt_ids[:, c_start:c_end]  # [1, chunk_len]
        
        # Concatenate: [repeat_query, chunk_answer]
        # This simulates the model being asked to repeat, then "generating" the chunk
        query_ids = torch.cat([repeat_ids, chunk_ids], dim=1)  # [1, repeat_len + chunk_len]
        q_total_len = query_ids.shape[1]
        
        # We need to run this query against the existing KV cache (context)
        # Position ids continue from where the cache ends
        cache_for_recons = copy.deepcopy(cache)
        pos_start = cache_for_recons.get_seq_length()
        position_ids = torch.arange(pos_start, pos_start + q_total_len,
                                    device=device).unsqueeze(0)
        
        with torch.no_grad():
            recons_out = model(
                input_ids=query_ids,
                position_ids=position_ids,
                past_key_values=cache_for_recons,
                use_cache=False,  # don't need to extend cache
                output_attentions=True,
            )
        
        # Extract scores: attention from query tokens to context KV positions [c_start, c_end)
        for li in range(n_layers):
            attn = recons_out.attentions[li]  # [1, n_q_heads, q_total_len, cache_len + q_total_len]
            # We want attention to the original cache positions [c_start, c_end)
            attn_chunk = attn[:, :, :, c_start:c_end]  # [1, n_q_heads, q_total_len, chunk_len]
            attn_chunk = attn_chunk.view(1, n_kv_heads, n_groups, q_total_len, c_end - c_start)
            # Max over groups and query positions
            # attn_chunk: [1, n_kv_heads, n_groups, q_total_len, chunk_len]
            # amax over dim -3 (groups) and -2 (q_positions) → [1, n_kv_heads, chunk_len]
            chunk_scores = attn_chunk.amax(dim=(-3, -2))  # [1, n_kv_heads, chunk_len]
            recons_scores[li, :, c_start:c_end] = torch.max(
                recons_scores[li, :, c_start:c_end],
                chunk_scores[0].cpu()  # [n_kv_heads, chunk_len]
            )
        
        del recons_out, cache_for_recons
        torch.cuda.empty_cache()

    # ================================================================
    # Step 4: Evict and evaluate for each method × ratio
    # ================================================================
    methods = {"qa": qa_scores, "recons": recons_scores}
    for method_name, scores in methods.items():
        for ratio in EVICT_RATIOS:
            cache_copy = copy.deepcopy(cache)

            if n_middle > 0:
                n_keep = max(1, int(n_middle * (1 - ratio)))

                for li in range(n_layers):
                    for h in range(n_kv_heads):
                        s = scores[li, h, middle_start:middle_end]
                        _, topk_idx = s.topk(min(n_keep, len(s)))

                        keep_mid = torch.zeros(n_middle, dtype=torch.bool)
                        keep_mid[topk_idx] = True
                        evict_positions = (~keep_mid).nonzero(as_tuple=True)[0] + middle_start
                        evict_positions = evict_positions.to(device)

                        cache_copy.key_cache[li][0, h, evict_positions] = 0
                        cache_copy.value_cache[li][0, h, evict_positions] = 0

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

    # Running summary every 10 samples
    if (idx_i + 1) % 10 == 0:
        print(f"\n--- Progress: {idx_i+1}/{len(selected)} ---")
        for method in ["qa", "recons"]:
            row = []
            for ratio in EVICT_RATIOS:
                mr = [r for r in results
                      if r["method"] == method and r["ratio"] == ratio]
                if mr:
                    acc = sum(r["correct"] for r in mr) / len(mr) * 100
                    row.append(f"{ratio}:{acc:4.0f}%")
            print(f"  {method:>8}: {' | '.join(row)}")
        print()

        # Save intermediate
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

for method in ["full_kv", "qa", "recons"]:
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
    for method in ["qa", "recons"]:
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

# Save
os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
with open(args.output, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {args.output}")
print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")
print("DONE")
