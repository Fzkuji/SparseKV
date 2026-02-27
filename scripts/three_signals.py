"""
Compare 2 KV importance scoring methods across eviction ratios on RULER.
- QA: attention from question tokens to context (max over queries & groups)
- Recons: KVzip-style reconstruction — append "Repeat the previous context exactly",
  do chunked forward passes, collect max attention to each context KV position.

Both use `amax` over (groups, query_positions) following Fast KVzip (arXiv 2601.17668).

Memory-efficient: uses hooks to compute QK attention scores one KV-group at a time,
never materializing the full attention matrix.

Usage:
    python -u scripts/three_signals.py [--model MODEL] [--n_samples N] [--output PATH]
"""
import torch, numpy as np, json, random, copy, os, argparse, time, math
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import rotate_half as _rotate_half

def apply_rotary(x, cos, sin):
    return (x * cos) + (_rotate_half(x) * sin)

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

selected = []
n_tasks = len(task_samples)
per_task = max(1, args.n_samples // n_tasks) if args.n_samples > 0 else None
for task, indices in sorted(task_samples.items()):
    n = min(per_task, len(indices)) if per_task else len(indices)
    selected.extend(random.sample(indices, n))
selected.sort()
print(f"Selected {len(selected)} samples from {n_tasks} tasks ({per_task} per task)")


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


def compute_scores_via_hooks(model, input_ids, scores_out, target_range, q_start=0,
                              position_ids=None, past_key_values=None,
                              subsampled=False, n_sink=4):
    """
    Run forward pass with hooks that compute attention scores efficiently.
    
    For each layer, we extract Q and K, compute attention scores for target KV range.
    
    Args:
        scores_out: [n_layers, n_kv_heads, target_len] tensor to accumulate max scores into
        target_range: (start, end) KV positions to score
        q_start: only use query tokens from this position onward
        subsampled: if True, use KVzip-style subsampled softmax (sink + target + query tokens only)
        n_sink: number of sink tokens to include in subsampled softmax
    """
    t_start, t_end = target_range
    hooks = []
    
    def make_hook(li):
        def hook_fn(module, args, kwargs, output):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is None:
                return
            bsz, sl, _ = hs.shape
            pe = kwargs.get("position_embeddings", None) if kwargs else None
            
            with torch.no_grad():
                q = module.q_proj(hs).view(bsz, sl, n_q_heads, head_dim).transpose(1, 2)
                k = module.k_proj(hs).view(bsz, sl, n_kv_heads, head_dim).transpose(1, 2)
                
                if pe is not None:
                    cos, sin = pe
                    q = apply_rotary(q, cos.unsqueeze(1), sin.unsqueeze(1))
                    k = apply_rotary(k, cos.unsqueeze(1), sin.unsqueeze(1))
                
                if past_key_values is not None and past_key_values.get_seq_length() > 0:
                    cached_k = past_key_values.layers[li].keys
                    q_use = q[:, :, q_start:, :]
                else:
                    cached_k = None
                    q_use = q[:, :, q_start:, :]
                
                for g in range(n_kv_heads):
                    qh = slice(g * n_groups, (g + 1) * n_groups)
                    q_g = q_use[0, qh]  # [n_groups, q_len, head_dim]
                    
                    if subsampled and cached_k is not None:
                        # KVzip-style: softmax only over sink + target_chunk + new_query_tokens
                        sink_end = min(n_sink, t_start)
                        k_sink = cached_k[0, g, :sink_end]      # [sink, head_dim]
                        k_chunk = cached_k[0, g, t_start:t_end]  # [chunk_len, head_dim]
                        k_new = k[0, g]                           # [sl, head_dim] (repeat tokens)
                        
                        k_sub = torch.cat([k_sink, k_chunk, k_new], dim=0)  # [sink+chunk+sl, hd]
                        k_sub = k_sub.unsqueeze(0).expand(n_groups, -1, -1)
                        
                        logits = torch.matmul(q_g, k_sub.transpose(-1, -2)) / scale
                        
                        # Causal mask for new tokens among themselves
                        sub_len = k_sub.shape[1]
                        q_len_use = q_g.shape[1]
                        new_start = sink_end + (t_end - t_start)  # where new tokens start in sub
                        if q_len_use > 1:
                            q_pos = torch.arange(q_len_use, device=device)
                            new_k_pos = torch.arange(sl, device=device)
                            causal_new = new_k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
                            logits[:, :, new_start:].masked_fill_(causal_new.unsqueeze(0), float('-inf'))
                        
                        attn = torch.softmax(logits.float(), dim=-1)
                        
                        # Extract target chunk portion (after sink, before new tokens)
                        attn_target = attn[:, :, sink_end:sink_end + (t_end - t_start)]
                        
                        score = attn_target.amax(dim=(0, 1))
                        scores_out[li, g] = torch.max(scores_out[li, g], score.cpu())
                        
                        del k_sub, logits, attn, attn_target
                    else:
                        # Full softmax mode (for QA scoring during prefill)
                        if cached_k is not None:
                            full_k = torch.cat([cached_k[0, g:g+1], k[0, g:g+1]], dim=1)
                        else:
                            full_k = k[0, g:g+1]
                        
                        full_k = full_k.expand(n_groups, -1, -1)
                        full_logits = torch.matmul(q_g, full_k.transpose(-1, -2)) / scale
                        
                        total_kv = full_k.shape[1]
                        q_len_use = q_g.shape[1]
                        if cached_k is None:
                            q_pos = torch.arange(q_start, sl, device=device)
                            k_pos = torch.arange(total_kv, device=device)
                            causal_full = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
                            full_logits.masked_fill_(causal_full.unsqueeze(0), float('-inf'))
                        elif q_len_use > 1:
                            cache_len = past_key_values.get_seq_length()
                            q_pos = torch.arange(q_len_use, device=device)
                            new_k_pos = torch.arange(q_len_use, device=device)
                            causal_new = new_k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
                            full_logits[:, :, cache_len:].masked_fill_(causal_new.unsqueeze(0), float('-inf'))
                        
                        attn = torch.softmax(full_logits.float(), dim=-1)
                        attn_target = attn[:, :, t_start:t_end]
                        
                        score = attn_target.amax(dim=(0, 1))
                        scores_out[li, g] = torch.max(scores_out[li, g], score.cpu())
                        
                        del full_logits, attn, attn_target, full_k
                del q, k
        return hook_fn
    
    for layer in model.model.layers:
        li = int(layer.self_attn.layer_idx)
        hooks.append(layer.self_attn.register_forward_hook(make_hook(li), with_kwargs=True))
    
    kwargs = {}
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, **kwargs)
    
    for h in hooks:
        h.remove()
    
    return out


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

    prompt = f"{context}\n\n{question}\n{answer_prefix}"
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(full_text, return_tensors="pt",
                                  add_special_tokens=False).to(device)
    L = prompt_ids.shape[1]

    q_text = f"\n\n{question}\n{answer_prefix}"
    q_only_ids = tokenizer.encode(q_text, add_special_tokens=False)
    q_len = len(q_only_ids)
    ctx_end = L - q_len

    middle_start = SINK
    middle_end = max(SINK, min(ctx_end, L - RECENT))
    n_middle = middle_end - middle_start

    # ================================================================
    # Step 1: Prefill + QA scoring (hooks compute scores during forward)
    # ================================================================
    qa_scores = torch.zeros(n_layers, n_kv_heads, L)
    out = compute_scores_via_hooks(
        model, prompt_ids, qa_scores,
        target_range=(0, L), q_start=L - q_len,
    )
    cache = out.past_key_values
    torch.cuda.empty_cache()

    # ================================================================
    # Step 2: Full KV baseline
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
    # Step 3: Reconstruction scoring (KVzip-style chunked)
    # ================================================================
    recons_scores = torch.zeros(n_layers, n_kv_heads, L)
    
    chunk_size = RECONS_CHUNK
    n_chunks = max(1, (ctx_end + chunk_size - 1) // chunk_size)
    
    for ci in range(n_chunks):
        c_start = ci * chunk_size
        c_end = min((ci + 1) * chunk_size, ctx_end)
        
        if ci == 0:
            repeat_prompt = "\n\nRepeat the previous context exactly."
        else:
            prev_end = ci * chunk_size
            prev_start = max(0, prev_end - 8)
            hint_ids = prompt_ids[0, prev_start:prev_end]
            hint_text = tokenizer.decode(hint_ids)
            repeat_prompt = f"\n\nRepeat the part of the previous context exactly, starting with {hint_text}"
        
        repeat_ids = tokenizer.encode(repeat_prompt, return_tensors="pt",
                                       add_special_tokens=False).to(device)
        chunk_ids = prompt_ids[:, c_start:c_end]
        query_ids = torch.cat([repeat_ids, chunk_ids], dim=1)
        q_total_len = query_ids.shape[1]
        
        cache_for_recons = copy.deepcopy(cache)
        pos_start = cache_for_recons.get_seq_length()
        position_ids = torch.arange(pos_start, pos_start + q_total_len,
                                    device=device).unsqueeze(0)
        
        compute_scores_via_hooks(
            model, query_ids, recons_scores[:, :, c_start:c_end],
            target_range=(c_start, c_end), q_start=0,
            position_ids=position_ids, past_key_values=cache_for_recons,
            subsampled=True, n_sink=SINK,
        )
        
        del cache_for_recons
        torch.cuda.empty_cache()

    # ================================================================
    # Step 4: Evict and evaluate
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

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  (intermediate save to {args.output})")


# Final Summary
print(f"\n{'='*90}")
print(f"FINAL RESULTS")
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

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
with open(args.output, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {args.output}")
print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")
print("DONE")
