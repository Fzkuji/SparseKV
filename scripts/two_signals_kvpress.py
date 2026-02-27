"""
Evaluate KVzip reconstruction scoring using kvpress's KVzipPress.
Optimization: run KVzip scoring ONCE per sample, then manually evict at different ratios.

Usage:
    python -u scripts/two_signals_kvpress.py [--model MODEL] [--n_samples N] [--output PATH]
"""
import torch, json, random, os, argparse, time, copy, logging
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvpress.presses.kvzip_press import KVzipPress

logging.getLogger("kvpress").setLevel(logging.ERROR)

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
device = model.device

n_layers = model.config.num_hidden_layers
n_kv_heads = model.config.num_key_value_heads

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


def generate_greedy(model, tokenizer, cache, max_new_tokens):
    """Greedy generation from cache (last token already processed)."""
    # We need to generate from the cache. The cache already contains all context KVs.
    # Use a simple loop.
    generated_ids = []
    
    # Get the last position from cache to create proper input
    cache_len = cache.get_seq_length()
    
    # We need to do a forward pass to get the first token's logits
    # The pipeline already did the prefill, so we just need to decode
    # Actually we need to call model with a token to get logits
    # Use the EOS token as a dummy — no wait, we need the actual last-token logits
    
    # The prefill forward already happened and cache is populated.
    # But we don't have the logits. We need to re-run the last token.
    # Actually kvpress pipeline handles generation internally.
    # Let me use a different approach: use model.generate with the cache.
    
    # Create a dummy input_ids (just need 1 token that was the last context token)
    # But we don't know it. Let's use generate() properly.
    
    # We'll use the generate method with past_key_values
    # Actually, we need to produce the right input. Let me think...
    
    # Simplest: just do generation ourselves
    # We need the logits from the last token. Let's get them by running a single forward.
    # But the cache already has all tokens processed. To get logits for the last token,
    # we'd need to run it through the model again, but that would duplicate the last KV entry.
    
    # The cleanest approach: during prefill, save the logits.
    return None  # This approach won't work cleanly


def run_kvzip_scoring_and_eval(model, tokenizer, prompt, question, answer_prefix, 
                                max_new_tokens, evict_ratios):
    """
    Run KVzip scoring ONCE, then evaluate at multiple compression ratios.
    Returns dict of {ratio: generated_text}
    """
    results = {}
    
    # Run full KV first (no compression)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{prompt}\n\n{question}\n{answer_prefix}"}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    input_ids = tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False).to(device)
    
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                              use_cache=True)
    gen_ids = out[0, input_ids.shape[1]:]
    results["full_kv"] = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    # Now run KVzip scoring using kvpress
    # We use compression_ratio=0.5 as dummy (we'll extract score_val before compression)
    # Actually, we need to hack KVzipPress to only do scoring without compression
    
    # Better approach: run KVzipPress at each ratio individually but cache the model
    # The overhead is the chunked forward pass per ratio
    
    # Actually the fastest approach for our comparison:
    # Just run KVzipPress at each ratio. Each run does:
    #   1. Prefill (~1x)
    #   2. Chunked scoring (~2x)
    #   3. Compress + generate
    # Total ~3-4x per ratio, 11 ratios = 33-44x per sample. Very slow.
    
    # The CORRECT optimization: subclass KVzipPress, override compress_post to try all ratios
    
    return results


class MultiRatioKVzipPress(KVzipPress):
    """
    Subclass that does KVzip scoring once, then evaluates at multiple ratios.
    Stores results in self.multi_ratio_results.
    """
    
    def __init__(self, ratios, model_ref, tokenizer_ref, max_new_tokens, **kwargs):
        # Set compression_ratio to max so scoring covers everything
        super().__init__(compression_ratio=ratios[-1], **kwargs)
        self.ratios = ratios
        self.model_ref = model_ref
        self.tokenizer_ref = tokenizer_ref
        self.max_new_tokens = max_new_tokens
        self.multi_ratio_results = {}  # {ratio: generated_text}
    
    def compress_post(self, model):
        """Override to evaluate at multiple ratios."""
        if self.score_val is None:
            return
            
        n_layer, bsz, num_kv_heads, ctx_len = self.score_val.shape
        
        # Save original cache state
        original_cache = copy.deepcopy(self._cache)
        
        for ratio in self.ratios:
            if ratio <= 0:
                continue
            
            # Restore cache
            for li in range(n_layer):
                self._cache.layers[li].keys = original_cache.layers[li].keys.clone()
                self._cache.layers[li].values = original_cache.layers[li].values.clone()
            
            # Compute eviction indices for this ratio
            n_pruned_total = int(self.score_val.numel() * ratio)
            pruned_indices = torch.topk(-self.score_val.reshape(-1), n_pruned_total).indices
            n_tokens_per_layer = bsz * num_kv_heads * ctx_len
            n_pruned_layers = torch.bincount(pruned_indices // n_tokens_per_layer, minlength=n_layer).int()
            
            for layer in model.model.layers:
                module = layer.self_attn
                layer_idx = int(module.layer_idx)
                scores = self.score_val[layer_idx]
                n_pruned = n_pruned_layers[layer_idx].cpu()
                
                if n_pruned > 0:
                    indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten().cpu()
                    batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
                    head_indices = indices // ctx_len
                    seq_indices = indices % ctx_len
                    
                    # Zero out evicted positions
                    self._cache.layers[layer_idx].keys[batch_indices, head_indices, seq_indices] = 0
                    self._cache.layers[layer_idx].values[batch_indices, head_indices, seq_indices] = 0
            
            # Generate with compressed cache
            cache_len = self._cache.get_seq_length()
            gen_ids = []
            # Create a forward pass to get first token logits
            # We need to re-run with the compressed cache
            # Actually, the cache position is already set. We just need to generate.
            
            # Use a simple generation loop
            with torch.no_grad():
                # Generate first token
                dummy_input = torch.tensor([[self.tokenizer_ref.eos_token_id]], device=model.device)
                # Actually this won't work — we need proper continuation
                
                # Better: use model.generate with past_key_values
                # But model.generate may not accept pre-filled cache directly...
                
                # Simplest working approach: create input_ids that are the full prompt,
                # but with use_cache pointing to our modified cache
                # Actually we can't — the cache already has all tokens processed
                
                # Let me try using the pipeline approach differently
                pass
            
            self.multi_ratio_results[ratio] = "[generation_pending]"
        
        # Restore original
        for li in range(n_layer):
            self._cache.layers[li].keys = original_cache.layers[li].keys
            self._cache.layers[li].values = original_cache.layers[li].values
        
        del original_cache


# Actually this subclass approach is getting too complex because generation from
# a modified cache isn't straightforward with HF APIs.
# 
# Let me go back to the simpler approach: use kvpress pipeline per-ratio,
# but it's slow. Let me estimate: with 91 samples and 11 ratios, ~1.5 min per (sample, ratio),
# that's 91 * 11 * 1.5 / 60 = 25 hours. Way too slow.
#
# Better plan: use kvpress pipeline ONCE per sample at a reference ratio,
# but extract score_val, then manually do eviction + generation.
# The key is we need to get score_val out and do our own generation.

class ScoringOnlyKVzipPress(KVzipPress):
    """Only do scoring, save score_val, don't compress."""
    
    def __init__(self, **kwargs):
        super().__init__(compression_ratio=0.5, **kwargs)  # dummy ratio
        self.saved_score = None
        self.saved_cache = None
    
    def compress_post(self, model):
        """Don't compress, just save scores and cache."""
        self.saved_score = self.score_val.clone() if self.score_val is not None else None
        self.saved_cache = copy.deepcopy(self._cache) if self._cache is not None else None
        # Don't call super — skip compression


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
    prompt_text = f"{context}\n\n{question}\n{answer_prefix}"
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    input_ids = tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False).to(device)
    L = input_ids.shape[1]

    # ================================================================
    # Full KV baseline
    # ================================================================
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new, do_sample=False, use_cache=True)
    gen_full = tokenizer.decode(out[0, L:], skip_special_tokens=True)
    full_correct = scorer(gen_full, answers)
    
    results.append({
        "sample_idx": int(sample_idx), "task": task,
        "method": "full_kv", "ratio": 0.0, "correct": bool(full_correct),
        "gen": gen_full[:200],
    })

    # ================================================================
    # KVzip scoring (once)
    # ================================================================
    press = ScoringOnlyKVzipPress()
    
    with press(model):
        cache = DynamicCache()
        with torch.no_grad():
            prefill_out = model(input_ids=input_ids, past_key_values=cache, num_logits_to_keep=1)
    
    score_val = press.saved_score  # [n_layer, 1, n_kv_heads, context_length]
    base_cache = press.saved_cache
    
    if score_val is None or base_cache is None:
        print(f"  [{idx_i+1}] KVzip scoring failed, skipping")
        for ratio in EVICT_RATIOS:
            results.append({
                "sample_idx": int(sample_idx), "task": task,
                "method": "kvzip", "ratio": ratio, "correct": False, "gen": "",
            })
        continue
    
    # Get the logits from prefill for first generated token
    first_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    # ================================================================
    # Evaluate at each ratio
    # ================================================================
    ctx_len = score_val.shape[-1]
    
    for ratio in EVICT_RATIOS:
        # Copy cache
        cache_copy = copy.deepcopy(base_cache)
        
        # Compute eviction
        n_pruned_total = int(score_val.numel() * ratio)
        if n_pruned_total > 0:
            pruned_flat = torch.topk(-score_val.reshape(-1), min(n_pruned_total, score_val.numel())).indices
            n_tokens_per_layer = score_val.shape[1] * n_kv_heads * ctx_len  # bsz * heads * ctx
            n_pruned_layers = torch.bincount(pruned_flat // n_tokens_per_layer, minlength=n_layers).int()
            
            for li in range(n_layers):
                n_p = n_pruned_layers[li].item()
                if n_p > 0:
                    scores_li = score_val[li]  # [1, n_kv_heads, ctx_len]
                    indices = torch.topk(-scores_li.reshape(1, -1), n_p, dim=1).indices.flatten()
                    head_idx = indices // ctx_len
                    seq_idx = indices % ctx_len
                    cache_copy.layers[li].keys[0, head_idx, seq_idx] = 0
                    cache_copy.layers[li].values[0, head_idx, seq_idx] = 0
        
        # Generate from compressed cache
        generated = [first_token.item()]
        next_token = first_token
        with torch.no_grad():
            for _ in range(max_new - 1):
                out_step = model(input_ids=next_token, past_key_values=cache_copy, use_cache=True)
                next_token = out_step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if next_token.item() == tokenizer.eos_token_id:
                    break
                generated.append(next_token.item())
        
        gen_text = tokenizer.decode(generated, skip_special_tokens=True)
        correct = scorer(gen_text, answers)
        
        results.append({
            "sample_idx": int(sample_idx), "task": task,
            "method": "kvzip", "ratio": ratio, "correct": bool(correct),
            "gen": gen_text[:200],
        })
        
        del cache_copy
    
    del base_cache, score_val, press
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


# Final
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
