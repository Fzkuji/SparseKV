"""
Preliminary experiment: Compare QA vs Reconstruction importance signals for KV eviction.
Both use QA task data (RULER). Evaluate answer quality after eviction at various ratios.

Method 1 (QA): Prefill [context+question+answer], score by question+answer attention, evict, re-answer
Method 2 (Recons): Prefill [context+question], reconstruct via "repeat" query, score by recons attention, evict, answer

Usage:
    python -u scripts/two_signals.py [--model MODEL] [--n_samples N] [--output PATH]
"""
import torch, numpy as np, json, random, copy, os, argparse, time, math
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--n_samples", type=int, default=100)
parser.add_argument("--output", default="results/two_signals.json")
parser.add_argument("--dataset_dir", default="4096")
parser.add_argument("--chunk_size", type=int, default=2000, help="Chunk size for reconstruction")
args = parser.parse_args()

MODEL = args.model
EVICT_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
SINK = 4
RECENT = 64

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
    """Generate tokens using existing KV cache."""
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


def generate_from_cache(cache, query_ids, max_new_tokens):
    """Forward query_ids using existing cache, then generate."""
    with torch.no_grad():
        out = model(input_ids=query_ids, past_key_values=cache, use_cache=True)
    return generate_with_cache(cache, out, max_new_tokens)


class AttentionScoreCollector:
    """Collect max attention received by each KV position during forward pass."""
    
    def __init__(self, model, n_layers, n_q_heads, n_kv_heads, n_groups, head_dim, scale, device):
        self.model = model
        self.n_layers = n_layers
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_groups
        self.head_dim = head_dim
        self.scale = scale
        self.device = device
        self.scores = None  # [n_layers, n_kv_heads, kv_len]
        self.hooks = []
        self.kv_len = None  # length of KV cache (context) to score
        self.query_offset = 0  # offset for new query tokens in full sequence
    
    def init_scores(self, kv_len):
        """Initialize score tensors."""
        self.kv_len = kv_len
        self.scores = torch.zeros(self.n_layers, self.n_kv_heads, kv_len, device='cpu')
    
    def register_hooks(self, score_range=None):
        """Register attention hooks on all layers.
        score_range: (start, end) indices in KV dimension to score. If None, score all [0, kv_len).
        """
        self.score_range = score_range or (0, self.kv_len)
        self.hooks = []
        
        for layer in self.model.model.layers:
            li = int(layer.self_attn.layer_idx)
            self.hooks.append(
                layer.self_attn.register_forward_hook(
                    self._make_hook(li), with_kwargs=True
                )
            )
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def _make_hook(self, li):
        def hook_fn(module, args, kwargs, output):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is None:
                return
            bsz, q_len, _ = hs.shape
            
            pe = kwargs.get("position_embeddings", None)
            
            with torch.no_grad():
                q = module.q_proj(hs).view(bsz, q_len, self.n_q_heads, self.head_dim).transpose(1, 2)
                k = module.k_proj(hs).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
                if pe is not None:
                    cos, sin = pe
                    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                    k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
                
                # If we have KV cache, we need full key states
                # During prefill, q_len == total_len, k comes from current input only
                # We compute attention and extract scores for the target range
                
                for g in range(self.n_kv_heads):
                    qh = slice(g * self.n_groups, (g + 1) * self.n_groups)
                    q_g = q[0, qh]  # [n_groups, q_len, head_dim]
                    k_g = k[0, g:g+1].expand(self.n_groups, -1, -1)  # [n_groups, k_len, head_dim]
                    
                    logits = torch.matmul(q_g, k_g.transpose(-1, -2)) / self.scale
                    # Causal mask
                    if q_len == k_g.shape[1]:  # prefill: q_len == k_len
                        causal = torch.triu(torch.ones(q_len, q_len, device=self.device, dtype=torch.bool), diagonal=1)
                        logits.masked_fill_(causal.unsqueeze(0), float('-inf'))
                    
                    attn = torch.softmax(logits.float(), dim=-1)  # [n_groups, q_len, k_len]
                    
                    # Max attention received by each KV position in score_range
                    # across all query positions and groups
                    s_start, s_end = self.score_range
                    attn_slice = attn[:, :, s_start:s_end]  # [n_groups, q_len, score_len]
                    max_score = attn_slice.amax(dim=(0, 1))  # [score_len] - max over groups and queries
                    
                    # Update scores (element-wise max across chunks)
                    prev = self.scores[li, g, s_start:s_end]
                    self.scores[li, g, s_start:s_end] = torch.max(prev, max_score.cpu())
                    
                    del logits, attn, attn_slice, max_score
                del q, k
        return hook_fn


def evict_cache(cache, scores, n_layers, n_kv_heads, kv_len, ratio, device):
    """Evict KV pairs based on scores. Keep sink + recent + top-scored middle tokens."""
    cache_copy = copy.deepcopy(cache)
    
    middle_start = SINK
    middle_end = max(0, kv_len - RECENT)
    n_middle = middle_end - middle_start
    
    if n_middle <= 0:
        return cache_copy, 0.0
    
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
    
    n_kept = SINK + RECENT + min(n_keep, n_middle)
    actual_cr = 1.0 - n_kept / kv_len
    return cache_copy, actual_cr


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
    
    # Build the full answer text (for QA signal, we need ground truth answer)
    gt_answer = ", ".join(str(a) for a in answers) if isinstance(answers, list) else str(answers)
    
    # ================================================================
    # Method 1: QA Signal
    # Prefill [context + question + answer], use question+answer attention
    # ================================================================
    
    # Build prompt with answer included
    prompt_with_answer = f"{context}\n\n{question}\n{answer_prefix}{gt_answer}"
    full_text_qa = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_with_answer}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    qa_ids = tokenizer.encode(full_text_qa, return_tensors="pt", add_special_tokens=False).to(device)
    L_qa = qa_ids.shape[1]
    
    # Find where question starts (approximate)
    context_only = f"{context}\n\n"
    context_ids = tokenizer.encode(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": context_only}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        ),
        add_special_tokens=False
    )
    q_start_qa = len(context_ids)
    
    # Also build prompt WITHOUT answer for evaluation
    prompt_no_answer = f"{context}\n\n{question}\n{answer_prefix}"
    full_text_eval = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_no_answer}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    eval_ids = tokenizer.encode(full_text_eval, return_tensors="pt", add_special_tokens=False).to(device)
    L_eval = eval_ids.shape[1]
    
    # --- QA scoring ---
    collector_qa = AttentionScoreCollector(model, n_layers, n_q_heads, n_kv_heads, n_groups, head_dim, scale, device)
    collector_qa.init_scores(L_qa)
    collector_qa.register_hooks(score_range=(0, L_qa))
    
    cache_qa = DynamicCache()
    with torch.no_grad():
        out_qa = model(input_ids=qa_ids, past_key_values=cache_qa, use_cache=True)
    collector_qa.remove_hooks()
    
    # For QA signal, we only want attention FROM question+answer tokens TO context
    # The hook already computed max over all queries. We need to re-score using only q+a queries.
    # Actually, the hook computes max over ALL query positions. We need a modified version.
    # Let's re-do with a focused hook that only considers queries from q_start onward.
    
    # Re-score: zero out scores from context-only queries
    # Simpler approach: run a second pass where we mask scores from non-QA queries
    # Actually, let's just use a separate scoring approach for QA
    
    # Clear and redo with QA-focused scoring
    del cache_qa, out_qa
    torch.cuda.empty_cache()
    
    qa_scores = torch.zeros(n_layers, n_kv_heads, L_qa)
    
    qa_hooks = []
    def make_qa_hook(li, q_start):
        def hook_fn(module, args, kwargs, output):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is None:
                return
            bsz, seq_len, _ = hs.shape
            if seq_len != L_qa:
                return
            pe = kwargs.get("position_embeddings", None)
            with torch.no_grad():
                q = module.q_proj(hs).view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                k = module.k_proj(hs).view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                if pe is not None:
                    cos, sin = pe
                    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                    k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
                for g in range(n_kv_heads):
                    qh = slice(g * n_groups, (g + 1) * n_groups)
                    q_g = q[0, qh]
                    k_g = k[0, g:g+1].expand(n_groups, -1, -1)
                    logits = torch.matmul(q_g, k_g.transpose(-1, -2)) / scale
                    causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
                    logits.masked_fill_(causal.unsqueeze(0), float('-inf'))
                    attn = torch.softmax(logits.float(), dim=-1)
                    # Only use attention FROM question+answer tokens (q_start:)
                    attn_qa = attn[:, q_start:, :]  # [n_groups, qa_len, full_len]
                    max_score = attn_qa.amax(dim=(0, 1))  # [full_len]
                    qa_scores[li, g] = max_score.cpu()
                    del logits, attn, attn_qa, max_score
                del q, k
        return hook_fn
    
    for layer in model.model.layers:
        li = int(layer.self_attn.layer_idx)
        qa_hooks.append(layer.self_attn.register_forward_hook(
            make_qa_hook(li, q_start_qa), with_kwargs=True))
    
    cache_qa_score = DynamicCache()
    with torch.no_grad():
        out_qa_score = model(input_ids=qa_ids, past_key_values=cache_qa_score, use_cache=True)
    for h in qa_hooks:
        h.remove()
    del cache_qa_score, out_qa_score
    torch.cuda.empty_cache()
    
    # Now build eval cache (without answer) for actual generation
    cache_eval = DynamicCache()
    with torch.no_grad():
        out_eval = model(input_ids=eval_ids, past_key_values=cache_eval, use_cache=True)
    
    # Full KV baseline
    cache_full = copy.deepcopy(cache_eval)
    gen_full = generate_with_cache(cache_full, out_eval, max_new)
    full_correct = scorer(gen_full, answers)
    del cache_full
    
    results.append({
        "sample_idx": int(sample_idx), "task": task,
        "method": "full_kv", "ratio": 0.0, "correct": bool(full_correct),
        "gen": gen_full[:200],
    })
    
    # QA signal eviction at various ratios
    # Use qa_scores but only for the eval-length prefix (context+question, no answer)
    for ratio in EVICT_RATIOS:
        cache_evicted, actual_cr = evict_cache(
            cache_eval, qa_scores[:, :, :L_eval], n_layers, n_kv_heads, L_eval, ratio, device
        )
        gen_text = generate_with_cache(cache_evicted, out_eval, max_new)
        correct = scorer(gen_text, answers)
        results.append({
            "sample_idx": int(sample_idx), "task": task,
            "method": "qa", "ratio": ratio, "actual_cr": round(actual_cr, 4),
            "correct": bool(correct), "gen": gen_text[:200],
        })
        del cache_evicted
        torch.cuda.empty_cache()
    
    del cache_eval
    torch.cuda.empty_cache()
    
    # ================================================================
    # Method 2: Reconstruction Signal
    # Prefill [context + question], then reconstruct via "repeat" query
    # ================================================================
    
    # Build context+question cache
    cache_recons = DynamicCache()
    with torch.no_grad():
        out_recons = model(input_ids=eval_ids, past_key_values=cache_recons, use_cache=True)
    
    # Reconstruction scoring: chunk the context+question and reconstruct each chunk
    recons_scores = torch.zeros(n_layers, n_kv_heads, L_eval)
    
    # Chunk the eval_ids for reconstruction
    chunk_size = args.chunk_size
    n_chunks = math.ceil(L_eval / chunk_size)
    
    for ci in range(n_chunks):
        c_start = ci * chunk_size
        c_end = min((ci + 1) * chunk_size, L_eval)
        chunk_tokens = eval_ids[0, c_start:c_end]
        
        # Build reconstruction query
        if ci == 0:
            recons_prompt = "\n\nRepeat the previous context exactly."
        else:
            # Include last 8 tokens of previous chunk as hint
            prev_end = c_start
            prev_start = max(0, prev_end - 8)
            hint_tokens = eval_ids[0, prev_start:prev_end]
            hint_text = tokenizer.decode(hint_tokens, skip_special_tokens=False)
            recons_prompt = f"\n\nRepeat the part of the previous context exactly, starting with {hint_text}"
        
        recons_query_ids = tokenizer.encode(recons_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        # Append the chunk tokens as "answer" (self-supervised)
        recons_input = torch.cat([recons_query_ids, chunk_tokens.unsqueeze(0)], dim=1)
        
        # Register hooks for this chunk
        recons_hooks = []
        def make_recons_hook(li, kv_len, score_start, score_end):
            def hook_fn(module, args, kwargs, output):
                hs = kwargs.get("hidden_states", args[0] if args else None)
                if hs is None:
                    return
                bsz, q_len, _ = hs.shape
                # This is a decode-like forward: q_len is the recons query length
                # The KV cache has kv_len entries, and we're adding q_len new ones
                # We want attention from new queries to existing KV
                pe = kwargs.get("position_embeddings", None)
                with torch.no_grad():
                    q = module.q_proj(hs).view(bsz, q_len, n_q_heads, head_dim).transpose(1, 2)
                    k = module.k_proj(hs).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
                    if pe is not None:
                        cos, sin = pe
                        q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                        k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
                    
                    # Get full key states from cache + current
                    # During this forward, the cache already has kv_len entries
                    # We need to get the cached keys too
                    cached_k = module.k_proj.weight  # This won't work...
                    
                    # Actually, for reconstruction scoring, we need the attention weights
                    # between new query tokens and ALL KV cache entries.
                    # The model's attention mechanism handles this internally.
                    # We need to intercept the actual attention weights, not recompute them.
                    # 
                    # Problem: with use_cache, k only contains NEW keys, not cached ones.
                    # We can't easily get full attention from hooks alone.
                    # 
                    # Alternative: Don't update cache during recons scoring.
                    # Build full input = [cached_context ... recons_query ... chunk_answer]
                    # and do a full prefill-style forward.
                    pass
                del q, k
            return hook_fn
        
        # The hook approach for decode-mode is complicated because we don't have
        # access to cached keys in the hook. Instead, let's do a different approach:
        # Build a full sequence and do prefill-style scoring.
        
        # Actually, let's use a simpler approach: for each chunk, we'll concatenate
        # the full context + recons_query + chunk as one big prefill.
        # But that's expensive. Let me use the cache properly.
        
        # Simplest correct approach: use output attention weights
        # Or: concatenate everything and do a single prefill with hooks
        
        del recons_hooks
        
        # --- Simpler approach: single prefill with reconstruction query appended ---
        # Build: [eval_ids] + [recons_query] + [chunk_tokens]
        full_recons_input = torch.cat([eval_ids, recons_query_ids, chunk_tokens.unsqueeze(0)], dim=1)
        L_full = full_recons_input.shape[1]
        
        chunk_recons_hooks = []
        def make_chunk_recons_hook(li, kv_len, q_offset):
            """Score KV positions [0, kv_len) using attention from positions [q_offset:]"""
            def hook_fn(module, args, kwargs, output):
                hs = kwargs.get("hidden_states", args[0] if args else None)
                if hs is None:
                    return
                bsz, seq_len, _ = hs.shape
                if seq_len != L_full:
                    return
                pe = kwargs.get("position_embeddings", None)
                with torch.no_grad():
                    q = module.q_proj(hs).view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
                    k = module.k_proj(hs).view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                    if pe is not None:
                        cos, sin = pe
                        q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
                        k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
                    for g in range(n_kv_heads):
                        qh = slice(g * n_groups, (g + 1) * n_groups)
                        q_g = q[0, qh]
                        k_g = k[0, g:g+1].expand(n_groups, -1, -1)
                        logits = torch.matmul(q_g, k_g.transpose(-1, -2)) / scale
                        causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
                        logits.masked_fill_(causal.unsqueeze(0), float('-inf'))
                        attn = torch.softmax(logits.float(), dim=-1)
                        # Attention FROM recons query+answer TO context+question KV
                        attn_recons = attn[:, q_offset:, :kv_len]  # [groups, recons_len, kv_len]
                        chunk_max = attn_recons.amax(dim=(0, 1))  # [kv_len]
                        # Element-wise max with existing scores
                        prev = recons_scores[li, g, :kv_len]
                        recons_scores[li, g, :kv_len] = torch.max(prev, chunk_max.cpu())
                        del logits, attn, attn_recons, chunk_max
                    del q, k
            return hook_fn
        
        for layer in model.model.layers:
            li = int(layer.self_attn.layer_idx)
            chunk_recons_hooks.append(layer.self_attn.register_forward_hook(
                make_chunk_recons_hook(li, L_eval, L_eval), with_kwargs=True))
        
        # Forward (no cache needed, full prefill)
        with torch.no_grad():
            _ = model(input_ids=full_recons_input, use_cache=False)
        
        for h in chunk_recons_hooks:
            h.remove()
        del full_recons_input
        torch.cuda.empty_cache()
    
    del cache_recons, out_recons
    torch.cuda.empty_cache()
    
    # Now evict using recons scores and evaluate
    cache_eval2 = DynamicCache()
    with torch.no_grad():
        out_eval2 = model(input_ids=eval_ids, past_key_values=cache_eval2, use_cache=True)
    
    for ratio in EVICT_RATIOS:
        cache_evicted, actual_cr = evict_cache(
            cache_eval2, recons_scores, n_layers, n_kv_heads, L_eval, ratio, device
        )
        gen_text = generate_with_cache(cache_evicted, out_eval2, max_new)
        correct = scorer(gen_text, answers)
        results.append({
            "sample_idx": int(sample_idx), "task": task,
            "method": "recons", "ratio": ratio, "actual_cr": round(actual_cr, 4),
            "correct": bool(correct), "gen": gen_text[:200],
        })
        del cache_evicted
        torch.cuda.empty_cache()
    
    del cache_eval2
    torch.cuda.empty_cache()
    
    elapsed = time.time() - t0
    eta = elapsed / (idx_i + 1) * (len(selected) - idx_i - 1)
    status = "OK" if full_correct else "FAIL(fullkv)"
    print(f"[{idx_i+1}/{len(selected)}] {task:>20} fullkv={status}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")
    
    # Progress summary
    if (idx_i + 1) % 10 == 0:
        print(f"\n--- Progress: {idx_i+1}/{len(selected)} ---")
        for method in ["qa", "recons"]:
            row = []
            for ratio in EVICT_RATIOS:
                mr = [r for r in results if r["method"] == method and r["ratio"] == ratio]
                if mr:
                    acc = sum(r["correct"] for r in mr) / len(mr) * 100
                    row.append(f"{ratio}:{acc:4.0f}%")
            print(f"  {method:>8}: {' | '.join(row)}")
        print()
        
        # Save intermediate
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

# ============================================================
# Final Summary
# ============================================================
print(f"\n{'='*90}")
print(f"FINAL RESULTS: QA vs Reconstruction Signal")
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
            mr = [r for r in results if r["task"] == task and r["method"] == method and r["ratio"] == ratio]
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
