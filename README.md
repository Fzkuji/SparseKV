# SparseKV: Adaptive KV Cache Eviction for LLMs

## Overview

SparseKV explores methods for compressing the KV cache in large language models during inference, enabling efficient long-context processing with minimal accuracy loss.

The project has gone through two main phases:
1. **Training-based approach** (v1-v9): Train LLMs with KV cache dropout to make them robust to eviction
2. **Two-Hop attention-based approach** (current): A training-free method that uses attention patterns to identify and preserve critical tokens

## Method Evolution

### Phase 1: Training-Based Approaches (v1-v9)

The original idea was to train the model to concentrate attention on "anchor tokens" (punctuation, special tokens, sink positions, recent tokens), so that non-anchor KV cache entries can be safely evicted at inference time.

**Architecture:**
```
Teacher forward (full KV, no grad) → teacher logits
Anchor mask + random KV dropout → sparse mask
Student forward (sparse KV, with grad) → student logits
Loss = CE(student, labels) + λ · KL(student || teacher)
```

- **v1-v5**: Basic block dropout training with various configurations
- **v6a/v6b**: Anchor-based training with LoRA (r=64, q/k/v/o projections)
- **v7**: Curriculum compression (keep_ratio 0.9 → 0.3)
- **v8**: Attention-based mask generation (matching inference-time eviction)
- **v9**: Entropy + KL regularization on attention patterns

**Result**: All training-based versions failed to beat even the weakest baseline (SnapKV) on high-CR eviction tasks. Root cause: **train-inference mismatch** — models learned to resist random dropout, but faced attention-score-based eviction at inference.

### Phase 2: Attention Analysis & Two-Hop Method (Current)

Through extensive attention pattern analysis on Qwen3-8B (documented in 30 sections of analysis), we discovered a training-free method that outperforms all baselines at high compression ratios.

#### Key Discoveries

1. **Period tokens as sentence anchors** (Section 11): Period "." tokens act as universal sentence-summary tokens, aggregating 66-99% attention from value digits in L5-8.

2. **Cross-layer information flow** (Section 18): Information retrieval spans multiple layers — no single layer captures the full query→key→value chain. Single-layer methods fundamentally cannot work.

3. **Ratio scoring** (Section 27): Dividing query attention by baseline attention (`q_mean[j] / baseline[j]`) dramatically outperforms subtraction (`q_mean[j] - baseline[j]`) by normalizing positional bias.

#### Two-Hop Method

```
Hop1: Differential attention (L21, ratio scoring)
      → Identify target sentence's period "."
      Question tokens attend to periods differently than non-question tokens.
      Ratio scoring normalizes positional bias, achieving mean rank 1.4 for target period.

Hop2: Period outgoing attention (L3-8)
      → Recover value tokens from the target sentence
      Period tokens attend strongly to their sentence's value digits in early layers.

Final: Anchor set = sink + recent + Hop1 periods + Hop2 value tokens
       → Apply to all layers uniformly
```

## Results

### NIAH (Needle-in-a-Haystack, 200 samples, 150 distractors, Qwen3-8B)

#### Head-to-Head: Two-Hop vs KVzip+

| CR Range | Two-Hop (ours) | KVzip+ | Delta |
|----------|---------------|--------|-------|
| ~89.5% | 99.0% (k15) | — | — |
| ~92.5% | **95.5%** (k10) | 93.0% | **+2.5%** |
| ~94.2% | **94.5%** (k7) | ~75%* | **+19.5%** |
| ~95.4% | **93.5%** (k5) | 61.0% | **+32.5%** |
| ~96.8% | **89.5%** (k3) | 15.0% | **+74.5%** |

*KVzip+ interpolated between 92.5% and 95% CR

#### All Baselines (50-sample)

| Method | CR=50% | CR=70% | CR=80% | CR=85% | CR=90% | CR=95% |
|--------|--------|--------|--------|--------|--------|--------|
| SnapKV | 16% | 8% | 6% | 2% | 2% | 0% |
| ExpectedAttn | 26% | 4% | 0% | 0% | 0% | 0% |
| KVzip | 100% | 100% | 98% | 98% | 94% | 56% |
| KVzip+ | 100% | 100% | 100% | 98% | 98% | 66% |

**Key findings:**
- At CR > 92%, our method dominates all baselines
- KVzip+ collapses at high CR (61% at 95%, 15% at 97%), while ours degrades gracefully
- Only 2 real method failures out of 193 valid samples at k=10

### FactQA (Knowledge Base QA, 200 samples, 150 entries, Qwen3-8B)

A generalization test with a different format: 150 fictional location entries with population, founding year, elevation, and mayor.

| Strategy | Accuracy (excl. fullFAIL) | Mean CR |
|----------|--------------------------|---------|
| ratio_k15 | 92.4% (182/197) | 92.6% |
| ratio_k10 | 90.4% (178/197) | 94.8% |
| ratio_k7 | 87.3% (172/197) | 96.2% |

KVzip+ FactQA results (partial): 96.0% @ 80% CR, 88.0% @ 85% CR, 83.0% @ 90% CR, 65.0% @ 92.5% CR — our method shows large advantage at high CR on this task as well.

**Known issue**: Year-type questions (~50% accuracy) need further investigation.

## Failed Approaches (for reference)

These methods were explored and documented before arriving at the Two-Hop method:

| Approach | Why It Failed |
|----------|--------------|
| Per-layer attention thresholding | Most layers don't attend to needle; per-layer decisions lose critical info |
| Cross-layer union eviction | 36-layer union causes severe budget inflation (top-5% per layer → 47% actual) |
| Iterative softmax sharpening | `softmax(probabilities)` compresses rather than sharpens [0,1] inputs |
| Power sharpening `p^α` | Same root cause: per-layer attention can't capture cross-layer information flow |
| Per-layer adaptive Hop2 | 0% accuracy — most layers have near-zero coverage of value tokens |
| 3-Hop tracing (key→period→value) | The chain spans different layers; single-layer computation breaks the chain |
| Training-based v1-v9 | Train-inference mismatch: random dropout ≠ attention-score-based eviction |

## Project Structure

```
SparseKV/
├── sparsekv/                    # Core package
│   ├── training/                # EIT training infrastructure (Phase 1)
│   │   ├── anchor.py            # Anchor token selection
│   │   ├── kv_dropout.py        # KV cache masking
│   │   ├── eit_trainer.py       # EIT training loop
│   │   └── scheduler.py         # Compression curriculum
│   ├── presses/                 # KV cache eviction policies
│   └── evaluation/              # Evaluation utilities
├── scripts/                     # Experiment scripts
│   ├── twohop_ratio_200.py      # Two-Hop NIAH evaluation (200 samples)
│   ├── kvzip_200.py             # KVzip+ NIAH baseline (200 samples)
│   ├── factqa_ratio.py          # Two-Hop FactQA evaluation
│   ├── factqa_kvzip.py          # KVzip+ FactQA baseline
│   ├── baseline_50samples.py    # Multi-baseline comparison
│   └── ...                      # Analysis and exploration scripts
├── configs/                     # Training configurations
├── docs/                        # Documentation
│   └── experiment_plan.md
└── analysis/                    # Attention analysis outputs
```

## Setup

### Prerequisites
- Python 3.10+, PyTorch 2.0+, CUDA 11.8+
- 1x A100 80GB (for inference experiments)

### Installation
```bash
git clone https://github.com/Fzkuji/AdaSparseKV.git
cd AdaSparseKV
pip install -e .
pip install kvpress transformers accelerate
```

For KVzip+ baseline:
```bash
cd ~/kvpress && pip install -e .
python -c "from kvpress.attention_patch import patch_attention_functions; patch_attention_functions()"
```

### Running Experiments

```bash
# Two-Hop evaluation (NIAH, 200 samples)
sbatch scripts/run_ratio200.sh

# KVzip+ baseline (NIAH, 200 samples)
sbatch scripts/run_kvzip200.sh

# FactQA evaluation
sbatch scripts/run_factqa_ratio.sh
sbatch scripts/run_factqa_kvzip.sh
```

## Current Status & Next Steps

- [x] Attention pattern analysis (30 sections documented)
- [x] Two-Hop method design and implementation
- [x] Ratio scoring discovery
- [x] NIAH 200-sample evaluation (our method + KVzip+)
- [x] FactQA generalization test
- [ ] Year question accuracy investigation
- [ ] Adaptive layer selection (model-specific, not task-specific)
- [ ] Low CR verification (our method at CR=80%)
- [ ] Additional task evaluation

## Model

Currently tested on **Qwen3-8B** (36 layers, 8 KV heads, 32 query heads, head_dim=128).

## License

Apache License 2.0
