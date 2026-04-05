# SparseKV - Implementation Guide

## Project Overview

SparseKV: Adaptive Block Dropout Training for Robust KV Cache Eviction

**Core Idea**: Train LLMs with block-wise KV cache dropout so that inference-time eviction methods (SnapKV, H2O, etc.) work better. The model learns to be robust to missing KV cache entries.

**Key dependency**: `kvpress` (NVIDIA's KV cache compression toolkit, `pip install kvpress`)

## Architecture

This project builds ON TOP of kvpress. We inherit from `kvpress.presses.base_press.BasePress` for our custom presses, and use kvpress's evaluation infrastructure for benchmarking.

### kvpress Key Classes

```python
# BasePress: base class, uses forward_hook on attention layers
# - compress(module, hidden_states, keys, values, attentions, kwargs) -> (keys, values)
# - forward_hook(): registered as post-forward hook on attention layers
# - __call__(model): context manager, registers hooks

# ScorerPress(BasePress): score-based selection
# - score() -> importance scores per token
# - compress() -> top-k selection based on scores
```

## Project Structure

```
SparseKV/
├── README.md                    # Professional README with badges, installation, usage, citation
├── LICENSE                      # Apache 2.0 (already exists)
├── pyproject.toml              # Modern Python packaging
├── .gitignore                  # Already exists
│
├── sparsekv/                  # Main package
│   ├── __init__.py             # Export all presses and key functions
│   │
│   ├── presses/                # Custom press implementations (inherit kvpress BasePress)
│   │   ├── __init__.py
│   │   ├── block_dropout_press.py      # Fixed block dropout
│   │   ├── adaptive_dropout_press.py   # Attention-density-aware adaptive dropout
│   │   ├── variable_block_press.py     # Variable-length block dropout
│   │   ├── sentence_dropout_press.py   # Sentence-level dropout
│   │   ├── soft_threshold_press.py     # Sigmoid soft masking
│   │   ├── eviction_aug_press.py       # Use existing eviction as data augmentation
│   │   └── sparse_reg_press.py         # Entropy/L1 regularization on attention
│   │
│   ├── training/               # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py          # Custom HuggingFace Trainer with press integration
│   │   ├── objectives.py       # Training objectives (LM, sparse LM, reconstruction, mixed)
│   │   ├── curriculum.py       # Curriculum learning (gradually increase dropout)
│   │   └── data.py             # Data loading utilities
│   │
│   ├── evaluation/             # Evaluation utilities
│   │   ├── __init__.py
│   │   ├── sparsity.py         # Measure attention sparsity metrics
│   │   ├── stability.py        # Measure eviction stability (DefensiveKV-style)
│   │   └── evaluate.py         # Evaluation script wrapping kvpress + lm-evaluation-harness
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── attention.py        # Attention analysis utilities
│       └── visualization.py    # Attention pattern visualization
│
├── configs/                    # Training/evaluation configs
│   ├── train/
│   │   ├── block_dropout_1b.yaml
│   │   ├── block_dropout_8b.yaml
│   │   ├── adaptive_dropout_8b.yaml
│   │   └── mixed_training_8b.yaml
│   └── eval/
│       ├── baseline.yaml
│       └── full_eval.yaml
│
├── scripts/                    # Shell scripts
│   ├── train.sh
│   ├── eval_baseline.sh
│   ├── eval_trained.sh
│   └── visualize.sh
│
├── notebooks/                  # Demo notebooks
│   ├── quickstart.ipynb
│   └── visualization.ipynb
│
└── tests/
    ├── test_presses.py
    └── test_training.py
```

## Implementation Details

### 1. Presses

#### BlockDropoutPress (block_dropout_press.py)
- Fixed-size block dropout on KV cache
- Parameters: block_size (default 64), drop_ratio (default 0.3), protect_start (4, sink tokens), protect_recent (64)
- During training: randomly drop blocks
- During eval: no dropout (pass-through), or use other kvpress presses

#### AdaptiveBlockDropoutPress (adaptive_dropout_press.py)
- Uses attention weights to compute per-block importance
- Attention density = intra-block attention sum / block_length
- Sink-aware: blocks with high sink attention ratio are more likely to be dropped
- Parameters: base_drop_ratio, block_size, temperature, sink_tokens

#### SoftThresholdPress (soft_threshold_press.py)
- Sigmoid soft mask: mask = sigmoid(temperature * (attn - threshold))
- Fully differentiable
- Temperature annealing during training

#### EvictionAugPress (eviction_aug_press.py)
- Wraps any kvpress ScorerPress to use during training as data augmentation
- No gradient through the eviction selection
- Can randomly switch between different eviction methods each step

#### SparseRegPress (sparse_reg_press.py)
- Adds entropy or L1 regularization loss on attention weights
- Returns the regularization loss for adding to the training objective

### 2. Training

#### SparseKVTrainer (trainer.py)
- Extends HuggingFace Trainer
- Integrates press into training loop via `with press(model):`
- Supports multiple training objectives
- Supports curriculum learning (increasing dropout ratio over time)

#### Training Objectives (objectives.py)
1. `StandardLMObjective`: Normal next-token prediction with dropout press
2. `SparseLMObjective`: LM loss + sparsity regularization
3. `ReconstructionObjective`: Reconstruct dropped content from sparse cache
4. `MixedObjective`: Combination of above with configurable weights

#### Curriculum (curriculum.py)
- LinearCurriculum: linearly increase drop_ratio from 0 to target
- StepCurriculum: step-wise increase
- CosineCurriculum: cosine schedule

### 3. Evaluation

#### Sparsity Metrics (sparsity.py)
- `effective_support`: exp(entropy(attention)) - lower = sparser
- `top_k_coverage`: fraction of attention mass in top-k tokens
- `block_sparsity`: fraction of blocks with very low attention

#### Stability Metrics (stability.py)
- `min_retained_importance`: minimum importance of retained tokens (DefensiveKV metric)
- `importance_variance`: variance of retained importance across steps

#### evaluate.py
- Wraps kvpress evaluation CLI
- Adds sparsity and stability metrics
- Supports comparing original vs trained model with various presses

### 4. Config Format (YAML)

```yaml
# Example: configs/train/block_dropout_8b.yaml
model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct
  dtype: bfloat16

press:
  type: adaptive_block_dropout
  block_size: 64
  base_drop_ratio: 0.3
  sink_tokens: 4
  protect_recent: 64

training:
  objective: mixed
  objective_weights:
    lm: 1.0
    sparse_lm: 0.5
    sparsity_reg: 0.01
  
  curriculum:
    type: linear
    warmup_steps: 1000
    start_ratio: 0.0
    end_ratio: 0.5
  
  learning_rate: 1e-5
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  max_seq_length: 4096
  
  dataset:
    name: allenai/c4
    split: train
    streaming: true

evaluation:
  presses: [snapkv, h2o, streaming_llm, expected_attention]
  compression_ratios: [0.3, 0.5, 0.7]
  datasets: [ruler, longbench, niah]
```

## README Style

The README should be professional and comprehensive, similar to kvpress's README:
- Logo/banner area
- Badges (license, python version, pip)
- Clear motivation section
- Installation
- Quick start code
- Available methods table
- Training guide
- Evaluation guide
- Results table (placeholder)
- Citation (BibTeX)
- Contributing
- Acknowledgments (kvpress, etc.)

## Important Notes

1. All presses MUST inherit from `kvpress.presses.base_press.BasePress`
2. Training presses modify the KV cache during forward pass (same hook mechanism as kvpress)
3. The project should work with `pip install -e .` and have proper dependencies in pyproject.toml
4. Use dataclasses for press parameters (same as kvpress convention)
5. Support batch_size > 1 where possible
6. Include type hints throughout
7. The code should be clean, well-documented, and ready for open-source
