<p align="center">
  <h1 align="center">SparseKV</h1>
  <p align="center"><b>Adaptive Block Dropout Training for Robust KV Cache Eviction</b></p>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/NVIDIA/kvpress"><img src="https://img.shields.io/badge/built%20on-kvpress-green.svg" alt="kvpress"></a>
</p>

---

Deploying LLMs with long contexts requires KV cache eviction — but current methods assume that a fixed subset of tokens remains important throughout generation. **This assumption is fragile** ([DefensiveKV, 2025](https://arxiv.org/abs/2510.13334)).

**SparseKV** takes a different approach: instead of building better eviction heuristics, we **train the model itself** to have sparser attention patterns, so that *any* eviction method works better.

## 💡 Key Idea

```
Training with block-wise KV cache dropout
  → Model learns to concentrate attention on fewer, more important tokens
    → Inference-time eviction is safer (deleted tokens truly don't matter)
      → Works with ANY eviction method (SnapKV, H2O, StreamingLLM, ...)
```

## 🔧 Installation

```bash
pip install -e .

# With evaluation dependencies:
pip install -e ".[eval]"

# With visualization:
pip install -e ".[vis]"

# Everything:
pip install -e ".[all]"
```

## 🚀 Quick Start

### Training with Block Dropout

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from sparsekv import BlockDropoutPress
from sparsekv.training import SparseKVTrainer, LinearCurriculum

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Create press — drops 30% of KV cache blocks during training
press = BlockDropoutPress(block_size=64, drop_ratio=0.3, protect_start=4, protect_recent=64)

# Gradually increase dropout ratio
curriculum = LinearCurriculum(start_ratio=0.0, end_ratio=0.5, warmup_steps=1000)

trainer = SparseKVTrainer(
    press=press,
    curriculum=curriculum,
    model=model,
    args=TrainingArguments(output_dir="./output", learning_rate=1e-5, num_train_epochs=1),
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### Evaluation with kvpress

```python
from transformers import pipeline
from kvpress import SnapKVPress, ExpectedAttentionPress

# Load your trained model
pipe = pipeline("kv-press-text-generation", model="./output", device_map="auto")

# Test with ANY eviction method — trained model should be more robust
for press in [SnapKVPress(compression_ratio=0.5), ExpectedAttentionPress(compression_ratio=0.5)]:
    result = pipe(context, question=question, press=press)
    print(result["answer"])
```

### CLI Training

```bash
# Quick iteration on small model
sparsekv-train --config configs/train/block_dropout_1b.yaml

# Full training
sparsekv-train --config configs/train/adaptive_dropout_8b.yaml

# Evaluation
sparsekv-eval --config configs/eval/full_eval.yaml
```

## 📦 Available Methods

### Training Presses

| Press | Description | Key Feature |
|-------|-------------|-------------|
| `BlockDropoutPress` | Fixed-size block dropout | Simple, effective baseline |
| `AdaptiveBlockDropoutPress` | Attention-density-aware dropout | ⭐ Drops unimportant blocks more aggressively |
| `VariableBlockDropoutPress` | Variable-length blocks | More diverse dropout patterns |
| `SentenceDropoutPress` | Sentence-level dropout | Semantically meaningful boundaries |
| `SoftThresholdPress` | Sigmoid soft masking | Fully differentiable, temperature annealing |
| `EvictionAugPress` | Existing methods as augmentation | Use SnapKV/H2O during training |
| `SparseRegPress` | Entropy/L1 regularization | Encourage sparse attention directly |

### Training Objectives

| Objective | Description |
|-----------|-------------|
| `StandardLMObjective` | Next-token prediction with dropout (default) |
| `SparseLMObjective` | LM + sparsity regularization |
| `ReconstructionObjective` | LM + reconstruct dropped content |
| `MixedObjective` | Configurable combination of all above |

### Curriculum Schedules

| Schedule | Description |
|----------|-------------|
| `LinearCurriculum` | Linear ramp from 0% to target dropout |
| `StepCurriculum` | Step-wise increase |
| `CosineCurriculum` | Cosine annealing schedule |

## 📊 Evaluation Metrics

### Sparsity Metrics
- **Effective Support**: `exp(entropy(attention))` — lower = sparser
- **Top-K Coverage**: fraction of attention in top-K tokens — higher = sparser
- **Block Sparsity**: fraction of blocks with negligible attention

### Stability Metrics (DefensiveKV-inspired)
- **Min Retained Importance**: worst-case attention retention after eviction
- **Jaccard Stability**: consistency of eviction decisions across generation steps
- **Flip Rate**: how often tokens switch between retained/evicted

## 🏗️ Architecture

SparseKV is built on [NVIDIA kvpress](https://github.com/NVIDIA/kvpress):

```
kvpress BasePress (forward hook on attention layers)
    ↑ inherit
SparseKV Presses (block dropout, adaptive, soft threshold, ...)
    ↓ used by
SparseKVTrainer (HuggingFace Trainer + press integration + curriculum)
    ↓ evaluated with
kvpress evaluation (RULER, LongBench, NIAH, ...) + sparsity/stability metrics
```

All presses use kvpress's hook mechanism: they register a `forward_hook` on each attention layer that automatically modifies the KV cache after the attention computation. **No model code is modified.**

## 📂 Project Structure

```
SparseKV/
├── sparsekv/
│   ├── presses/           # Training presses (inherit kvpress BasePress)
│   ├── training/          # Trainer, objectives, curriculum, data
│   ├── evaluation/        # Sparsity & stability metrics
│   └── utils/             # Attention analysis & visualization
├── configs/               # YAML configs for training & evaluation
├── scripts/               # Shell scripts
├── tests/                 # Unit tests
└── notebooks/             # Demo notebooks
```

## 🔬 Experimental Design

### Key Experiments

1. **Block Dropout vs No Dropout**: Does training with dropout improve eviction?
2. **Adaptive vs Fixed**: Does attention-aware dropout help?
3. **Generalization**: Train with method A, evaluate with method B
4. **Curriculum**: Effect of gradually increasing dropout
5. **Sparsity Analysis**: How do attention patterns change?

### Recommended Pipeline

```bash
# 1. Baseline evaluation
bash scripts/eval_baseline.sh

# 2. Train with block dropout
bash scripts/train.sh configs/train/block_dropout_8b.yaml

# 3. Evaluate trained model
bash scripts/eval_trained.sh ./output/block_dropout_8b

# 4. Train with adaptive dropout
bash scripts/train.sh configs/train/adaptive_dropout_8b.yaml

# 5. Compare all results
```

## 📖 Citation

```bibtex
@article{sparsekv2026,
  title={SparseKV: Adaptive Block Dropout Training for Robust KV Cache Eviction},
  author={},
  year={2026},
}
```

## 🙏 Acknowledgments

- [NVIDIA kvpress](https://github.com/NVIDIA/kvpress) — KV cache compression framework
- [DefensiveKV](https://arxiv.org/abs/2510.13334) — stability assumption analysis
- [DropKey](https://arxiv.org/abs/2208.02646) — attention dropout for ViT (CVPR 2023)

## License

Apache 2.0
