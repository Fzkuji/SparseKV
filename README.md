# SparseKV: Training LLMs for Zero-Cost KV Cache Eviction

**SparseKV** trains large language models to concentrate their attention on "anchor tokens" (punctuation, special tokens, sink positions, recent tokens), enabling nearly zero-cost KV cache compression at inference time.

## 🔑 Key Idea

Instead of designing better eviction policies, we **train the model** to naturally focus on a sparse set of anchor tokens. At inference, we can safely evict non-anchor KV cache with minimal performance loss.

### Anchor Tokens
- **Special tokens**: BOS, EOS, PAD, etc.
- **Punctuation**: periods, commas, etc. (high attention receivers)
- **Sink tokens**: first K tokens (StreamingLLM observation)
- **Recent tokens**: last N tokens (recency bias)

## 📊 Method Overview

```
┌─────────────────────────────────────────────────────┐
│              Training (EIT)                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Step 1: Full Forward (no grad)                    │
│     input → [Full KV Cache] → teacher_logits       │
│                                                     │
│  Step 2: Build KV Mask                             │
│     Anchors: [special, punct, sink, recent]        │
│     + Random sample of non-anchors (keep_ratio)    │
│                                                     │
│  Step 3: Evicted Forward (with grad)               │
│     input → [Sparse KV Cache] → student_logits     │
│                                                     │
│  Step 4: Loss                                       │
│     L = CE(student, labels) + λ·KL(student||teacher)│
│                                                     │
│  Curriculum: keep_ratio 0.9 → 0.3 over training   │
│                                                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              Inference                              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Simply keep anchor tokens' KV cache               │
│  = 70-90% compression with <1% performance loss    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 80GB+ GPU memory recommended (for 8B models)

### Setup

```bash
# Clone repository
git clone https://github.com/Fzkuji/SparseKV.git
cd SparseKV

# Create conda environment
conda create -n sparsekv python=3.10
conda activate sparsekv

# Install package in editable mode
pip install -e .

# Install additional dependencies
pip install transformers accelerate peft datasets torch bitsandbytes
```

## 🚀 Quick Start

### 1. Analyze Pretrained Model (Optional)

Discover which token types naturally receive more attention:

```bash
python scripts/analyze_attention.py \
    --model Qwen/Qwen3-8B \
    --num_samples 100 \
    --output_dir ./analysis
```

### 2. Train SparseKV

Train the model to focus on anchor tokens:

```bash
python scripts/train_eit.py \
    --model Qwen/Qwen3-8B \
    --output_dir ./output/qwen3_sparsekv \
    --num_train_samples 10000 \
    --lambda_kl 1.0 \
    --initial_keep_ratio 0.9 \
    --min_keep_ratio 0.3
```

**Key arguments:**
- `--lambda_kl`: Weight for KL divergence loss (default: 1.0)
- `--initial_keep_ratio`: Starting compression (default: 0.9 = 90% kept)
- `--min_keep_ratio`: Target compression (default: 0.3 = 30% kept)
- `--sink_size`: Number of sink tokens (default: 4)
- `--recent_size`: Number of recent tokens (default: 64)
- `--scheduler_mode`: Compression schedule (curriculum|adaptive|fixed)

### 3. Evaluate

Test the trained model on benchmarks:

```bash
# Evaluate on LongBench, RULER, etc.
python scripts/eval_wrapper.py \
    --model ./output/qwen3_sparsekv/final \
    --dataset longbench \
    --press_name snapkv \
    --compression_ratio 0.5
```

## 🧪 Complete Experimental Workflow

### Phase 0: Attention Analysis

Understand attention patterns in the pretrained model:

```bash
# On cluster (Slurm)
sbatch scripts/slurm_phase0_analyze.sh

# Locally
python scripts/analyze_attention.py --model Qwen/Qwen3-8B --num_samples 100
```

**Output:** `./analysis/attention_analysis_Qwen--Qwen3-8B.json`

### Phase 1: Baseline Evaluation

Evaluate the pretrained model with various press methods:

```bash
# On cluster
bash scripts/submit_all.sh qwen3        # Qwen3-8B baseline
bash scripts/submit_all.sh llama        # Llama-3.1-8B baseline

# This submits evaluation jobs for:
# - Datasets: RULER (4k, 16k), LongBench, AIME25
# - Methods: SnapKV, StreamingLLM, Critical-SnapKV, KVZip
# - Compression ratios: 0.3, 0.5, 0.7
```

**Output:** `./results/phase1_qwen3/*`

### Phase 2: SparseKV Training

Train the model with anchor-based KV dropout:

```bash
# On cluster
sbatch scripts/slurm_phase2_train_qwen.sh   # Train Qwen3-8B
sbatch scripts/slurm_phase2_train_llama.sh  # Train Llama-3.1-8B

# Locally (if you have sufficient GPU memory)
python scripts/train_eit.py \
    --model Qwen/Qwen3-8B \
    --output_dir ./output/qwen3_sparsekv \
    --num_train_samples 10000
```

**Training details:**
- LoRA (r=64, α=128) on q/k/v/o projections
- Curriculum compression: 0.9 → 0.3 over training
- Two forward passes per batch (teacher + student)
- Loss: CE + KL divergence

**Output:** `./output/qwen3_sparsekv/final/` (trained model checkpoint)

### Phase 3: Evaluate Trained Model

Test the trained model with the same press methods:

```bash
# On cluster
bash scripts/submit_all.sh qwen3_trained    # Evaluate trained Qwen3-8B
bash scripts/submit_all.sh llama_trained    # Evaluate trained Llama-3.1-8B
```

**Output:** `./results/phase3_qwen3_trained/*`

## 📁 Project Structure

```
SparseKV/
├── sparsekv/
│   ├── training/
│   │   ├── anchor.py            # Anchor token selection
│   │   ├── kv_dropout.py        # KV cache masking during forward
│   │   ├── eit_trainer.py       # Main EIT training loop
│   │   └── scheduler.py         # Compression curriculum
│   ├── presses/                 # KV cache eviction policies
│   │   ├── snapkv_press.py
│   │   ├── streaming_llm_press.py
│   │   └── ...
│   ├── evaluation/              # Evaluation utilities
│   └── utils/                   # Helpers (attention analysis, etc.)
├── scripts/
│   ├── train_eit.py            # Training launch script
│   ├── analyze_attention.py    # Attention pattern analysis
│   ├── eval_wrapper.py         # Evaluation wrapper
│   ├── submit_all.sh           # Batch job submission
│   └── slurm_*.sh              # Slurm job scripts
├── configs/                     # Configuration files
├── docs/                        # Documentation
│   └── experiment_plan.md
├── tests/                       # Unit tests
├── pyproject.toml              # Package metadata
└── README.md
```

## 🎯 Expected Results

After training, the model should:
- Concentrate ≥80% of attention on ~20-30% of tokens (anchors)
- Maintain <1% performance drop with 70% KV cache eviction
- Outperform heuristic methods (SnapKV, StreamingLLM) at the same compression ratio

**Example (target):**

| Method | Compression | LongBench Acc | RULER@16k F1 |
|--------|-------------|---------------|--------------|
| Full KV | 0% | 65.2 | 78.5 |
| SnapKV | 50% | 62.1 (-3.1) | 71.3 (-7.2) |
| **SparseKV (ours)** | 50% | **64.8 (-0.4)** | **77.9 (-0.6)** |

## 🛠️ Advanced Usage

### Custom Anchor Configuration

```python
from sparsekv.training import EITConfig, AnchorConfig

config = EITConfig(
    model_name="Qwen/Qwen3-8B",
    anchor=AnchorConfig(
        use_special=True,
        use_punctuation=True,
        use_sink=True,
        use_recent=True,
        sink_size=8,           # Increase sink size
        recent_size=128,       # Increase recent window
    ),
    lambda_kl=0.5,             # Lighter KL penalty
    scheduler_mode="adaptive", # Adaptive compression
)
```

### Adaptive Compression Scheduler

Instead of curriculum (linear decrease), use adaptive mode:

```bash
python scripts/train_eit.py \
    --model Qwen/Qwen3-8B \
    --scheduler_mode adaptive \
    --initial_keep_ratio 0.9 \
    --min_keep_ratio 0.2
```

This monitors validation perplexity and automatically finds the maximum safe compression.

### Multi-GPU Training

The trainer automatically uses all available GPUs via `device_map="auto"`:

```bash
# Will use all GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_eit.py ...
```

## 📚 Citation

```bibtex
@software{sparsekv2024,
  title = {SparseKV: Training LLMs for Zero-Cost KV Cache Eviction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/Fzkuji/SparseKV}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **kvpress**: Evaluation framework for KV cache compression
- **StreamingLLM**: Inspiration for sink tokens
- **SnapKV**: Attention-based eviction baseline
- **Transformers & PEFT**: Model training infrastructure

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Contact

- GitHub Issues: [https://github.com/Fzkuji/SparseKV/issues](https://github.com/Fzkuji/SparseKV/issues)
- Email: your.email@example.com

---

**Built with ❤️ for efficient LLM inference**
