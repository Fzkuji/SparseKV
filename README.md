# SparseKV

**Anchor-Aware Training for Zero-Cost KV Cache Eviction**

Train LLMs to concentrate attention on *anchor tokens* (punctuation, sink, recent), so at inference time you only keep anchor tokens' KV cache — achieving effective compression with zero runtime eviction cost.

## Method

```
┌─────────────────────────────────────────────────────┐
│                    Training                          │
│                                                      │
│  Input: [The] [cat] [sat] [on] [the] [mat] [.]     │
│                                              ↑anchor │
│                                                      │
│  Teacher (frozen, full attention):                   │
│    → teacher_logits                                  │
│                                                      │
│  Student (LoRA, KV dropout mask):                    │
│    keep_ratio: 0.9 → 0.7 → 0.5 → 0.3 (curriculum)  │
│    anchor tokens: ALWAYS kept                        │
│    non-anchor: randomly dropped (increasing rate)    │
│    → student_logits                                  │
│                                                      │
│  Loss = CE(student, labels) + λ·KL(teacher, student) │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                   Inference                          │
│                                                      │
│  Only keep anchor tokens' KV cache                   │
│  → Zero-cost eviction (no scoring, no selection)     │
│  → Performance comparable to full cache              │
└─────────────────────────────────────────────────────┘
```

**Key insight**: Instead of building smarter eviction algorithms, we train the model to not need the evicted tokens in the first place.

## Installation

```bash
# Clone
git clone https://github.com/Fzkuji/SparseKV.git
cd SparseKV

# Install
pip install -e .

# Dependencies
pip install flash-attn --no-cache-dir --no-build-isolation  # Optional, for teacher model
pip install peft datasets  # For training
```

## Quick Start

```python
from sparsekv.training import SparseKVTrainer, TrainConfig

config = TrainConfig(
    model_name="Qwen/Qwen3-8B",
    output_dir="./output/qwen3_sparsekv",
)
trainer = SparseKVTrainer(config)
# ... see scripts/train_sparsekv.py for full example
```

## Complete Experiment Workflow

### Prerequisites

```bash
# On your GPU server
conda activate adasparse   # or your conda env
cd ~/SparseKV
pip install -e .
```

### Phase 0: Attention Pattern Analysis

Discover which tokens/positions naturally receive more attention.

```bash
# Submit analysis job
sbatch scripts/slurm_phase0_analyze.sh

# Check results when done
cat analysis/attention_analysis_Qwen--Qwen3-8B.json | python -m json.tool | head -50
```

**Output**: `analysis/attention_analysis_<model>.json` — per-token-type and per-position attention statistics.

### Phase 1: Baseline Evaluation

Evaluate the original (untrained) model with various KV cache eviction methods.

```bash
# Submit first batch of 4 jobs (Qwen3-8B)
bash scripts/submit_all.sh qwen3

# Check progress
squeue -u $(whoami)

# When first 4 finish, submit next 4
bash scripts/submit_all.sh qwen3

# Repeat until all 52 combinations are done
# (4 datasets × 13 press configs = 52 jobs)

# Same for Llama
bash scripts/submit_all.sh llama
```

**Press methods tested**: no_press, snapkv, streaming_llm, critical_snapkv, kvzip
**Compression ratios**: 0.3, 0.5, 0.7
**Benchmarks**: RULER 4k, RULER 16k, LongBench, AIME25

**Output**: `~/kvpress/evaluation/results/phase1_<model>/` — metrics.json + profiling.json per experiment.

### Phase 2: SparseKV Training

Train the model with anchor-aware KV dropout.

```bash
# Train Qwen3-8B
sbatch scripts/slurm_phase2_train_qwen.sh

# Train Llama-3.1-8B (after Qwen3 is done, or on a different machine)
sbatch scripts/slurm_phase2_train_llama.sh

# Monitor training
tail -f ~/logs/output_<job_id>.txt
```

**Output**: `./output/<model>_sparsekv/merged/` — merged LoRA model ready for evaluation.

### Phase 3: Evaluate Trained Model

Same evaluation as Phase 1, but with the trained model.

```bash
# Evaluate trained Qwen3-8B
bash scripts/submit_all.sh qwen3_trained

# Evaluate trained Llama
bash scripts/submit_all.sh llama_trained
```

**Output**: `~/kvpress/evaluation/results/phase1_<model>_trained/`

### Collect All Results

```bash
# View all results
find ~/kvpress/evaluation/results/ -name "metrics.json" | while read f; do
    echo "=== $(basename "$(dirname "$f")") ==="
    python -c "
import json
d=json.load(open('$f'))
vals=[v.get('string_match', v.get('score', 0)) for v in d.values() if isinstance(v, dict)]
if vals: print(f'  Avg: {sum(vals)/len(vals):.2f}')
"
done
```

## Project Structure

```
SparseKV/
├── sparsekv/
│   ├── training/
│   │   ├── anchor.py           # Anchor token definition
│   │   ├── kv_dropout.py       # KV dropout mask creation
│   │   ├── eit_trainer.py      # Main trainer (teacher-student)
│   │   ├── scheduler.py        # Compression ratio curriculum
│   │   └── loss.py             # Loss functions
│   ├── evaluation/             # Evaluation utilities
│   └── presses/                # Custom press implementations
├── scripts/
│   ├── analyze_attention.py    # Phase 0: attention analysis
│   ├── train_sparsekv.py       # Phase 2: training launch
│   ├── submit_all.sh           # Phase 1 & 3: evaluation submission
│   ├── eval_wrapper.py         # Adds profiling to kvpress evaluation
│   ├── slurm_phase0_analyze.sh
│   ├── slurm_phase2_train_qwen.sh
│   └── slurm_phase2_train_llama.sh
├── docs/
│   ├── experiment_plan.md      # Detailed experiment design
│   └── server_setup.md         # Server configuration guide
├── configs/                    # YAML configs
└── analysis/                   # Phase 0 outputs
```

## Key Design Choices

| Choice | Decision | Rationale |
|--------|----------|-----------|
| Two models | Teacher (frozen) + Student (LoRA) | Teacher provides stable target; student learns eviction robustness |
| Attention impl | Teacher: flash_attn, Student: SDPA | Flash for speed, SDPA for custom 4D mask support |
| Anchor types | Sink + Recent + Punctuation | These naturally receive high attention (see Phase 0 analysis) |
| Curriculum | 0.9 → 0.3 keep_ratio | Gradual increase prevents training collapse |
| LoRA | r=64, target=QKVO | Efficient training, <1% params |

## Citation

```
@misc{sparsekv2026,
  title={SparseKV: Anchor-Aware Training for Zero-Cost KV Cache Eviction},
  year={2026},
}
```
