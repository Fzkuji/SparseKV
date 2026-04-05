#!/bin/bash
#SBATCH --job-name=qa_signal
#SBATCH --output=logs/qa_signal_%j.out
#SBATCH --error=logs/qa_signal_%j.err
#SBATCH --gres=gpu:2
#SBATCH --qos=xiaowqian2
#SBATCH --time=24:00:00

cd ~/SparseKV
mkdir -p logs results
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run 2 shards in parallel on 2 GPUs
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/qa_signal_kvpress.py \
    --model Qwen/Qwen3-8B \
    --n_samples 100 \
    --output results/qa_signal_ruler4096_s0.json \
    --dataset_dir 4096 \
    --shard 0 --n_shards 2 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python3 -u scripts/qa_signal_kvpress.py \
    --model Qwen/Qwen3-8B \
    --n_samples 100 \
    --output results/qa_signal_ruler4096_s1.json \
    --dataset_dir 4096 \
    --shard 1 --n_shards 2 &
PID1=$!

wait $PID0 $PID1

# Merge results
python3 -c "
import json
r0 = json.load(open('results/qa_signal_ruler4096_s0.json'))
r1 = json.load(open('results/qa_signal_ruler4096_s1.json'))
merged = r0 + r1
json.dump(merged, open('results/qa_signal_ruler4096.json','w'), indent=2)
print(f'Merged {len(r0)} + {len(r1)} = {len(merged)} results')

# Print summary
ratios = [0.3, 0.5, 0.7, 0.9, 0.95]
kvzip_ref = {0.30: 95.23, 0.50: 95.21, 0.70: 95.15, 0.90: 87.22, 0.95: 37.65}
fk = [r for r in merged if r['method'] == 'full_kv']
fk_acc = sum(r['correct'] for r in fk) / len(fk) * 100
print(f'Full KV: {fk_acc:.1f}%')
print(f\"{'Ratio':>8} | {'QA Signal':>12} | {'KVzip(ref)':>12}\")
for ratio in ratios:
    mr = [r for r in merged if r['method'] == 'qa_signal' and r['ratio'] == ratio]
    qa_acc = sum(r['correct'] for r in mr) / len(mr) * 100 if mr else 0
    kz = kvzip_ref.get(ratio, '-')
    print(f'{ratio:>8.2f} | {qa_acc:>11.1f}% | {kz:>11}%')
"

echo "=== Job $SLURM_JOB_ID finished at $(date) ==="
