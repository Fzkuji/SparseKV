#!/bin/bash
#SBATCH --job-name=qa_press
#SBATCH --output=logs/qa_press_%j.out
#SBATCH --error=logs/qa_press_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=xiaowqian2
#SBATCH --time=24:00:00

cd ~/SparseKV
mkdir -p logs results
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u scripts/qa_signal_press.py \
    --model Qwen/Qwen3-8B \
    --n_samples 100 \
    --output results/qa_signal_ruler4096.json \
    --dataset_dir 4096

echo "=== Job $SLURM_JOB_ID finished at $(date) ==="
