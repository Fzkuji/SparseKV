#!/bin/bash
#SBATCH --job-name=cross_8k
#SBATCH --output=logs/cross_eval_8k_%j.out
#SBATCH --error=logs/cross_eval_8k_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=xiaowqian2
#SBATCH --time=24:00:00

cd ~/SparseKV
mkdir -p logs results
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RULER_LEN=8192 N_SAMPLES=100 ~/kvpress/.venv/bin/python -u scripts/cross_eval.py

echo "=== Job $SLURM_JOB_ID finished at $(date) ==="
