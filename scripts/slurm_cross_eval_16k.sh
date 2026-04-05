#!/bin/bash
#SBATCH --job-name=cross16k
#SBATCH --output=logs/cross_eval_16k_%j.out
#SBATCH --error=logs/cross_eval_16k_%j.err
#SBATCH --gres=gpu:2
#SBATCH --qos=xiaowqian2
#SBATCH --time=24:00:00

cd ~/SparseKV
mkdir -p logs results
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RULER_LEN=16384 N_SAMPLES=100 ~/kvpress/.venv/bin/python -u scripts/cross_eval.py

echo "=== Job $SLURM_JOB_ID finished at $(date) ==="
