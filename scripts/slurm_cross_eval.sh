#!/bin/bash
#SBATCH --job-name=cross_ev
#SBATCH --output=logs/cross_eval_%j.out
#SBATCH --error=logs/cross_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=xiaowqian2
#SBATCH --time=24:00:00

cd ~/SparseKV
mkdir -p logs results
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

N_SAMPLES=100 ~/kvpress/.venv/bin/python -u scripts/cross_eval.py

echo "=== Job $SLURM_JOB_ID finished at $(date) ==="
