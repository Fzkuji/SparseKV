#!/bin/bash
#SBATCH --job-name=verify_kz
#SBATCH --output=logs/verify_kvzip_%j.out
#SBATCH --error=logs/verify_kvzip_%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=xiaowqian2
#SBATCH --time=24:00:00

cd ~/SparseKV
mkdir -p logs results
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

N_SAMPLES=100 python3 -u scripts/verify_kvzip.py

echo "=== Job $SLURM_JOB_ID finished at $(date) ==="
