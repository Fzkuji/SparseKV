#!/bin/bash
#SBATCH --job-name=three_signals
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/zichuanfu2/SparseKV/logs/three_signals_%j.out
#SBATCH --error=/home/zichuanfu2/SparseKV/logs/three_signals_%j.err

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate adasparse

cd /home/zichuanfu2/SparseKV
mkdir -p logs results

python -u scripts/three_signals.py \
    --model Qwen/Qwen3-8B \
    --n_samples 100 \
    --output results/three_signals.json \
    --dataset_dir 4096
