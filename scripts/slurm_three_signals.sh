#!/bin/bash
#SBATCH --job-name=two_signals
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/zichuanfu2/SparseKV/logs/two_signals_%j.out
#SBATCH --error=/home/zichuanfu2/SparseKV/logs/two_signals_%j.err

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate adasparse

cd /home/zichuanfu2/SparseKV
mkdir -p logs results

python -u scripts/three_signals.py \
    --model Qwen/Qwen3-8B \
    --n_samples 100 \
    --output results/two_signals.json \
    --dataset_dir 4096 \
    --recons_chunk 2000
