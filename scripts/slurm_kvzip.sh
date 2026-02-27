#!/bin/bash
#SBATCH --job-name=kvzip_ruler
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/zichuanfu2/SparseKV/logs/kvzip_ruler_%j.out
#SBATCH --error=/home/zichuanfu2/SparseKV/logs/kvzip_ruler_%j.err

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate adasparse

cd /home/zichuanfu2/SparseKV
mkdir -p logs results

python -u scripts/two_signals_kvpress.py \
    --model Qwen/Qwen3-8B \
    --n_samples 100 \
    --output results/kvzip_ruler.json \
    --dataset_dir 4096
