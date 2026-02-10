#!/bin/bash
#SBATCH --job-name=attn_analysis
#SBATCH --output=/home/zichuanfu2/logs/output_%j.txt
#SBATCH --error=/home/zichuanfu2/logs/error_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

conda activate sparsekv

cd ~/SparseKV

# Analyze Qwen3-8B attention patterns
# Use max_len=2048 to fit eager attention in memory (2048^2 is fine)
# 50 samples should give stable statistics
python scripts/analyze_attention.py \
    --model Qwen/Qwen3-8B \
    --num_samples 50 \
    --max_len 2048 \
    --output_dir ./analysis \
    --bf16

echo "Analysis complete!"
