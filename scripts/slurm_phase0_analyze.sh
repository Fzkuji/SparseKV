#!/bin/bash
#SBATCH --job-name=phase0_analyze
#SBATCH --output=/home/zichuanfu2/logs/output_%j.txt
#SBATCH --error=/home/zichuanfu2/logs/error_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

conda activate adasparse
cd ~/SparseKV

python scripts/analyze_attention.py \
    --model Qwen/Qwen3-8B \
    --num_samples 50 \
    --max_len 2048 \
    --output_dir ./analysis \
    --bf16

echo "Phase 0 analysis complete!"
