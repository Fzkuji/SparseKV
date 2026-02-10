#!/bin/bash
#SBATCH --job-name=phase2_train_llama
#SBATCH --output=/home/zichuanfu2/logs/output_%j.txt
#SBATCH --error=/home/zichuanfu2/logs/error_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

conda activate adasparse

cd ~/SparseKV

echo "=== Phase 2: SparseKV Training (Llama-3.1-8B) ==="

CUDA_VISIBLE_DEVICES="0,1" python scripts/train_sparsekv.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output_dir ./output/llama_sparsekv \
    --lr 2e-5 \
    --epochs 1 \
    --batch_size 1 \
    --grad_accum 8 \
    --max_seq_len 4096 \
    --num_train_samples 10000 \
    --num_val_samples 500 \
    --lora_r 64 \
    --lambda_kl 1.0 \
    --initial_keep_ratio 0.9 \
    --min_keep_ratio 0.3 \
    --scheduler_mode curriculum \
    --sink_size 4 \
    --recent_size 64

echo "Training complete! Model saved to ./output/llama_sparsekv/"
