#!/bin/bash
#SBATCH --job-name=baseline_llama8b
#SBATCH --output=/home/zichuanfu2/logs/output_%j.txt
#SBATCH --error=/home/zichuanfu2/logs/error_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00

conda activate sparsekv

cd ~/kvpress/evaluation

MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="./results/phase1_llama8b"
GPU="0,1"  # NVLink pair

echo "===== Baseline: $MODEL ====="

# RULER 4k
for press in no_press snapkv streaming_llm critical_snapkv kvzip; do
    for cr in 0 0.3 0.5 0.7; do
        if [ "$press" = "no_press" ] && [ "$cr" != "0" ]; then continue; fi
        if [ "$press" != "no_press" ] && [ "$cr" = "0" ]; then continue; fi
        echo ">>> RULER 4k: $press @ $cr"
        CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
            --model $MODEL --dataset ruler --data_dir 4096 \
            --press_name $press --compression_ratio $cr --fraction 0.3 --output_dir $OUTPUT_DIR
    done
done

# RULER 16k
for press in no_press snapkv streaming_llm critical_snapkv kvzip; do
    for cr in 0 0.3 0.5 0.7; do
        if [ "$press" = "no_press" ] && [ "$cr" != "0" ]; then continue; fi
        if [ "$press" != "no_press" ] && [ "$cr" = "0" ]; then continue; fi
        echo ">>> RULER 16k: $press @ $cr"
        CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
            --model $MODEL --dataset ruler --data_dir 16384 \
            --press_name $press --compression_ratio $cr --fraction 0.3 --output_dir $OUTPUT_DIR
    done
done

# LongBench
for press in no_press snapkv streaming_llm critical_snapkv kvzip; do
    for cr in 0 0.3 0.5 0.7; do
        if [ "$press" = "no_press" ] && [ "$cr" != "0" ]; then continue; fi
        if [ "$press" != "no_press" ] && [ "$cr" = "0" ]; then continue; fi
        echo ">>> LongBench: $press @ $cr"
        CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
            --model $MODEL --dataset longbench \
            --press_name $press --compression_ratio $cr --fraction 0.3 --output_dir $OUTPUT_DIR
    done
done

# AIME25
for press in no_press snapkv streaming_llm critical_snapkv; do
    for cr in 0 0.5 0.7; do
        if [ "$press" = "no_press" ] && [ "$cr" != "0" ]; then continue; fi
        if [ "$press" != "no_press" ] && [ "$cr" = "0" ]; then continue; fi
        echo ">>> AIME25: $press @ $cr"
        CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
            --model $MODEL --dataset aime25 \
            --press_name $press --compression_ratio $cr --fraction 0.3 --output_dir $OUTPUT_DIR
    done
done

echo "===== Done: $MODEL ====="
