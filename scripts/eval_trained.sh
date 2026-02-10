#!/bin/bash
# Evaluate trained model and compare with baseline
set -e

MODEL_PATH=${1:-"./output/block_dropout_8b"}

python -m sparsekv.evaluation.evaluate \
    --model_path "$MODEL_PATH" \
    --baseline_model meta-llama/Llama-3.1-8B-Instruct \
    --press_names snapkv expected_attention streaming_llm knorm \
    --compression_ratios 0.3 0.5 0.7 \
    --eval_sparsity \
    --output_dir "./eval_output/$(basename $MODEL_PATH)"
