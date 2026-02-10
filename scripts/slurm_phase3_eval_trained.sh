#!/bin/bash
#SBATCH --job-name=sparsekv_eval_trained
#SBATCH --output=logs/phase3_eval_trained_%j.out
#SBATCH --error=logs/phase3_eval_trained_%j.err
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# Phase 3: Evaluate Trained SparseKV Model
# Tests all press methods on the trained model

set -e

# Create logs directory
mkdir -p logs

# Environment
echo "=========================================="
echo "Phase 3: Evaluate Trained Model"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo ""

# Activate environment
if conda info --envs | grep -q "^sparsekv "; then
    echo "Activating conda environment: sparsekv"
    source activate sparsekv
elif conda info --envs | grep -q "^adasparse "; then
    echo "Activating conda environment: adasparse"
    source activate adasparse
else
    echo "ERROR: Neither 'sparsekv' nor 'adasparse' conda environment found!"
    exit 1
fi

# GPU selection
export CUDA_VISIBLE_DEVICES="0,1"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Configuration
TRAINED_MODEL="./output/qwen3_sparsekv/final"
MODEL_TYPE="qwen3_trained"

if [ ! -d "$TRAINED_MODEL" ]; then
    echo "ERROR: Trained model not found at $TRAINED_MODEL"
    echo "Please run Phase 2 training first: sbatch scripts/slurm_phase2_train_qwen.sh"
    exit 1
fi

echo "Evaluating trained model: $TRAINED_MODEL"
echo "Model type: $MODEL_TYPE"
echo ""

# Use submit_all.sh with the trained model type
bash scripts/submit_all.sh "$MODEL_TYPE"

echo ""
echo "=========================================="
echo "Phase 3 Complete"
echo "=========================================="
echo "Results saved in evaluation outputs for model: $MODEL_TYPE"
echo ""
