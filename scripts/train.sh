#!/bin/bash
# SparseKV Training Script
# Usage: bash scripts/train.sh [config_file]

set -e

CONFIG=${1:-"configs/train/block_dropout_8b.yaml"}

echo "=========================================="
echo "  SparseKV Training"
echo "  Config: $CONFIG"
echo "=========================================="

python -m sparsekv.training.train --config "$CONFIG"
