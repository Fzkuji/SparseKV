#!/bin/bash
# AdaSparseKV Training Script
# Usage: bash scripts/train.sh [config_file]

set -e

CONFIG=${1:-"configs/train/block_dropout_8b.yaml"}

echo "=========================================="
echo "  AdaSparseKV Training"
echo "  Config: $CONFIG"
echo "=========================================="

python -m adasparse.training.train --config "$CONFIG"
