#!/bin/bash
# Evaluate baseline model (no training) with various eviction methods
set -e

python -m sparsekv.evaluation.evaluate \
    --config configs/eval/baseline.yaml
