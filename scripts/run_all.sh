#!/bin/bash
# Run all baseline evaluations on a multi-GPU server (no slurm).
# Strict batch mode: one job per GPU, wait for entire batch to finish before next.
#
# Assumes kvpress repo is at ../kvpress relative to this project (sibling directories).
#
# Usage:
#   bash scripts/run_all.sh
#
# Environment variables:
#   MODEL         - model name/path (default: Qwen/Qwen3-8B)
#   GPUS          - comma-separated GPU ids (default: all available)

set -e

# Note: don't set HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE - breaks datasets cache lookup

# Project root (where this script lives: SparseKV/scripts/)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KVPRESS_DIR="$(cd "${PROJECT_DIR}/../kvpress" && pwd)"

MODEL=${MODEL:-Qwen/Qwen3-8B}
OUTPUT_DIR="${PROJECT_DIR}/results/phase1_qwen3"
LOG_DIR="${PROJECT_DIR}/results/phase1_qwen3/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Auto-detect GPUs
if [ -z "$GPUS" ]; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
    GPUS=$(seq -s, 0 $((NUM_GPUS-1)))
fi
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

echo "Model:        ${MODEL}"
echo "Output:       ${OUTPUT_DIR}"
echo "GPUs:         ${GPUS} (${NUM_GPUS} total)"
echo "Logs:         ${LOG_DIR}"
echo ""

DATASETS=("ruler:4096" "ruler:16384" "longbench-v2:" "infinitebench:")
PRESSES=("no_press:0" "snapkv:0.3" "snapkv:0.5" "snapkv:0.7" "streaming_llm:0.3" "streaming_llm:0.5" "streaming_llm:0.7" "critical_snapkv:0.3" "critical_snapkv:0.5" "critical_snapkv:0.7" "kvzip:0.3" "kvzip:0.5" "kvzip:0.7")

# Build list of pending jobs
JOBS=()
SKIPPED=0
for ds_entry in "${DATASETS[@]}"; do
    DS_NAME="${ds_entry%%:*}"
    DS_DIR="${ds_entry##*:}"

    for press_entry in "${PRESSES[@]}"; do
        PRESS="${press_entry%%:*}"
        CR="${press_entry##*:}"
        CR_FMT=$(printf "%.2f" "$CR")

        JOB_NAME="${DS_NAME}_${DS_DIR:-default}_${PRESS}_${CR_FMT}"

        DATA_DIR_ARG=""
        if [ -n "$DS_DIR" ]; then
            DATA_DIR_ARG="--data_dir $DS_DIR"
        fi

        # Check if already done
        RESULT_NAME="${DS_NAME}__${DS_DIR:-4096}__$(echo $MODEL | sed 's|/|--|g')__${PRESS}__${CR_FMT}"
        RESULT_DIR="${OUTPUT_DIR}/${RESULT_NAME}"
        if [ -f "${RESULT_DIR}/metrics.json" ]; then
            echo "[skip] ${JOB_NAME} (done)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Clean up incomplete/failed results
        if [ -d "$RESULT_DIR" ]; then
            rm -rf "$RESULT_DIR"
        fi

        JOBS+=("${JOB_NAME}|${DS_NAME}|${DATA_DIR_ARG}|${PRESS}|${CR}")
    done
done

TOTAL_JOBS=${#JOBS[@]}
echo ""
echo "Pending: ${TOTAL_JOBS}  Skipped: ${SKIPPED}"
echo ""

if [ "$TOTAL_JOBS" -eq 0 ]; then
    echo "All jobs already completed!"
    exit 0
fi

# Change to kvpress evaluation dir (evaluate.py must be in cwd)
cd "${KVPRESS_DIR}/evaluation"

# Run in batches of NUM_GPUS
LAUNCHED=0
FAILED=0
for ((batch_start=0; batch_start<TOTAL_JOBS; batch_start+=NUM_GPUS)); do
    PIDS=()
    BATCH_NAMES=()

    for ((i=0; i<NUM_GPUS && batch_start+i<TOTAL_JOBS; i++)); do
        idx=$((batch_start + i))
        GPU=${GPU_LIST[$i]}

        IFS='|' read -r JOB_NAME DS_NAME DATA_DIR_ARG PRESS CR <<< "${JOBS[$idx]}"
        LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

        echo "[gpu:${GPU}] ${JOB_NAME}"

        CUDA_VISIBLE_DEVICES="${GPU}" python "${PROJECT_DIR}/scripts/eval_wrapper.py" \
            --model "${MODEL}" \
            --dataset "${DS_NAME}" ${DATA_DIR_ARG} \
            --press_name "${PRESS}" \
            --compression_ratio "${CR}" \
            --output_dir "${OUTPUT_DIR}" \
            > "${LOG_FILE}" 2>&1 &

        PIDS+=($!)
        BATCH_NAMES+=("${JOB_NAME}")
        LAUNCHED=$((LAUNCHED + 1))
    done

    echo "  Waiting for batch (${#PIDS[@]} jobs)..."
    for j in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$j]}"; then
            echo "  [FAIL] ${BATCH_NAMES[$j]}"
            FAILED=$((FAILED + 1))
        else
            echo "  [done] ${BATCH_NAMES[$j]}"
        fi
    done
    echo ""
done

echo "============================================================"
echo "  COMPLETE"
echo "  Launched: ${LAUNCHED}  Skipped: ${SKIPPED}  Failed: ${FAILED}"
echo "  Results: ${OUTPUT_DIR}"
echo "  Logs: ${LOG_DIR}"
echo "============================================================"
