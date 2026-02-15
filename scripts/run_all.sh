#!/bin/bash
# Run all baseline evaluations on a multi-GPU server (no slurm).
# Uses a GPU pool: as soon as any GPU finishes, the next job is dispatched.
# kvzip jobs run serially (one at a time) to avoid tokenizer cache conflicts.
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

# Cleanup: kill entire process group on exit (Ctrl+C, kill, or error)
cleanup() {
    echo ""
    echo "Caught signal, killing all child processes..."
    trap - SIGINT SIGTERM EXIT  # prevent re-entry
    pkill -TERM -P $$ 2>/dev/null  # kill all descendants
    sleep 1
    pkill -KILL -P $$ 2>/dev/null  # force kill stragglers
    exit 1
}
trap cleanup SIGINT SIGTERM EXIT

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

# Presses that must run serially (one GPU at a time) due to resource conflicts
SERIAL_PRESSES="kvzip"

# Build list of pending jobs (parallel and serial separately)
PARALLEL_JOBS=()
SERIAL_JOBS=()
SKIPPED=0
for ds_entry in "${DATASETS[@]}"; do
    DS_NAME="${ds_entry%%:*}"
    DS_DIR="${ds_entry##*:}"

    for press_entry in "${PRESSES[@]}"; do
        PRESS="${press_entry%%:*}"
        CR="${press_entry##*:}"
        CR_FMT=$(printf "%.2f" "$CR")

        JOB_NAME="${DS_NAME}_${DS_DIR:-default}_${PRESS}_${CR_FMT}"

        if [ -n "$DS_DIR" ]; then
            DATA_DIR_ARG="--data_dir $DS_DIR"
        else
            DATA_DIR_ARG=""
        fi

        # Check if already done
        if [ -n "$DS_DIR" ]; then
            RESULT_NAME="${DS_NAME}__${DS_DIR}__$(echo $MODEL | sed 's|/|--|g')__${PRESS}__${CR_FMT}"
        else
            RESULT_NAME="${DS_NAME}__$(echo $MODEL | sed 's|/|--|g')__${PRESS}__${CR_FMT}"
        fi
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

        JOB_ENTRY="${JOB_NAME}|${DS_NAME}|${DATA_DIR_ARG}|${PRESS}|${CR}"
        if [[ "$SERIAL_PRESSES" == *"$PRESS"* ]]; then
            SERIAL_JOBS+=("$JOB_ENTRY")
        else
            PARALLEL_JOBS+=("$JOB_ENTRY")
        fi
    done
done

TOTAL_PARALLEL=${#PARALLEL_JOBS[@]}
TOTAL_SERIAL=${#SERIAL_JOBS[@]}
TOTAL_JOBS=$((TOTAL_PARALLEL + TOTAL_SERIAL))
echo ""
echo "Pending: ${TOTAL_JOBS} (${TOTAL_PARALLEL} parallel + ${TOTAL_SERIAL} serial)  Skipped: ${SKIPPED}"
echo ""

if [ "$TOTAL_JOBS" -eq 0 ]; then
    echo "All jobs already completed!"
    exit 0
fi

# Change to kvpress evaluation dir (evaluate.py must be in cwd)
cd "${KVPRESS_DIR}/evaluation"

# --- Helper: launch a job on a specific GPU ---
launch_job() {
    local GPU=$1
    local JOB_NAME=$2
    local DS_NAME=$3
    local DATA_DIR_ARG=$4
    local PRESS=$5
    local CR=$6
    local LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

    echo "  [gpu:${GPU}] ${JOB_NAME}"

    CUDA_VISIBLE_DEVICES="${GPU}" python "${PROJECT_DIR}/scripts/eval_wrapper.py" \
        --config_file /dev/null \
        --model "${MODEL}" \
        --dataset "${DS_NAME}" ${DATA_DIR_ARG} \
        --press_name "${PRESS}" \
        --compression_ratio "${CR}" \
        --output_dir "${OUTPUT_DIR}" \
        > "${LOG_FILE}" 2>&1 &
}

LAUNCHED=0
FAILED=0

# --- Phase 1: Run parallel jobs with GPU pool ---
if [ "$TOTAL_PARALLEL" -gt 0 ]; then
    echo "=== Phase 1: Parallel jobs (${TOTAL_PARALLEL}) ==="

    # Track which GPU is running which PID
    declare -A GPU_PID    # GPU -> PID
    declare -A PID_NAME   # PID -> job name
    declare -A PID_GPU    # PID -> GPU

    # Initialize all GPUs as free
    FREE_GPUS=("${GPU_LIST[@]}")

    JOB_IDX=0
    while [ "$JOB_IDX" -lt "$TOTAL_PARALLEL" ] || [ "${#PID_NAME[@]}" -gt 0 ]; do
        # Fill free GPUs with pending jobs
        while [ "${#FREE_GPUS[@]}" -gt 0 ] && [ "$JOB_IDX" -lt "$TOTAL_PARALLEL" ]; do
            GPU="${FREE_GPUS[0]}"
            FREE_GPUS=("${FREE_GPUS[@]:1}")  # pop first

            IFS='|' read -r JOB_NAME DS_NAME DATA_DIR_ARG PRESS CR <<< "${PARALLEL_JOBS[$JOB_IDX]}"
            launch_job "$GPU" "$JOB_NAME" "$DS_NAME" "$DATA_DIR_ARG" "$PRESS" "$CR"
            PID=$!
            GPU_PID[$GPU]=$PID
            PID_NAME[$PID]="$JOB_NAME"
            PID_GPU[$PID]="$GPU"
            LAUNCHED=$((LAUNCHED + 1))
            JOB_IDX=$((JOB_IDX + 1))
        done

        # Wait for any one process to finish
        if [ "${#PID_NAME[@]}" -gt 0 ]; then
            # Collect all active PIDs
            ACTIVE_PIDS=("${!PID_NAME[@]}")

            # Wait for any to finish (poll every 5 seconds)
            while true; do
                for pid in "${ACTIVE_PIDS[@]}"; do
                    if ! kill -0 "$pid" 2>/dev/null; then
                        # Process finished, check exit code
                        EXIT_CODE=0
                        wait "$pid" 2>/dev/null || EXIT_CODE=$?
                        FINISHED_NAME="${PID_NAME[$pid]}"
                        FINISHED_GPU="${PID_GPU[$pid]}"

                        if [ "$EXIT_CODE" -ne 0 ]; then
                            echo "  [FAIL] ${FINISHED_NAME}"
                            FAILED=$((FAILED + 1))
                        else
                            echo "  [done] ${FINISHED_NAME}"
                        fi

                        # Free the GPU
                        FREE_GPUS+=("$FINISHED_GPU")
                        unset GPU_PID[$FINISHED_GPU]
                        unset PID_NAME[$pid]
                        unset PID_GPU[$pid]
                    fi
                done
                # If any GPU was freed, break to dispatch new jobs
                if [ "${#FREE_GPUS[@]}" -gt 0 ]; then
                    break
                fi
                sleep 5
            done
        fi
    done
    echo ""
fi

# --- Phase 2: Run serial jobs one at a time ---
if [ "$TOTAL_SERIAL" -gt 0 ]; then
    echo "=== Phase 2: Serial jobs (${TOTAL_SERIAL}) ==="

    for ((i=0; i<TOTAL_SERIAL; i++)); do
        # Use GPU 0 for serial jobs
        GPU="${GPU_LIST[0]}"

        IFS='|' read -r JOB_NAME DS_NAME DATA_DIR_ARG PRESS CR <<< "${SERIAL_JOBS[$i]}"
        LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

        echo "  [gpu:${GPU}] ${JOB_NAME} (serial $((i+1))/${TOTAL_SERIAL})"

        CUDA_VISIBLE_DEVICES="${GPU}" python "${PROJECT_DIR}/scripts/eval_wrapper.py" \
            --config_file /dev/null \
            --model "${MODEL}" \
            --dataset "${DS_NAME}" ${DATA_DIR_ARG} \
            --press_name "${PRESS}" \
            --compression_ratio "${CR}" \
            --output_dir "${OUTPUT_DIR}" \
            > "${LOG_FILE}" 2>&1

        EXIT_CODE=$?
        LAUNCHED=$((LAUNCHED + 1))

        if [ "$EXIT_CODE" -ne 0 ]; then
            echo "  [FAIL] ${JOB_NAME}"
            FAILED=$((FAILED + 1))
        else
            echo "  [done] ${JOB_NAME}"
        fi
    done
    echo ""
fi

# Disable cleanup trap on normal exit
trap - SIGINT SIGTERM EXIT

echo "============================================================"
echo "  COMPLETE"
echo "  Launched: ${LAUNCHED}  Skipped: ${SKIPPED}  Failed: ${FAILED}"
echo "  Results: ${OUTPUT_DIR}"
echo "  Logs: ${LOG_DIR}"
echo "============================================================"
