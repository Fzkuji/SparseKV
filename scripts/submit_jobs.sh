#!/bin/bash
# Submit kvpress evaluation jobs via Slurm on AML.
# Result paths match run_all.sh exactly so run_all.sh will skip completed jobs.
#
# Uses eval_wrapper.py for LongBench v1 (kvpress native).
# Uses cross_eval_scbench.py for SCBench (custom script).
#
# AML constraints: QoS xiaowqian2, max 2 running + 2 queued, GPU 0,1 only.
#
# Usage: bash scripts/submit_jobs.sh [--longbench] [--scbench] [--dry-run]

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KVPRESS_DIR="${PROJECT_DIR}/../kvpress"
PYTHON="${KVPRESS_DIR}/.venv/bin/python"

MODEL="Qwen/Qwen3-8B"
MODEL_TAG="Qwen--Qwen3-8B"
OUTPUT_DIR="${PROJECT_DIR}/results/phase1_qwen3"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Parse flags
DO_LONGBENCH=true
DO_SCBENCH=false
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --longbench) DO_LONGBENCH=true ;;
        --scbench) DO_SCBENCH=false ;;
        --dry-run) DRY_RUN=true ;;
    esac
done
# If neither specified, do both
if ! $DO_LONGBENCH && ! $DO_SCBENCH; then
    DO_LONGBENCH=true
    DO_SCBENCH=false
fi

# Methods and compression ratios (same as run_all.sh)
PRESSES=("no_press:0" "snapkv:0.3" "snapkv:0.5" "snapkv:0.7" "snapkv:0.9" "snapkv:0.95" \
         "streaming_llm:0.3" "streaming_llm:0.5" "streaming_llm:0.7" "streaming_llm:0.9" "streaming_llm:0.95" \
         "fastkvzip:0.3" "fastkvzip:0.5" "fastkvzip:0.7" "fastkvzip:0.9" "fastkvzip:0.95" \
         "critical_snapkv:0.3" "critical_snapkv:0.5" "critical_snapkv:0.7" "critical_snapkv:0.9" "critical_snapkv:0.95")

SUBMITTED=0
SKIPPED=0

submit_one() {
    local JOB_NAME=$1
    local SCRIPT=$2

    local RUNNING
    RUNNING=$(squeue -u "$(whoami)" -h 2>/dev/null | wc -l)
    if [ "$RUNNING" -ge 4 ]; then
        echo "  [QUEUE FULL] $RUNNING jobs queued/running, stopping."
        return 1
    fi

    if $DRY_RUN; then
        echo "  [dry-run] $JOB_NAME"
        SUBMITTED=$((SUBMITTED + 1))
        return 0
    fi

    local GPU_COUNT="${3:-2}"
    local JOB_ID
    JOB_ID=$(sbatch --parsable \
        --job-name="$JOB_NAME" \
        --partition=LocalQ --qos=xiaowqian2 \
        --gres=gpu:${GPU_COUNT} --cpus-per-task=8 --mem=80G \
        --time=24:00:00 \
        --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
        --wrap="$SCRIPT")
    echo "  [submit] $JOB_NAME → job $JOB_ID"
    SUBMITTED=$((SUBMITTED + 1))
}

# ============================================================
# Part 1: LongBench v1 (kvpress native, via eval_wrapper.py)
# Each subset is a separate job. data_dir = subset config name.
# Result dir: longbench__{subset}__{MODEL_TAG}__{press}__{cr}/metrics.json
# ============================================================
# LongBench v1 subsets (matching run_all.sh convention)
LONGBENCH_SUBSETS=(
    narrativeqa qasper multifieldqa_en
    hotpotqa 2wikimqa musique
    gov_report qmsum multi_news
    trec triviaqa samsum
    passage_count passage_retrieval_en
    lcc repobench-p
)

if $DO_LONGBENCH; then
    echo "=== LongBench v1 ==="
    for press_entry in "${PRESSES[@]}"; do
        PRESS="${press_entry%%:*}"
        CR="${press_entry##*:}"
        CR_FMT=$(printf "%.2f" "$CR")

        for SUBSET in "${LONGBENCH_SUBSETS[@]}"; do
            RESULT_NAME="longbench__${SUBSET}__${MODEL_TAG}__${PRESS}__${CR_FMT}"
            RESULT_DIR="${OUTPUT_DIR}/${RESULT_NAME}"

            if [ -f "${RESULT_DIR}/metrics.json" ]; then
                echo "  [skip] lb_${PRESS}_${CR_FMT}_${SUBSET} (done)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            JOB_NAME="lb_${PRESS}_${CR_FMT}_${SUBSET}"
            if [ "$PRESS" = "fastkvzip" ]; then CUDA_DEV="0"; GPU_N=1; else CUDA_DEV="0,1"; GPU_N=2; fi
            SCRIPT="export CUDA_VISIBLE_DEVICES=${CUDA_DEV} && \\
cd ${KVPRESS_DIR}/evaluation && \\
${PYTHON} ${PROJECT_DIR}/scripts/eval_wrapper.py \\
  --model_tag ${MODEL_TAG} \\
  --config_file /dev/null \\
  --model ${MODEL} \\
  --dataset longbench \\
  --data_dir ${SUBSET} \\
  --press_name ${PRESS} \\
  --compression_ratio ${CR} \\
  --output_dir ${OUTPUT_DIR}"

            submit_one "$JOB_NAME" "$SCRIPT" "$GPU_N" || break 2
        done
    done
    echo ""
fi

# ============================================================
# Part 2: SCBench (custom script, all 11 tasks)
# Result dir: scbench__{task}__Qwen--Qwen3-8B__{press}__{cr}/metrics.json
# We save in a compatible format: create the dir + metrics.json
# ============================================================
SCBENCH_TASKS=(
    scbench_kv scbench_prefix_suffix scbench_vt scbench_repoqa
    scbench_qa_eng scbench_choice_eng scbench_many_shot
    scbench_summary scbench_mf scbench_summary_with_needles scbench_repoqa_and_kv
)

if $DO_SCBENCH; then
    echo "=== SCBench ==="
    for press_entry in "${PRESSES[@]}"; do
        PRESS="${press_entry%%:*}"
        CR="${press_entry##*:}"
        CR_FMT=$(printf "%.2f" "$CR")

        for TASK in "${SCBENCH_TASKS[@]}"; do
            RESULT_NAME="${TASK}__${MODEL_TAG}__${PRESS}__${CR_FMT}"
            RESULT_DIR="${OUTPUT_DIR}/${RESULT_NAME}"

            if [ -f "${RESULT_DIR}/metrics.json" ]; then
                echo "  [skip] sc_${PRESS}_${CR_FMT}_${TASK} (done)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            JOB_NAME="sc_${PRESS}_${CR_FMT}_${TASK##scbench_}"
            if [ "$PRESS" = "fastkvzip" ]; then CUDA_DEV="0"; GPU_N=1; else CUDA_DEV="0,1"; GPU_N=2; fi
            SCRIPT="export CUDA_VISIBLE_DEVICES=${CUDA_DEV} && \\
cd ${PROJECT_DIR} && \\
${PYTHON} -u scripts/cross_eval_scbench.py \\
  --task ${TASK} \\
  --press_name ${PRESS} \\
  --compression_ratio ${CR} \\
  --model ${MODEL} \\
  --model_tag ${MODEL_TAG} \\
  --output_dir ${OUTPUT_DIR} \\
  --max_seq_length 170000"

            submit_one "$JOB_NAME" "$SCRIPT" "$GPU_N" || break 2
        done
    done
    echo ""
fi

echo "============================================================"
echo "  Submitted: ${SUBMITTED}  Skipped: ${SKIPPED}"
echo "  Results: ${OUTPUT_DIR}"
echo "============================================================"
