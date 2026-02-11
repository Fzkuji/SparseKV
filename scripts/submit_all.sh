#!/bin/bash
# Submit baseline or trained-model evaluation jobs.
#
# Usage:
#   bash scripts/submit_all.sh qwen3           # Phase 1: baseline Qwen3-8B
#   bash scripts/submit_all.sh llama           # Phase 1: baseline Llama
#   bash scripts/submit_all.sh qwen3_trained   # Phase 3: trained Qwen3-8B
#   bash scripts/submit_all.sh llama_trained   # Phase 3: trained Llama
set -e

MODEL_KEY=${1:-qwen3}

# Determine model path and output directory
case $MODEL_KEY in
    qwen3)
        MODEL="Qwen/Qwen3-8B"
        OUTPUT_DIR="./results/phase1_qwen3"
        ;;
    llama)
        MODEL="meta-llama/Llama-3.1-8B-Instruct"
        OUTPUT_DIR="./results/phase1_llama"
        ;;
    qwen3_trained)
        MODEL="./output/qwen3_sparsekv/final"
        OUTPUT_DIR="./results/phase3_qwen3"
        ;;
    llama_trained)
        MODEL="./output/llama_sparsekv/final"
        OUTPUT_DIR="./results/phase3_llama"
        ;;
    *)
        MODEL="$MODEL_KEY"
        OUTPUT_DIR="./results/custom_${MODEL_KEY}"
        ;;
esac

# GPU selection: override with GPUS env var, e.g. GPUS="2,3" bash scripts/submit_all.sh qwen3
# Jobs alternate between GPUs for parallel execution
GPUS=${GPUS:-"0,1"}
IFS=',' read -ra GPU_LIST <<< "$GPUS"

DATASETS=("ruler:4096" "ruler:16384" "longbench:" "aime25:")
PRESSES=("no_press:0" "snapkv:0.3" "snapkv:0.5" "snapkv:0.7" "streaming_llm:0.3" "streaming_llm:0.5" "streaming_llm:0.7" "critical_snapkv:0.3" "critical_snapkv:0.5" "critical_snapkv:0.7" "kvzip:0.3" "kvzip:0.5" "kvzip:0.7")

echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Will submit up to 4 jobs (${#DATASETS[@]} datasets x ${#PRESSES[@]} presses = $(( ${#DATASETS[@]} * ${#PRESSES[@]} )) total)"
echo ""

COUNT=0
SLURM_COUNT=0
PENDING_JOBS=()
for ds_entry in "${DATASETS[@]}"; do
    DS_NAME="${ds_entry%%:*}"
    DS_DIR="${ds_entry##*:}"

    for press_entry in "${PRESSES[@]}"; do
        PRESS="${press_entry%%:*}"
        CR="${press_entry##*:}"

        JOB_NAME="${MODEL_KEY}_${DS_NAME}_${PRESS}_${CR}"

        DATA_DIR_ARG=""
        if [ -n "$DS_DIR" ]; then
            DATA_DIR_ARG="--data_dir $DS_DIR"
        fi

        # Skip if already done (use printf to match kvpress's %.2f format)
        CR_FMT=$(printf "%.2f" "$CR")
        RESULT_NAME="${DS_NAME}__${DS_DIR:-4096}__$(echo $MODEL | sed 's|/|--|g')__${PRESS}__${CR_FMT}"
        RESULT_PATH="$HOME/kvpress/evaluation/${OUTPUT_DIR}/${RESULT_NAME}/metrics.json"
        echo "    checking: $RESULT_PATH"
        if [ -f "$RESULT_PATH" ]; then
            echo "  [skip] $JOB_NAME (done)"
            continue
        fi

        # Collect jobs to pair them (2 evals per slurm job, 1 per GPU)
        PENDING_JOBS+=("${DS_NAME}|${DATA_DIR_ARG}|${PRESS}|${CR}|${JOB_NAME}")
        COUNT=$((COUNT + 1))

        # When we have 2 pending jobs, submit them as a single slurm job
        if [ ${#PENDING_JOBS[@]} -eq 2 ]; then
            J1="${PENDING_JOBS[0]}"
            J2="${PENDING_JOBS[1]}"
            IFS='|' read -r DS1 DARG1 PR1 CR1 JN1 <<< "$J1"
            IFS='|' read -r DS2 DARG2 PR2 CR2 JN2 <<< "$J2"

            PAIR_NAME="${JN1}+${JN2}"
            cat > /tmp/job_${PAIR_NAME}.sh << HEREDOC
#!/bin/bash
#SBATCH --job-name=${PAIR_NAME}
#SBATCH --output=/home/zichuanfu2/logs/output_%j.txt
#SBATCH --error=/home/zichuanfu2/logs/error_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

eval "\$(conda shell.bash hook 2>/dev/null)" && conda activate adasparse
cd ~/kvpress/evaluation

IFS=',' read -ra _GPUS <<< "${GPUS}"

echo "=== Starting job 1: ${JN1} on GPU \${_GPUS[0]} ==="
CUDA_VISIBLE_DEVICES="\${_GPUS[0]}" python ~/SparseKV/scripts/eval_wrapper.py \\
    --model ${MODEL} \\
    --dataset ${DS1} ${DARG1} \\
    --press_name ${PR1} \\
    --compression_ratio ${CR1} \\
    --output_dir ${OUTPUT_DIR} &
PID1=\$!

echo "=== Starting job 2: ${JN2} on GPU \${_GPUS[1]} ==="
CUDA_VISIBLE_DEVICES="\${_GPUS[1]}" python ~/SparseKV/scripts/eval_wrapper.py \\
    --model ${MODEL} \\
    --dataset ${DS2} ${DARG2} \\
    --press_name ${PR2} \\
    --compression_ratio ${CR2} \\
    --output_dir ${OUTPUT_DIR} &
PID2=\$!

wait \$PID1 \$PID2
echo "=== Both jobs completed ==="
HEREDOC

            echo "  [$((COUNT-2)),$((COUNT-1))] $JN1 + $JN2 (paired)"
            sbatch /tmp/job_${PAIR_NAME}.sh
            SLURM_COUNT=$((SLURM_COUNT + 1))
            PENDING_JOBS=()
        fi

        BATCH_SIZE=${BATCH:-4}
        if [ $SLURM_COUNT -ge $BATCH_SIZE ]; then
            echo ""
            echo "--- Submitted $SLURM_COUNT slurm jobs ($COUNT eval tasks). Run again after they finish. ---"
            echo "--- Check: squeue -u zichuanfu2 ---"
            exit 0
        fi
    done
done

# Submit any remaining unpaired job
if [ ${#PENDING_JOBS[@]} -eq 1 ]; then
    J1="${PENDING_JOBS[0]}"
    IFS='|' read -r DS1 DARG1 PR1 CR1 JN1 <<< "$J1"

    cat > /tmp/job_${JN1}.sh << HEREDOC
#!/bin/bash
#SBATCH --job-name=${JN1}
#SBATCH --output=/home/zichuanfu2/logs/output_%j.txt
#SBATCH --error=/home/zichuanfu2/logs/error_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

eval "\$(conda shell.bash hook 2>/dev/null)" && conda activate adasparse
cd ~/kvpress/evaluation

CUDA_VISIBLE_DEVICES="${GPUS}" python ~/SparseKV/scripts/eval_wrapper.py \\
    --model ${MODEL} \\
    --dataset ${DS1} ${DARG1} \\
    --press_name ${PR1} \\
    --compression_ratio ${CR1} \\
    --output_dir ${OUTPUT_DIR}
HEREDOC

    echo "  [$((COUNT-1))] $JN1 (single)"
    sbatch /tmp/job_${JN1}.sh
    SLURM_COUNT=$((SLURM_COUNT + 1))
fi

echo ""
echo "All done: $SLURM_COUNT slurm jobs submitted ($COUNT eval tasks). Check: squeue -u zichuanfu2"
