#!/bin/bash
# 提交所有 baseline 评测任务
# 用法: bash scripts/submit_all.sh [model_key]
#   model_key: qwen3 (默认), llama, gpt-oss
set -e

MODEL_KEY=${1:-qwen3}

case $MODEL_KEY in
    qwen3)   MODEL="Qwen/Qwen3-8B" ;;
    llama)   MODEL="meta-llama/Llama-3.1-8B-Instruct" ;;
    gpt-oss) MODEL="openai/gpt-oss-20b" ;;
    *)       MODEL="$MODEL_KEY" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="./results/phase1_${MODEL_KEY}"

DATASETS=("ruler:4096" "ruler:16384" "longbench:" "aime25:")
PRESSES=("no_press:0" "snapkv:0.3" "snapkv:0.5" "snapkv:0.7" "streaming_llm:0.3" "streaming_llm:0.5" "streaming_llm:0.7" "critical_snapkv:0.3" "critical_snapkv:0.5" "critical_snapkv:0.7" "kvzip:0.3" "kvzip:0.5" "kvzip:0.7")

echo "Model: $MODEL"
echo "Will submit ${#DATASETS[@]} x ${#PRESSES[@]} = $(( ${#DATASETS[@]} * ${#PRESSES[@]} )) jobs"
echo ""

COUNT=0
for ds_entry in "${DATASETS[@]}"; do
    DS_NAME="${ds_entry%%:*}"
    DS_DIR="${ds_entry##*:}"
    
    for press_entry in "${PRESSES[@]}"; do
        PRESS="${press_entry%%:*}"
        CR="${press_entry##*:}"
        
        JOB_NAME="${MODEL_KEY}_${DS_NAME}_${PRESS}_${CR}"
        
        # 构建 data_dir 参数
        DATA_DIR_ARG=""
        if [ -n "$DS_DIR" ]; then
            DATA_DIR_ARG="--data_dir $DS_DIR"
        fi

        cat > /tmp/job_${JOB_NAME}.sh << HEREDOC
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=/home/zichuanfu2/logs/output_%j.txt
#SBATCH --error=/home/zichuanfu2/logs/error_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

conda activate sparsekv

cd ~/kvpress/evaluation

CUDA_VISIBLE_DEVICES="0,1" python ~/SparseKV/scripts/eval_wrapper.py \\
    --model ${MODEL} \\
    --dataset ${DS_NAME} ${DATA_DIR_ARG} \\
    --press_name ${PRESS} \\
    --compression_ratio ${CR} \\
    \\
    --output_dir ${OUTPUT_DIR}
HEREDOC

        # 构建结果目录名，跳过已完成的
        RESULT_NAME="${DS_NAME}__${DS_DIR}__$(echo $MODEL | tr '/' '--')__${PRESS}__${CR}"
        if [ -z "$DS_DIR" ]; then
            RESULT_NAME="${DS_NAME}__4096__$(echo $MODEL | tr '/' '--')__${PRESS}__${CR}"
        fi
        RESULT_PATH="$(cd ~/kvpress/evaluation && pwd)/${OUTPUT_DIR}/${RESULT_NAME}/metrics.json"
        
        if [ -f "$RESULT_PATH" ]; then
            echo "  [skip] $JOB_NAME (already done)"
            continue
        fi
        
        echo "  [$COUNT] $JOB_NAME"
        sbatch /tmp/job_${JOB_NAME}.sh
        COUNT=$((COUNT + 1))
        
        # 每 4 个暂停，等用户确认
        if [ $((COUNT % 4)) -eq 0 ]; then
            echo ""
            echo "--- Submitted $COUNT jobs (limit 4). Wait for these to finish, then run again. ---"
            echo "--- Check: squeue -u zichuanfu2 ---"
            exit 0
        fi
    done
done

echo ""
echo "All $COUNT jobs submitted. Check: squeue -u zichuanfu2"
