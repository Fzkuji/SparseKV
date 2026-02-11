# 新服务器配置指南

## 1. 创建 Conda 环境

```bash
conda create -n sparsekv python=3.11 -y
conda activate sparsekv
echo "conda activate sparsekv" >> ~/.bashrc
```

## 2. 安装 PyTorch

```bash
# CUDA 12.1 (根据服务器 CUDA 版本调整)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 3. 安装 Flash Attention

```bash
pip install flash-attn --no-cache-dir --no-build-isolation
```

## 4. 克隆并安装 kvpress

```bash
cd ~
git clone https://github.com/NVIDIA/kvpress.git
cd kvpress
pip install -e ".[eval]"
```

## 5. 克隆 SparseKV

```bash
cd ~
git clone https://github.com/Fzkuji/SparseKV.git
```

## 6. 登录 HuggingFace（Llama 需要）

```bash
git config --global credential.helper store
python -c "from huggingface_hub import login; login(token='你的HF_TOKEN')"
```

> 需要先在 https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct 申请并获得 access。

## 7. 创建日志目录

```bash
mkdir -p ~/logs
```

## 8. 验证安装

```bash
python -c "
import torch
import kvpress
import flash_attn
from huggingface_hub import whoami
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
print(f'Flash Attention: {flash_attn.__version__}')
print(f'HuggingFace user: {whoami()[\"name\"]}')
print('All good!')
"
```

## 9. 运行评测

### 基本用法

```bash
cd ~/SparseKV

# Phase 1: Baseline 评测
bash scripts/submit_all.sh qwen3          # Qwen3-8B
bash scripts/submit_all.sh llama          # Llama-3.1-8B-Instruct

# Phase 3: 训练后模型评测
bash scripts/submit_all.sh qwen3_trained
bash scripts/submit_all.sh llama_trained
```

### 参数说明

所有参数通过环境变量设置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `GPUS` | `0,1` | 使用的 GPU 编号，逗号分隔 |
| `BATCH` | `16` | 每次提交的最大 job 数 |

```bash
# 使用 GPU 2 和 3，每次提交 16 个 job
GPUS="2,3" bash scripts/submit_all.sh qwen3

# 使用单张卡，提交 8 个 job
GPUS="0" BATCH=8 bash scripts/submit_all.sh qwen3

# 一次提交全部剩余任务
GPUS="2,3" BATCH=999 bash scripts/submit_all.sh qwen3
```

每个 job 使用 1 张 GPU，多张卡时 job 会交替分配到不同卡上并行执行。例如 `GPUS="2,3"` 时，奇数 job 用卡 2，偶数 job 用卡 3。

### 断点续跑

脚本会自动跳过已完成的任务（检查 `metrics.json` 是否存在）。每次提交一批 job，跑完后再执行一次同样的命令即可继续：

```bash
# 查看当前任务状态
squeue -u $(whoami)

# 任务跑完后，再次执行会自动跳过已完成的
GPUS="2,3" bash scripts/submit_all.sh qwen3
```

### 评测矩阵

每个模型会测试以下组合（4 datasets × 13 press configs = 52 个条件）：

**Datasets:**
- RULER 4k, RULER 16k, LongBench, AIME25

**Eviction methods × compression ratios:**
- no_press (baseline, 无压缩)
- snapkv: 0.3, 0.5, 0.7
- streaming_llm: 0.3, 0.5, 0.7
- critical_snapkv: 0.3, 0.5, 0.7
- kvzip: 0.3, 0.5, 0.7

## 10. 查看结果

```bash
# 查看已完成的条件
ls ~/kvpress/evaluation/results/phase1_qwen3/*/metrics.json 2>/dev/null | \
    sed 's|.*/phase1_qwen3/||;s|/metrics.json||'

# 查看所有结果分数
find ~/kvpress/evaluation/results/phase1_qwen3 -name "metrics.json" | sort | while read f; do
    echo "=== $(basename "$(dirname "$f")") ==="
    cat "$f" | python -c "
import sys,json
d=json.load(sys.stdin)
vals=[v.get('string_match', v.get('score', 0)) for v in d.values() if isinstance(v, dict)]
if vals: print(f'  Avg: {sum(vals)/len(vals):.2f}')
"
done

# 查看 profiling（时间/显存/吞吐量）
find ~/kvpress/evaluation/results/phase1_qwen3 -name "profiling.json" -exec cat {} \;
```

## 环境要求

| 项目 | 最低要求 |
|------|---------|
| GPU | 1× A100 80GB（单 job）|
| CUDA | 12.1+ |
| 内存 | 60GB+ |
| 磁盘 | 50GB+（模型缓存） |
| Python | 3.11 |

> 注：多张卡可并行跑多个 job，每个 job 独占 1 张卡。
