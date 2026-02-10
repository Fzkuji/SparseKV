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
git clone https://github.com/simon-mo/kvpress.git
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

## 9. 运行 Llama Baseline 评测

```bash
cd ~/SparseKV
bash scripts/submit_all.sh llama
squeue -u $(whoami)
```

每次提交 4 个任务，跑完后再执行一次 `bash scripts/submit_all.sh llama`，自动跳过已完成的。

## 10. 查看结果

```bash
# 查看所有结果
find ~/kvpress/evaluation/results/phase1_llama -name "metrics.json" | while read f; do
    echo "=== $(basename "$(dirname "$f")") ==="
    cat "$f" | python -c "
import sys,json
d=json.load(sys.stdin)
vals=[v.get('string_match', v.get('score', 0)) for v in d.values() if isinstance(v, dict)]
if vals: print(f'  Avg: {sum(vals)/len(vals):.2f}')
"
done

# 查看 profiling（时间/显存/吞吐量）
find ~/kvpress/evaluation/results/phase1_llama -name "profiling.json" -exec cat {} \;
```

## 环境要求

| 项目 | 最低要求 |
|------|---------|
| GPU | 2× A100 80GB（或同级） |
| CUDA | 12.1+ |
| 内存 | 80GB+ |
| 磁盘 | 50GB+（模型缓存） |
| Python | 3.11 |
