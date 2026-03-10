# Preliminary Experiments — Signal Comparison

**目标**: 比较三种 KV cache eviction scoring signal 的效果，验证 question-aware signal 的优势。

## 三种 Scoring Signal

1. **Reconstruction** — 用重建误差评分（kvzip/fastkvzip 使用的方式）
2. **Question-only (QA)** — 仅用 question 的 attention 评分（我们设计的 QASignalPress）
3. **Oracle (Q+A)** — 用完整 question+answer 的 attention 评分（理论上界）

## 数据位置

### Cross-eval 结果

| 位置 | 内容 | 数量 |
|------|------|------|
| `cross_eval/cross_eval_ruler4096*.json` | RULER 4096 汇总 + 13 个 per-task | ✅ 14 files |
| `cross_eval/cross_eval_ruler8192*.json` | RULER 8192 汇总 + 13 个 per-task | ✅ 14 files |
| `cross_eval/cross_eval_ruler16384*.json` | RULER 16384 汇总 + 13 个 per-task | ✅ 14 files |
| `cross_eval/cross_eval_scbench_*.json` | SCBench 11 tasks | ✅ 11 files |
| `cross_eval/cross_eval_{2wikimqa,...}.json` | LongBench v1 QA subsets | ✅ 7 files |
| `cross_eval/three_signals.json` | 三信号细粒度对比 | ✅ |
| `cross_eval/qa_signal_*.json` | QA signal 专项 | ✅ 3 files |
| `cross_eval/ruler*_merged.json` | RULER 8192/16384 合并结果 | ✅ 2 files |

### 可视化

| 目录 | 内容 | 数量 |
|------|------|------|
| `visualizations/attention/` | Question attention 按 layer/head 可视化 | 34 张 |
| `visualizations/retention/` | Token retention 分布、perlayer heatmap | 7 张 |
| `visualizations/threehop/` | 3-hop attention chain 可视化 | 45 张 |

### 分析文档

- `attention_analysis.md` — NIAH token importance 详细分析（2026-02-22）

### 画图脚本

- `scripts/cross_eval.py` — 运行 cross-eval 实验
- `scripts/three_signals.py` — 运行三信号对比
- `scripts/plot_cross_eval.py` — 画 cross-eval 图
- `scripts/plot_three_signals.py` — 画三信号对比图

## 已画的图（`cross_eval/` 下）

| 文件 | 内容 |
|------|------|
| `cross_eval_ruler4096_main.png` | **核心图**：3 种 signal accuracy vs compression ratio |
| `cross_eval_ruler4096_pertask.png` | 按 13 个 task 分开的对比 |
| `cross_eval_ruler4096_delta.png` | 与 baseline 的差值 |
| `cross_eval_ruler4096_persample.png` | Per-sample 分布 |

## 关键结论（RULER 4096, Qwen3-8B）

| Signal | cr=0.30 | cr=0.50 | cr=0.70 | cr=0.90 | cr=0.95 |
|--------|---------|---------|---------|---------|---------|
| no_press | 92.3% | — | — | — | — |
| oracle (Q+A) | 92.3% | 92.3% | 86.8% | 82.4% | 72.5% |
| reconstruction | 92.3% | 92.3% | 91.2% | 72.5% | 35.2% |
| question_only | 92.3% | 90.1% | 84.6% | 60.4% | 42.9% |

- **cr ≤ 0.70**: reconstruction 最好，几乎无损
- **cr = 0.90**: oracle >> reconstruction >> question_only
- **cr = 0.95**: oracle >> question_only >> reconstruction（reconstruction 崩塌）
- **结论**: reconstruction 低压缩比最强但高压缩比崩塌最快；oracle 全程最稳

## TODO

- [x] 从 AML 同步剩余 60+ cross-eval JSON 到本地（66 files, 24MB）
- [ ] 汇总 RULER 8192/16384 + LongBench + SCBench 的 cross-eval 结果
- [ ] 画多 benchmark 对比的汇总图
