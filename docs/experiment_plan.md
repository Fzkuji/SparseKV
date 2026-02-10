# SparseKV: Complete Experiment Plan

## 核心思想

通过训练，让模型的 attention 集中到特定的 **anchor tokens** 上。推理时只保留 anchor tokens 的 KV cache，实现零成本 eviction。

```
Phase 0: 分析 → 发现 anchor pattern
Phase 1: Baseline → 现有 eviction 方法的性能
Phase 2: 训练 → 让 attention 更集中到 anchor tokens
Phase 3: 评测 → 训练后的模型 + eviction 方法
Phase 4: 消融 → 验证每个设计选择
```

---

## Phase 0: Attention Pattern Analysis

### 目标
发现预训练模型中，哪些 token / 位置天然接收更多 attention。

### 实验

**0.1 Token Type 分析**
- 模型：Qwen3-8B, Llama-3.1-8B-Instruct
- 数据：FineWeb-Edu, 50 samples, max_len=2048
- 分析维度：
  - 每种 token type 的 per-token 平均 attention（被关注程度）
  - 每种 token type 占总 attention 的比例
  - Token types: special, punctuation, stopword, content, number, whitespace

**0.2 Position 分析**
- Sink effect（前 4 个 token）
- Recency bias（最后 N 个 token）
- 中间 token 的 attention 分布

**0.3 Per-Head 分析**
- 每个 head 的 attention concentration（多少 % 的 token 承载了 80% 的 attention）
- 不同 head 是否有不同的 pattern（有些 head 关注标点，有些关注位置）
- 哪些层/head 最集中，哪些最分散

### 预期结果
- Punctuation 和 special tokens 的 per-token attention 显著高于 content tokens
- Sink tokens 和 recent tokens 接收更多 attention
- 不同 head 有不同的 anchor pattern

### 脚本
```bash
sbatch scripts/slurm_analyze.sh  # Qwen3-8B
# TODO: 加 Llama 版本
```

### 输出
- `analysis/attention_analysis_Qwen--Qwen3-8B.json`
- 可视化图表（per-layer heatmap, token type bar chart）

---

## Phase 1: Baseline Evaluation (已在进行)

### 目标
获取未经训练的模型在各种 eviction 方法下的性能，作为对比基线。

### 实验矩阵

| 模型 | Benchmark | Press 方法 | 压缩率 |
|------|-----------|-----------|:------:|
| Qwen3-8B | RULER 4k, 16k | no_press, snapkv, streaming_llm, critical_snapkv, kvzip | 0, 0.3, 0.5, 0.7 |
| Llama-3.1-8B | LongBench, AIME25 | 同上 | 同上 |

### 状态
- [x] Qwen3-8B 脚本提交
- [ ] Llama-3.1-8B 待 HuggingFace 权限
- 每模型约 52 个任务，4 个一批提交

---

## Phase 2: Anchor-Aware Training

### 前置条件
Phase 0 分析结果，确定 anchor token 的定义。

### Step 2.1: 定义 Anchor Tokens

基于 Phase 0 的分析结果，定义 anchor token 规则。可能的规则：

**规则 A: Token Type Based**
```python
def is_anchor(token_id, position, seq_len):
    return (
        token_type[token_id] == 'punctuation' or
        token_type[token_id] == 'special' or
        position < 4 or                    # sink tokens
        position >= seq_len - 64           # recent window
    )
```

**规则 B: Attention Score Based (动态)**
```python
def is_anchor(attention_weights, keep_ratio):
    # 基于实际 attention score 动态选择
    importance = attention_weights.sum(dim=query_dim)
    topk = importance.topk(int(seq_len * keep_ratio))
    return topk.indices
```

**规则 C: Hybrid**
```python
def is_anchor(token_id, position, attention_weights, keep_ratio):
    # 固定保留 sink + recent + punctuation
    # 剩余 budget 按 attention score 动态选择
    fixed_anchors = sink | recent | punctuation
    remaining_budget = int(seq_len * keep_ratio) - len(fixed_anchors)
    dynamic_anchors = attention_score_topk(remaining_budget, exclude=fixed_anchors)
    return fixed_anchors | dynamic_anchors
```

**选择取决于 Phase 0 结果。** 如果分析发现 token type 效应很强 → 用规则 A/C。如果位置效应更强 → 用规则 B。

### Step 2.2: Training Objective

#### Loss 设计

```
L_total = L_LM + λ · L_concentration
```

**L_LM**: 标准语言模型 cross-entropy loss（保持模型能力）

**L_concentration**: 引导 attention 集中到 anchor tokens

设计选项：

**选项 1: Anchor Attention Maximization**
```python
# 最大化落在 anchor 上的 attention 总量
anchor_mask = is_anchor(input_ids)  # (B, L)
for layer_attn in attention_weights:  # (B, H, L, L)
    attn_on_anchor = layer_attn[:, :, :, anchor_mask].sum(dim=-1)  # 每个 query 分配给 anchor 的权重
    L_concentration += -attn_on_anchor.mean()  # 最大化
```

**选项 2: Eviction Invariance (EIT)**
```python
# 最小化 eviction 前后输出差异
full_out = attention(Q, K, V)
evict_out = attention(Q, K[anchor], V[anchor])  # 只用 anchor 的 KV
L_concentration = MSE(full_out, evict_out)
```

**选项 3: KL Divergence on Logits**
```python
# Output-level: eviction 前后 logits 一致性
full_logits = model(input_ids)  # flash attention
evict_logits = model(input_ids, kv_mask=anchor_mask)  # 只用 anchor KV
L_concentration = KL(full_logits.detach(), evict_logits)
```

**初步选择：选项 2 (EIT) 作为主方法，选项 1 作为辅助 loss。**

理由：
- 选项 1 直接约束 attention 分布，但不保证 eviction 后输出正确
- 选项 2 直接优化 eviction 后的输出质量，更接近最终目标
- 选项 3 需要两次完整 forward，开销更大

#### Attention 计算方案

训练时需要 attention weights（用于 eviction 决策 + loss 计算）:

- **序列长度 ≤ 4096**: 使用 SDPA (eager)，attention matrix 可以放入显存
  - 4096² × 32 heads × 2 bytes = 1 GB/layer，可行
- **序列长度 > 4096**: 使用 observation window
  - 只计算最后 W=128 个 query 的 attention score
  - Eviction 决策和 loss 都在这 W 个位置上
  - 这和推理时 SnapKV 的做法一致

### Step 2.3: Training Configuration

| 参数 | 值 | 说明 |
|------|-----|------|
| 基础模型 | Qwen3-8B, Llama-3.1-8B-Instruct | 和 Phase 1 对齐 |
| 训练方式 | LoRA (r=64, α=128) | 高效，<1% 参数 |
| Target modules | q_proj, k_proj, v_proj, o_proj | Attention 相关层 |
| 训练数据 | FineWeb-Edu (sample-10BT) | 通用文本 |
| 序列长度 | 4096 | 先用短序列验证 |
| Batch size | 1 | 显存限制 |
| Gradient accumulation | 8 | 有效 batch = 8 |
| Learning rate | 2e-5 | LoRA 常用 |
| Training steps | 1000-5000 | 需要实验确定 |
| λ (concentration weight) | 0.1 ~ 10.0 | 需要 sweep |
| Attention implementation | SDPA (eager) | 需要 attention weights |
| GPU | 2× A100 80GB | 服务器 A |

### Step 2.4: Compression Scheduling

训练过程中的压缩率（eviction 的比例）：

**Curriculum Strategy:**
```
Step 0-200:     keep_ratio = 0.9 (轻微压缩，模型适应)
Step 200-500:   keep_ratio = 0.7 (中等压缩)
Step 500-1000:  keep_ratio = 0.5 (强压缩)
Step 1000+:     keep_ratio = 0.3 (极限压缩)
```

**Adaptive Strategy (备选):**
- 监控 validation perplexity
- PPL 不涨 → 加大压缩
- PPL 涨 → 保持当前压缩率

### Step 2.5: 训练输出

- LoRA adapter checkpoint
- 合并后的完整模型
- 训练 log (loss curves, compression ratio, attention concentration)
- Attention 分布变化的可视化（训练前 vs 训练后）

---

## Phase 3: Post-Training Evaluation

### 目标
验证训练后的模型在 eviction 下性能更好。

### 实验矩阵

和 Phase 1 完全相同的矩阵，但用训练后的模型：

| 模型 | Benchmark | Press | 压缩率 |
|------|-----------|-------|:------:|
| Qwen3-8B + SparseKV | RULER 4k, 16k, LongBench, AIME25 | no_press, snapkv, streaming_llm, critical_snapkv, kvzip | 0, 0.3, 0.5, 0.7 |
| Llama-3.1-8B + SparseKV | 同上 | 同上 | 同上 |

### 核心指标

**主指标：Performance Retention Rate**
```
Retention(press, cr) = Score(model+SparseKV, press, cr) / Score(model+SparseKV, no_press)
```
对比：
```
Baseline_Retention(press, cr) = Score(model, press, cr) / Score(model, no_press)
```

**我们的目标：SparseKV Retention > Baseline Retention，在所有 (press, cr) 组合上。**

### 额外实验

**3.1 模型能力是否下降**
- no_press 下的性能：SparseKV 训练后 vs 训练前
- 如果 no_press 性能不降 → 训练没有损害模型能力
- 如果略降 (<1%) → 可接受的 trade-off

**3.2 Anchor Token Eviction (我们的推理方法)**
- 不用现有 press，直接按 anchor 规则 eviction
- 性能应该接近 best press method，但零推理成本

**3.3 Cross-Transfer**
- 用 SnapKV 规则训练 → 用 StreamingLLM 测试
- 验证训练的 anchor pattern 是否对所有 eviction 方法都有效

---

## Phase 4: Ablation Studies

### 4.1 Loss 选择
| Loss | 设置 |
|------|------|
| L_LM only | 无 concentration loss（= 标准 continual pretraining） |
| L_LM + L_anchor_max | 只有 attention 集中 loss |
| L_LM + L_eit | 只有 eviction invariance loss |
| L_LM + L_anchor_max + L_eit | 两个 loss 都有 |

### 4.2 λ 敏感性
- λ ∈ {0.01, 0.1, 1.0, 10.0}
- 观察：attention 集中程度 vs 模型性能 的 trade-off

### 4.3 训练步数
- 500, 1000, 2000, 5000 steps
- 找到收益饱和点

### 4.4 Anchor 定义
- 只保留 punctuation
- 只保留 sink + recent
- Punctuation + sink + recent
- Dynamic attention-based（AdaKV 风格）

### 4.5 压缩率
- 训练时和测试时用不同压缩率
- 训练时 keep_ratio=0.5 → 测试时 keep_ratio=0.3?
- 验证泛化性

---

## 时间线

| 阶段 | 时间 | 前置条件 | 产出 |
|------|:----:|---------|------|
| Phase 0 分析 | 1-2 天 | 无 | anchor 定义 |
| Phase 1 Baseline | 1 周 | 无（已在跑） | baseline 结果表 |
| Phase 2 训练 | 1-2 周 | Phase 0 结果 | 训练后模型 |
| Phase 3 评测 | 1 周 | Phase 2 模型 | 对比结果 |
| Phase 4 消融 | 1-2 周 | Phase 2/3 | 消融表 |
| 写论文 | 2 周 | 以上全部 | Paper |

**总计：6-8 周**

---

## 关键决策点

在 Phase 0 结果出来后，需要确定：

1. **Anchor 定义**：用 token type 还是 attention score 还是混合？
2. **Loss 选择**：EIT (layer-wise MSE) 还是 output-level KL？
3. **Attention 计算**：全部 SDPA 还是 observation window？
4. **是否需要修改模型结构**：比如加 special anchor tokens？还是只用自然存在的 token？

这些决策将基于 Phase 0 的实证结果，不提前锁定。

---

## 风险与备选

| 风险 | 可能性 | 备选方案 |
|------|:------:|---------|
| Attention 天然不够集中，训练效果差 | 中 | 改为 EIT (不依赖 anchor，直接优化 eviction invariance) |
| 训练后 no_press 性能下降严重 | 低 | 降低 λ，或加 KL 锚定 (like ASFT) |
| SDPA 太慢，训练时间过长 | 低 | 减少训练步数，或用 observation window |
| 服务器排队等太久 | 高 | 分批提交，利用夜间时段 |
