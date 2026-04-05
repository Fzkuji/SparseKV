"""
Visualize cross-eval results: Reconstruction vs Question-only vs Oracle scoring.
Generates:
1. Main comparison line chart (accuracy vs compression ratio)
2. Per-task breakdown heatmap
3. Per-sample delta analysis
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.dpi'] = 150

data = json.load(open("results/cross_eval_ruler4096.json"))

RATIOS = [0.30, 0.50, 0.70, 0.90, 0.95]
METHODS = {
    "reconstruction": ("Reconstruction (KVzip)", "#2196F3", "s"),
    "question_only": ("Question-only", "#FF9800", "^"),
    "qa_oracle": ("Oracle (Q+A)", "#4CAF50", "o"),
}

# ============================================================
# Figure 1: Main comparison line chart
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Full KV baseline
fk = [r for r in data if r["scoring"] == "none"]
fk_acc = sum(r["correct"] for r in fk) / len(fk) * 100

ax.axhline(y=fk_acc, color='gray', linestyle='--', alpha=0.7, label=f'Full KV ({fk_acc:.1f}%)')

for method, (label, color, marker) in METHODS.items():
    accs = []
    for ratio in RATIOS:
        rs = [r for r in data if r["scoring"] == method and r["ratio"] == ratio]
        acc = sum(r["correct"] for r in rs) / len(rs) * 100 if rs else 0
        accs.append(acc)
    ax.plot(RATIOS, accs, marker=marker, color=color, linewidth=2, markersize=8, label=label)

ax.set_xlabel("Compression Ratio", fontsize=13)
ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("KV Cache Eviction: Scoring Method Comparison\n(RULER 4096, Qwen3-8B, 91 samples)", fontsize=14)
ax.legend(loc='lower left', fontsize=11)
ax.set_xticks(RATIOS)
ax.set_ylim(20, 100)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/cross_eval_ruler4096_main.png", bbox_inches='tight')
print("Saved: results/cross_eval_ruler4096_main.png")

# ============================================================
# Figure 2: Per-task breakdown at CR=0.9 and CR=0.95
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax_idx, ratio in enumerate([0.90, 0.95]):
    ax = axes[ax_idx]
    
    # Collect per-task accuracies
    tasks = sorted(set(r["task"] for r in data if r["scoring"] == "none"))
    # Exclude tasks with 0% fullkv
    tasks = [t for t in tasks if sum(r["correct"] for r in data if r["scoring"] == "none" and r["task"] == t) > 0]
    
    x = np.arange(len(tasks))
    width = 0.25
    
    for i, (method, (label, color, marker)) in enumerate(METHODS.items()):
        task_accs = []
        for task in tasks:
            rs = [r for r in data if r["scoring"] == method and r["ratio"] == ratio and r["task"] == task]
            acc = sum(r["correct"] for r in rs) / len(rs) * 100 if rs else 0
            task_accs.append(acc)
        ax.bar(x + i * width, task_accs, width, label=label, color=color, alpha=0.8)
    
    # Full KV reference
    fk_accs = []
    for task in tasks:
        rs = [r for r in data if r["scoring"] == "none" and r["task"] == task]
        acc = sum(r["correct"] for r in rs) / len(rs) * 100 if rs else 0
        fk_accs.append(acc)
    
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Per-Task Accuracy at CR={ratio}", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("results/cross_eval_ruler4096_pertask.png", bbox_inches='tight')
print("Saved: results/cross_eval_ruler4096_pertask.png")

# ============================================================
# Figure 3: Per-sample accuracy curves (each sample = one line)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sample_indices = sorted(set(r["sample_idx"] for r in data if r["scoring"] == "none"))

for ax_idx, (method, (label, color, marker)) in enumerate(METHODS.items()):
    ax = axes[ax_idx]
    
    # Per-sample: at each ratio, correct or not (1 or 0)
    for si in sample_indices:
        accs = []
        for ratio in RATIOS:
            rs = [r for r in data if r["scoring"] == method and r["ratio"] == ratio and r["sample_idx"] == si]
            acc = rs[0]["correct"] if rs else 0
            accs.append(acc)
        # Add jitter for visibility
        jittered = [a + np.random.uniform(-0.02, 0.02) for a in accs]
        ax.plot(RATIOS, jittered, alpha=0.15, color=color, linewidth=0.8)
    
    # Aggregate line
    agg_accs = []
    for ratio in RATIOS:
        rs = [r for r in data if r["scoring"] == method and r["ratio"] == ratio]
        acc = sum(r["correct"] for r in rs) / len(rs) * 100 / 100 if rs else 0
        agg_accs.append(acc)
    ax.plot(RATIOS, agg_accs, color='black', linewidth=3, marker='o', markersize=6, label='Mean')
    
    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("Correct (0/1)", fontsize=12)
    ax.set_title(label, fontsize=13)
    ax.set_xticks(RATIOS)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle("Per-Sample Accuracy Curves (RULER 4096, Qwen3-8B)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("results/cross_eval_ruler4096_persample.png", bbox_inches='tight')
print("Saved: results/cross_eval_ruler4096_persample.png")

# ============================================================
# Figure 4: Delta analysis — Oracle vs Recon, Q-only vs Recon
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for method, (label, color, marker) in [
    ("qa_oracle", ("Oracle - Recon", "#4CAF50", "o")),
    ("question_only", ("Q-only - Recon", "#FF9800", "^")),
]:
    deltas = []
    for ratio in RATIOS:
        recon = [r for r in data if r["scoring"] == "reconstruction" and r["ratio"] == ratio]
        other = [r for r in data if r["scoring"] == method and r["ratio"] == ratio]
        r_acc = sum(r["correct"] for r in recon) / len(recon) * 100 if recon else 0
        o_acc = sum(r["correct"] for r in other) / len(other) * 100 if other else 0
        deltas.append(o_acc - r_acc)
    ax.plot(RATIOS, deltas, marker=marker, color=color, linewidth=2, markersize=8, label=label)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("Compression Ratio", fontsize=13)
ax.set_ylabel("Accuracy Δ (pp)", fontsize=13)
ax.set_title("Accuracy Gain over Reconstruction Scoring\n(RULER 4096, Qwen3-8B)", fontsize=14)
ax.legend(fontsize=11)
ax.set_xticks(RATIOS)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/cross_eval_ruler4096_delta.png", bbox_inches='tight')
print("Saved: results/cross_eval_ruler4096_delta.png")

print("\nAll figures saved!")
