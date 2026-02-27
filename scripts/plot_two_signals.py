"""
Visualize two_signals.py results: QA vs Reconstruction importance signals.

Usage:
    python scripts/plot_two_signals.py [--input results/two_signals.json] [--output figures/two_signals]
"""
import json, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="results/two_signals.json")
parser.add_argument("--output", default="figures/two_signals")
args = parser.parse_args()

with open(args.input) as f:
    results = json.load(f)

os.makedirs(args.output, exist_ok=True)

methods = ["qa", "recons"]
method_labels = {"qa": "QA Signal", "recons": "Reconstruction Signal"}
method_colors = {"qa": "#FF5722", "recons": "#4CAF50"}

ratios = sorted(set(r["ratio"] for r in results if r["method"] != "full_kv"))
sample_indices = sorted(set(r["sample_idx"] for r in results))
tasks = sorted(set(r["task"] for r in results))

# Build per-sample correctness lookup
lookup = {}
for r in results:
    lookup[(r["sample_idx"], r["method"], r["ratio"])] = r["correct"]

# ============================================================
# Figure 1: Mean accuracy curves with confidence bands
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Subplot 1 & 2: individual methods
for ax_idx, method in enumerate(methods):
    ax = axes[ax_idx]
    
    # Per-sample curves
    all_curves = []
    for si in sample_indices:
        curve = [1.0 if lookup.get((si, method, r), False) else 0.0 for r in ratios]
        all_curves.append(curve)
        ax.plot(ratios, curve, color=method_colors[method], alpha=0.04, linewidth=0.5)
    
    all_curves = np.array(all_curves)
    mean_curve = np.mean(all_curves, axis=0)
    
    # Smoothed mean using rolling window (window=3)
    def smooth(y, window=3):
        if len(y) < window:
            return y
        cumsum = np.cumsum(np.insert(y, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window
    
    # Confidence interval via bootstrap
    n_boot = 1000
    boot_means = np.zeros((n_boot, len(ratios)))
    for b in range(n_boot):
        idx = np.random.choice(len(all_curves), size=len(all_curves), replace=True)
        boot_means[b] = np.mean(all_curves[idx], axis=0)
    ci_lo = np.percentile(boot_means, 2.5, axis=0)
    ci_hi = np.percentile(boot_means, 97.5, axis=0)
    
    ax.fill_between(ratios, ci_lo, ci_hi, color=method_colors[method], alpha=0.2, label="95% CI")
    ax.plot(ratios, mean_curve, color=method_colors[method], linewidth=2.5, marker='o', 
            markersize=4, label=f"Mean (n={len(sample_indices)})")
    
    ax.set_xlabel("Eviction Ratio", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(method_labels[method], fontsize=14, fontweight="bold")
    ax.set_xlim(ratios[0] - 0.02, ratios[-1] + 0.02)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3)

# Subplot 3: overlay
ax = axes[2]
for method in methods:
    all_curves = []
    for si in sample_indices:
        curve = [1.0 if lookup.get((si, method, r), False) else 0.0 for r in ratios]
        all_curves.append(curve)
    all_curves = np.array(all_curves)
    mean_curve = np.mean(all_curves, axis=0)
    
    n_boot = 1000
    boot_means = np.zeros((n_boot, len(ratios)))
    for b in range(n_boot):
        idx = np.random.choice(len(all_curves), size=len(all_curves), replace=True)
        boot_means[b] = np.mean(all_curves[idx], axis=0)
    ci_lo = np.percentile(boot_means, 2.5, axis=0)
    ci_hi = np.percentile(boot_means, 97.5, axis=0)
    
    ax.fill_between(ratios, ci_lo, ci_hi, color=method_colors[method], alpha=0.15)
    ax.plot(ratios, mean_curve, color=method_colors[method], linewidth=2.5, marker='o',
            markersize=4, label=method_labels[method])

# Add full_kv baseline
fk = [r for r in results if r["method"] == "full_kv"]
fk_acc = sum(r["correct"] for r in fk) / len(fk)
ax.axhline(y=fk_acc, color='gray', linestyle='--', linewidth=1.5, label=f"Full KV ({fk_acc:.1%})")

ax.set_xlabel("Eviction Ratio", fontsize=13)
ax.set_ylabel("Accuracy", fontsize=13)
ax.set_title("QA vs Reconstruction (Overlay)", fontsize=14, fontweight="bold")
ax.set_xlim(ratios[0] - 0.02, ratios[-1] + 0.02)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=10, loc="lower left")
ax.grid(True, alpha=0.3)

fig.suptitle("KV Importance Signal Comparison: Accuracy vs Eviction Ratio", fontsize=16, y=1.02)
plt.tight_layout()
path1 = os.path.join(args.output, "accuracy_curves.png")
fig.savefig(path1, dpi=150, bbox_inches="tight")
print(f"Saved: {path1}")

# ============================================================
# Figure 2: Max successful eviction ratio distribution
# ============================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# 2a: Histogram / KDE
ax = axes2[0]
for method in methods:
    max_crs = []
    for si in sample_indices:
        correct_ratios = [r for r in ratios if lookup.get((si, method, r), False)]
        max_cr = max(correct_ratios) if correct_ratios else 0.0
        max_crs.append(max_cr)
    
    ax.hist(max_crs, bins=np.arange(0, 1.05, 0.05), alpha=0.5, 
            color=method_colors[method], label=method_labels[method], edgecolor='white')
    mean_cr = np.mean(max_crs)
    ax.axvline(x=mean_cr, color=method_colors[method], linestyle='--', linewidth=2,
               label=f"Mean: {mean_cr:.2f}")

ax.set_xlabel("Max Successful Eviction Ratio", fontsize=13)
ax.set_ylabel("Count", fontsize=13)
ax.set_title("Distribution of Max Successful CR", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 2b: Per-task comparison (paired bar chart)
ax = axes2[1]
x = np.arange(len(tasks))
width = 0.35

for mi, method in enumerate(methods):
    task_means = []
    for task in tasks:
        task_samples_list = [si for si in sample_indices 
                        if any(r["sample_idx"] == si and r["task"] == task for r in results)]
        max_crs = []
        for si in task_samples_list:
            correct_ratios = [r for r in ratios if lookup.get((si, method, r), False)]
            max_cr = max(correct_ratios) if correct_ratios else 0.0
            max_crs.append(max_cr)
        task_means.append(np.mean(max_crs) if max_crs else 0)
    
    offset = (mi - 0.5) * width
    ax.bar(x + offset, task_means, width, color=method_colors[method], 
           label=method_labels[method], alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Mean Max Successful CR", fontsize=12)
ax.set_title("Per-Task Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.05)

plt.tight_layout()
path2 = os.path.join(args.output, "max_cr_distribution.png")
fig2.savefig(path2, dpi=150, bbox_inches="tight")
print(f"Saved: {path2}")

# ============================================================
# Figure 3: Per-task heatmap (both methods side by side)
# ============================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(18, max(4, len(tasks) * 0.5 + 1)), sharey=True)

for ax_idx, method in enumerate(methods):
    ax = axes3[ax_idx]
    
    matrix = np.zeros((len(tasks), len(ratios)))
    for ti, task in enumerate(tasks):
        for ri, ratio in enumerate(ratios):
            task_results = [r for r in results 
                           if r["task"] == task and r["method"] == method and r["ratio"] == ratio]
            if task_results:
                matrix[ti, ri] = sum(r["correct"] for r in task_results) / len(task_results)
    
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"{r:.2f}" for r in ratios], rotation=45, fontsize=8)
    ax.set_xlabel("Eviction Ratio")
    ax.set_title(method_labels[method], fontsize=12, fontweight="bold")
    
    if ax_idx == 0:
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks, fontsize=9)
    
    for ti in range(len(tasks)):
        for ri in range(len(ratios)):
            val = matrix[ti, ri]
            color = "white" if val < 0.3 or val > 0.8 else "black"
            ax.text(ri, ti, f"{val:.0%}", ha="center", va="center", fontsize=7, color=color)

fig3.colorbar(im, ax=axes3, label="Accuracy", shrink=0.8)
fig3.suptitle("Per-Task Accuracy Heatmap", fontsize=14, y=1.02)
plt.tight_layout()
path3 = os.path.join(args.output, "per_task_heatmap.png")
fig3.savefig(path3, dpi=150, bbox_inches="tight")
print(f"Saved: {path3}")

# ============================================================
# Figure 4: Per-sample scatter — QA max CR vs Recons max CR
# ============================================================
fig4, ax4 = plt.subplots(figsize=(8, 8))

qa_max_crs = []
recons_max_crs = []
sample_tasks = []

for si in sample_indices:
    qa_correct = [r for r in ratios if lookup.get((si, "qa", r), False)]
    recons_correct = [r for r in ratios if lookup.get((si, "recons", r), False)]
    qa_max = max(qa_correct) if qa_correct else 0.0
    recons_max = max(recons_correct) if recons_correct else 0.0
    qa_max_crs.append(qa_max)
    recons_max_crs.append(recons_max)
    
    task = None
    for r in results:
        if r["sample_idx"] == si and r["method"] == "full_kv":
            task = r["task"]
            break
    sample_tasks.append(task)

qa_max_crs = np.array(qa_max_crs)
recons_max_crs = np.array(recons_max_crs)

# Color by task
unique_tasks = sorted(set(sample_tasks))
cmap = plt.cm.get_cmap('tab20', len(unique_tasks))
task_to_color = {t: cmap(i) for i, t in enumerate(unique_tasks)}

for t in unique_tasks:
    mask = np.array([st == t for st in sample_tasks])
    ax4.scatter(recons_max_crs[mask], qa_max_crs[mask], 
               c=[task_to_color[t]], label=t, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)

# Diagonal line
ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
ax4.set_xlabel("Reconstruction: Max Successful CR", fontsize=13)
ax4.set_ylabel("QA: Max Successful CR", fontsize=13)
ax4.set_title("Per-Sample: QA vs Reconstruction Max CR", fontsize=14, fontweight="bold")
ax4.set_xlim(-0.05, 1.05)
ax4.set_ylim(-0.05, 1.05)
ax4.legend(fontsize=7, loc="upper left", ncol=2)
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')

# Annotate quadrants
n_total = len(sample_indices)
qa_better = np.sum(qa_max_crs > recons_max_crs)
recons_better = np.sum(recons_max_crs > qa_max_crs)
tied = np.sum(qa_max_crs == recons_max_crs)
ax4.text(0.95, 0.05, f"Recons better: {recons_better}/{n_total}", 
         transform=ax4.transAxes, ha='right', fontsize=10, color=method_colors["recons"])
ax4.text(0.05, 0.95, f"QA better: {qa_better}/{n_total}", 
         transform=ax4.transAxes, ha='left', fontsize=10, color=method_colors["qa"])
ax4.text(0.5, 0.02, f"Tied: {tied}/{n_total}", 
         transform=ax4.transAxes, ha='center', fontsize=10, color='gray')

plt.tight_layout()
path4 = os.path.join(args.output, "qa_vs_recons_scatter.png")
fig4.savefig(path4, dpi=150, bbox_inches="tight")
print(f"Saved: {path4}")

print("\nAll plots generated.")
