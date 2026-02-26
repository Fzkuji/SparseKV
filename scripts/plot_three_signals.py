"""
Visualize three_signals.py results: per-sample accuracy curves with coverage bands.

Usage:
    python scripts/plot_three_signals.py [--input results/three_signals.json] [--output figures/]
"""
import json, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="results/three_signals.json")
parser.add_argument("--output", default="figures/three_signals")
args = parser.parse_args()

with open(args.input) as f:
    results = json.load(f)

os.makedirs(args.output, exist_ok=True)

methods = ["snapkv", "qa", "recons"]
method_labels = {"snapkv": "Next-Token (SnapKV)", "qa": "Query-Aware (QA)", "recons": "Reconstruction"}
method_colors = {"snapkv": "#2196F3", "qa": "#FF5722", "recons": "#4CAF50"}

# Get all ratios and sample indices
ratios = sorted(set(r["ratio"] for r in results if r["method"] != "full_kv"))
sample_indices = sorted(set(r["sample_idx"] for r in results))

# Build per-sample accuracy curves
# For binary accuracy, we use a sliding window approach: 
# for each sample, accuracy at ratio r = 1 if correct, 0 if wrong
sample_curves = {m: {} for m in methods}
for r in results:
    if r["method"] in methods:
        key = (r["sample_idx"], r["ratio"])
        sample_curves[r["method"]][key] = 1.0 if r["correct"] else 0.0

# ============================================================
# Figure 1: Per-method subplots with individual sample lines + band
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

for ax_idx, method in enumerate(methods):
    ax = axes[ax_idx]
    
    # Collect per-sample curves
    all_curves = []
    for si in sample_indices:
        curve = [sample_curves[method].get((si, r), np.nan) for r in ratios]
        if not all(np.isnan(c) for c in curve):
            all_curves.append(curve)
            # Plot individual lines (very transparent)
            ax.plot(ratios, curve, color=method_colors[method], alpha=0.05, linewidth=0.5)
    
    all_curves = np.array(all_curves)
    
    # Compute stats
    mean_curve = np.nanmean(all_curves, axis=0)
    p5 = np.nanpercentile(all_curves, 5, axis=0)
    p25 = np.nanpercentile(all_curves, 25, axis=0)
    p75 = np.nanpercentile(all_curves, 75, axis=0)
    p95 = np.nanpercentile(all_curves, 95, axis=0)
    
    # Band
    ax.fill_between(ratios, p5, p95, color=method_colors[method], alpha=0.15, label="5th-95th pct")
    ax.fill_between(ratios, p25, p75, color=method_colors[method], alpha=0.3, label="25th-75th pct")
    ax.plot(ratios, mean_curve, color=method_colors[method], linewidth=2.5, label="Mean")
    
    ax.set_xlabel("Eviction Ratio", fontsize=13)
    ax.set_title(method_labels[method], fontsize=14, fontweight="bold")
    ax.set_xlim(ratios[0], ratios[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

# Figure 4: Overlay of all three methods
ax = axes[3]
for method in methods:
    all_curves = []
    for si in sample_indices:
        curve = [sample_curves[method].get((si, r), np.nan) for r in ratios]
        if not all(np.isnan(c) for c in curve):
            all_curves.append(curve)
    all_curves = np.array(all_curves)
    
    mean_curve = np.nanmean(all_curves, axis=0)
    p25 = np.nanpercentile(all_curves, 25, axis=0)
    p75 = np.nanpercentile(all_curves, 75, axis=0)
    
    ax.fill_between(ratios, p25, p75, color=method_colors[method], alpha=0.15)
    ax.plot(ratios, mean_curve, color=method_colors[method], linewidth=2.5, label=method_labels[method])

ax.set_xlabel("Eviction Ratio", fontsize=13)
ax.set_title("All Methods (Overlay)", fontsize=14, fontweight="bold")
ax.set_xlim(ratios[0], ratios[-1])
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=10, loc="lower left")
ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Accuracy", fontsize=13)
fig.suptitle("Three Importance Signals: Per-Sample Accuracy vs Eviction Ratio", fontsize=16, y=1.02)
plt.tight_layout()
path1 = os.path.join(args.output, "three_signals_overview.png")
fig.savefig(path1, dpi=150, bbox_inches="tight")
print(f"Saved: {path1}")

# ============================================================
# Figure 2: Per-task heatmap
# ============================================================
tasks = sorted(set(r["task"] for r in results))
fig2, axes2 = plt.subplots(1, 3, figsize=(20, max(4, len(tasks) * 0.5 + 1)), sharey=True)

for ax_idx, method in enumerate(methods):
    ax = axes2[ax_idx]
    
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
    
    # Annotate
    for ti in range(len(tasks)):
        for ri in range(len(ratios)):
            val = matrix[ti, ri]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(ri, ti, f"{val:.0%}", ha="center", va="center", fontsize=7, color=color)

fig2.colorbar(im, ax=axes2, label="Accuracy", shrink=0.8)
fig2.suptitle("Per-Task Accuracy: Method × Eviction Ratio", fontsize=14, y=1.02)
plt.tight_layout()
path2 = os.path.join(args.output, "three_signals_per_task.png")
fig2.savefig(path2, dpi=150, bbox_inches="tight")
print(f"Saved: {path2}")

# ============================================================
# Figure 3: Max successful eviction ratio distribution
# ============================================================
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax_idx, method in enumerate(methods):
    ax = axes3[ax_idx]
    
    max_ratios_by_task = defaultdict(list)
    for si in sample_indices:
        task = None
        for r in results:
            if r["sample_idx"] == si and r["method"] == "full_kv":
                task = r["task"]
                break
        
        sample_results = [r for r in results 
                          if r["sample_idx"] == si and r["method"] == method]
        correct_ratios = [r["ratio"] for r in sample_results if r["correct"]]
        max_r = max(correct_ratios) if correct_ratios else 0.0
        if task:
            max_ratios_by_task[task].append(max_r)
    
    # Box plot per task
    task_data = [max_ratios_by_task[t] for t in tasks]
    bp = ax.boxplot(task_data, vert=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(method_colors[method])
        patch.set_alpha(0.5)
    
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
    ax.set_title(method_labels[method], fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

axes3[0].set_ylabel("Max Successful Eviction Ratio", fontsize=11)
fig3.suptitle("Distribution of Max Successful Eviction Ratio by Task", fontsize=14, y=1.02)
plt.tight_layout()
path3 = os.path.join(args.output, "three_signals_max_ratio.png")
fig3.savefig(path3, dpi=150, bbox_inches="tight")
print(f"Saved: {path3}")

print("All plots generated.")
