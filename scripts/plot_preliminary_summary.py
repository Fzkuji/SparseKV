#!/usr/bin/env python3
"""Plot comprehensive preliminary experiment summary.

Generates:
1. Multi-panel accuracy vs compression ratio (RULER 4096/8192/16384 + LongBench avg)
2. Per-benchmark bar chart at key compression ratios
3. Summary table printed to stdout

Usage:
    python scripts/plot_preliminary_summary.py [--data_dir results/preliminary/cross_eval] [--output_dir results/preliminary/figures]
"""

import argparse
import json
import os
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


SIGNAL_COLORS = {
    'none': '#333333',
    'reconstruction': '#2196F3',
    'question_only': '#FF9800',
    'qa_oracle': '#4CAF50',
}
SIGNAL_LABELS = {
    'none': 'No Press (baseline)',
    'reconstruction': 'Reconstruction',
    'question_only': 'Question-only',
    'qa_oracle': 'Oracle (Q+A)',
}
SIGNAL_MARKERS = {
    'none': 's',
    'reconstruction': 'o',
    'question_only': '^',
    'qa_oracle': 'D',
}


def load_cross_eval(path):
    """Load a cross-eval JSON and return list of records."""
    with open(path) as f:
        return json.load(f)


def compute_accuracy(records, scoring, ratio):
    """Compute accuracy for a given scoring+ratio."""
    subset = [r for r in records if r.get('scoring') == scoring and r.get('ratio') == ratio]
    if not subset:
        return None, 0
    correct = sum(1 for r in subset if r.get('correct', False))
    return correct / len(subset) * 100, len(subset)


def compute_score(records, scoring, ratio):
    """Compute average score for records with 'score' field."""
    subset = [r for r in records if r.get('scoring') == scoring and r.get('ratio') == ratio]
    if not subset:
        return None, 0
    scores = [r['score'] for r in subset if 'score' in r and r['score'] is not None]
    if not scores:
        return None, 0
    return sum(scores) / len(scores) * 100, len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='results/preliminary/cross_eval')
    parser.add_argument('--output_dir', default='results/preliminary/figures')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load RULER data - prefer merging per-task files over aggregated file
    ruler_data = {}
    for seq_len in [4096, 8192, 16384]:
        # Try per-task files first (more complete)
        per_task_files = sorted(glob.glob(str(data_dir / f'cross_eval_ruler{seq_len}_*.json')))
        if per_task_files:
            all_records = []
            for f in per_task_files:
                all_records.extend(load_cross_eval(f))
            ruler_data[f'RULER {seq_len}'] = all_records
            tasks = set(r.get('task') for r in all_records)
            print(f"RULER {seq_len}: loaded {len(all_records)} records from {len(per_task_files)} per-task files, {len(tasks)} tasks")
        else:
            # Fallback to aggregated file
            path = data_dir / f'cross_eval_ruler{seq_len}.json'
            if path.exists():
                ruler_data[f'RULER {seq_len}'] = load_cross_eval(path)

    # Load LongBench QA files
    longbench_names = ['2wikimqa', 'hotpotqa', 'multifieldqa_en', 'musique', 'narrativeqa', 'qasper', 'triviaqa']
    longbench_data = {}
    for name in longbench_names:
        path = data_dir / f'cross_eval_{name}.json'
        if path.exists():
            longbench_data[name] = load_cross_eval(path)

    scorings = ['none', 'reconstruction', 'question_only', 'qa_oracle']
    ratios = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]

    # ========== Figure 1: Multi-panel accuracy curves ==========
    panels = list(ruler_data.keys())
    if longbench_data:
        panels.append('LongBench QA (avg)')

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), sharey=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, panel_name in zip(axes, panels):
        if panel_name.startswith('RULER'):
            records = ruler_data[panel_name]
            for scoring in scorings:
                accs = []
                valid_ratios = []
                for ratio in ratios:
                    acc, n = compute_accuracy(records, scoring, ratio)
                    if acc is not None and (scoring != 'none' or ratio == 0.0):
                        accs.append(acc)
                        valid_ratios.append(ratio)
                if accs:
                    if scoring == 'none':
                        ax.axhline(y=accs[0], color=SIGNAL_COLORS[scoring], linestyle='--',
                                   alpha=0.5, label=SIGNAL_LABELS[scoring])
                    else:
                        ax.plot(valid_ratios, accs, color=SIGNAL_COLORS[scoring],
                                marker=SIGNAL_MARKERS[scoring], markersize=6, linewidth=2,
                                label=SIGNAL_LABELS[scoring])
        elif panel_name == 'LongBench QA (avg)':
            for scoring in scorings:
                avg_accs = []
                valid_ratios = []
                for ratio in ratios:
                    task_accs = []
                    for name, records in longbench_data.items():
                        acc, n = compute_accuracy(records, scoring, ratio)
                        if acc is not None:
                            task_accs.append(acc)
                    if task_accs and (scoring != 'none' or ratio == 0.0):
                        avg_accs.append(sum(task_accs) / len(task_accs))
                        valid_ratios.append(ratio)
                if avg_accs:
                    if scoring == 'none':
                        ax.axhline(y=avg_accs[0], color=SIGNAL_COLORS[scoring], linestyle='--',
                                   alpha=0.5, label=SIGNAL_LABELS[scoring])
                    else:
                        ax.plot(valid_ratios, avg_accs, color=SIGNAL_COLORS[scoring],
                                marker=SIGNAL_MARKERS[scoring], markersize=6, linewidth=2,
                                label=SIGNAL_LABELS[scoring])

        ax.set_title(panel_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Compression Ratio', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_xlim(-0.05, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')

    fig.suptitle('Cross-Eval: Three Scoring Signals Comparison (Qwen3-8B)', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'signal_comparison_all_benchmarks.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'signal_comparison_all_benchmarks.png'}")

    # ========== Figure 2: Per-LongBench-task breakdown ==========
    if longbench_data:
        key_ratios = [0.3, 0.5, 0.7, 0.9]
        signal_list = ['reconstruction', 'question_only', 'qa_oracle']

        fig2, axes2 = plt.subplots(1, len(key_ratios), figsize=(5 * len(key_ratios), 5), sharey=True)

        for ax, ratio in zip(axes2, key_ratios):
            task_names = sorted(longbench_data.keys())
            x = np.arange(len(task_names))
            width = 0.25

            for i, scoring in enumerate(signal_list):
                accs = []
                for name in task_names:
                    acc, _ = compute_accuracy(longbench_data[name], scoring, ratio)
                    accs.append(acc if acc is not None else 0)
                bars = ax.bar(x + i * width - width, accs, width,
                              color=SIGNAL_COLORS[scoring], label=SIGNAL_LABELS[scoring], alpha=0.85)

            # Add baseline
            baselines = []
            for name in task_names:
                acc, _ = compute_accuracy(longbench_data[name], 'none', 0.0)
                baselines.append(acc if acc is not None else 0)
            ax.plot(x, baselines, 'ks--', markersize=4, alpha=0.5, label='Baseline')

            ax.set_title(f'CR = {ratio}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([n[:8] for n in task_names], rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Accuracy (%)', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            if ratio == key_ratios[0]:
                ax.legend(fontsize=7, loc='upper right')

        fig2.suptitle('LongBench QA: Per-Task Signal Comparison', fontsize=14, fontweight='bold', y=1.02)
        fig2.tight_layout()
        fig2.savefig(output_dir / 'longbench_pertask_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'longbench_pertask_comparison.png'}")

    # ========== Figure 3: Delta from baseline ==========
    fig3, axes3 = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), sharey=False)
    if len(panels) == 1:
        axes3 = [axes3]

    for ax, panel_name in zip(axes3, panels):
        if panel_name.startswith('RULER'):
            records = ruler_data[panel_name]
            baseline_acc, _ = compute_accuracy(records, 'none', 0.0)
            for scoring in ['reconstruction', 'question_only', 'qa_oracle']:
                deltas = []
                valid_ratios = []
                for ratio in ratios:
                    if ratio == 0.0:
                        continue
                    acc, n = compute_accuracy(records, scoring, ratio)
                    if acc is not None and baseline_acc is not None:
                        deltas.append(acc - baseline_acc)
                        valid_ratios.append(ratio)
                if deltas:
                    ax.plot(valid_ratios, deltas, color=SIGNAL_COLORS[scoring],
                            marker=SIGNAL_MARKERS[scoring], markersize=6, linewidth=2,
                            label=SIGNAL_LABELS[scoring])
        elif panel_name == 'LongBench QA (avg)':
            # Compute avg baseline
            task_baselines = []
            for name, records in longbench_data.items():
                acc, _ = compute_accuracy(records, 'none', 0.0)
                if acc is not None:
                    task_baselines.append(acc)
            avg_baseline = sum(task_baselines) / len(task_baselines) if task_baselines else 0

            for scoring in ['reconstruction', 'question_only', 'qa_oracle']:
                deltas = []
                valid_ratios = []
                for ratio in ratios:
                    if ratio == 0.0:
                        continue
                    task_accs = []
                    for name, records in longbench_data.items():
                        acc, _ = compute_accuracy(records, scoring, ratio)
                        if acc is not None:
                            task_accs.append(acc)
                    if task_accs:
                        avg = sum(task_accs) / len(task_accs)
                        deltas.append(avg - avg_baseline)
                        valid_ratios.append(ratio)
                if deltas:
                    ax.plot(valid_ratios, deltas, color=SIGNAL_COLORS[scoring],
                            marker=SIGNAL_MARKERS[scoring], markersize=6, linewidth=2,
                            label=SIGNAL_LABELS[scoring])

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(panel_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Compression Ratio', fontsize=11)
        ax.set_ylabel('Δ Accuracy (% pts)', fontsize=11)
        ax.set_xlim(0.2, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')

    fig3.suptitle('Accuracy Drop from Baseline by Signal', fontsize=14, fontweight='bold', y=1.02)
    fig3.tight_layout()
    fig3.savefig(output_dir / 'signal_delta_all_benchmarks.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'signal_delta_all_benchmarks.png'}")

    # ========== Print Summary Table ==========
    print("\n" + "=" * 100)
    print("  PRELIMINARY RESULTS SUMMARY — Cross-Eval Signal Comparison (Qwen3-8B)")
    print("=" * 100)

    for panel_name in panels:
        print(f"\n### {panel_name}")
        print(f"{'Signal':20s}", end="")
        for r in ratios:
            print(f"  cr={r:<4}", end="")
        print()
        print("-" * 80)

        if panel_name.startswith('RULER'):
            records = ruler_data[panel_name]
            for scoring in scorings:
                print(f"{SIGNAL_LABELS.get(scoring, scoring):20s}", end="")
                for ratio in ratios:
                    acc, n = compute_accuracy(records, scoring, ratio)
                    if acc is not None:
                        print(f"  {acc:5.1f}", end="")
                    else:
                        print(f"  {'—':>5}", end="")
                print()
        elif panel_name == 'LongBench QA (avg)':
            for scoring in scorings:
                print(f"{SIGNAL_LABELS.get(scoring, scoring):20s}", end="")
                for ratio in ratios:
                    task_accs = []
                    for name, records in longbench_data.items():
                        acc, _ = compute_accuracy(records, scoring, ratio)
                        if acc is not None:
                            task_accs.append(acc)
                    if task_accs:
                        avg = sum(task_accs) / len(task_accs)
                        print(f"  {avg:5.1f}", end="")
                    else:
                        print(f"  {'—':>5}", end="")
                print()

    # Per LongBench task at cr=0.7
    if longbench_data:
        print(f"\n### LongBench QA Per-Task @ cr=0.70")
        print(f"{'Task':20s} {'Baseline':>8} {'Recons':>8} {'Q-only':>8} {'Oracle':>8}")
        print("-" * 60)
        for name in sorted(longbench_data.keys()):
            records = longbench_data[name]
            bl, _ = compute_accuracy(records, 'none', 0.0)
            rc, _ = compute_accuracy(records, 'reconstruction', 0.7)
            qo, _ = compute_accuracy(records, 'question_only', 0.7)
            oa, _ = compute_accuracy(records, 'qa_oracle', 0.7)
            print(f"{name:20s} {bl or 0:7.1f}% {rc or 0:7.1f}% {qo or 0:7.1f}% {oa or 0:7.1f}%")


if __name__ == '__main__':
    main()
