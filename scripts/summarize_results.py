#!/usr/bin/env python3
"""Summarize phase1 evaluation results into a table.

Usage:
    python scripts/summarize_results.py [--results_dir results/phase1_qwen3] [--csv results.csv]

Reads all metrics.json files, produces a table grouped by dataset × press × compression_ratio.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_result_dir_name(name: str):
    """Parse directory name like 'ruler__4096__Qwen--Qwen3-8B__snapkv__0.70'
    Returns (dataset, subset, model, press, compression_ratio) or None."""
    # Pattern: {dataset}__{subset}__{model}__{press}__{cr}
    # or:      {dataset}__{model}__{press}__{cr}  (no subset)
    parts = name.split("__")
    if len(parts) == 5:
        dataset, subset, model, press, cr = parts
        return dataset, subset, model, press, cr
    elif len(parts) == 4:
        dataset, model, press, cr = parts
        return dataset, "", model, press, cr
    return None


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("--results_dir", default="results/phase1_qwen3",
                        help="Directory containing result subdirectories")
    parser.add_argument("--csv", default=None,
                        help="Save CSV to this path (default: print to stdout)")
    parser.add_argument("--format", choices=["table", "csv", "markdown"], default="markdown",
                        help="Output format")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Collect all results
    rows = []
    missing = []
    failed = []

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir() or subdir.name == "logs":
            continue

        parsed = parse_result_dir_name(subdir.name)
        if not parsed:
            continue

        dataset, subset, model, press, cr = parsed
        ds_label = f"{dataset}:{subset}" if subset else dataset

        metrics_path = subdir / "metrics.json"
        profiling_path = subdir / "profiling.json"

        if not metrics_path.exists():
            missing.append(subdir.name)
            continue

        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except Exception as e:
            failed.append((subdir.name, str(e)))
            continue

        # Extract main score
        score = None
        score_key = None
        if isinstance(metrics, dict):
            # Try top-level scalar keys first
            for key in ["score", "accuracy", "exact_match", "f1", "rouge_score", "avg_score"]:
                if key in metrics and isinstance(metrics[key], (int, float)):
                    score = metrics[key]
                    score_key = key
                    break

            # If no top-level scalar, check if it's nested subtask format:
            # {"task1": {"string_match": 99.0}, "task2": {"string_match": 95.0}, ...}
            if score is None:
                subtask_scores = []
                for key, val in metrics.items():
                    if isinstance(val, dict):
                        # Try common metric keys within subtask
                        for mk in ["string_match", "score", "accuracy", "f1", "exact_match"]:
                            if mk in val and isinstance(val[mk], (int, float)):
                                subtask_scores.append(val[mk])
                                break
                    elif isinstance(val, (int, float)):
                        subtask_scores.append(val)
                if subtask_scores:
                    score = sum(subtask_scores) / len(subtask_scores)
                    score_key = f"avg({len(subtask_scores)} subtasks)"
        elif isinstance(metrics, (int, float)):
            score = metrics
            score_key = "value"

        # Profiling info
        time_min = None
        peak_gpu_gb = None
        if profiling_path.exists():
            try:
                with open(profiling_path) as f:
                    prof = json.load(f)
                time_min = prof.get("total_time_minutes")
                peak_gpu_gb = prof.get("peak_gpu_memory_gb")
            except Exception:
                pass

        rows.append({
            "dataset": ds_label,
            "press": press,
            "compression_ratio": cr,
            "score": score,
            "score_key": score_key,
            "time_min": time_min,
            "peak_gpu_gb": peak_gpu_gb,
            "all_metrics": metrics,
        })

    if not rows and not missing:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Sort: dataset → press → compression_ratio
    rows.sort(key=lambda r: (r["dataset"], r["press"], float(r["compression_ratio"])))

    # Print summary
    print(f"\n{'='*80}")
    print(f"  Results Summary: {results_dir}")
    print(f"  Completed: {len(rows)}  |  Missing metrics: {len(missing)}  |  Failed: {len(failed)}")
    print(f"{'='*80}\n")

    if missing:
        print(f"⚠️  Missing metrics.json ({len(missing)}):")
        for m in missing:
            print(f"   - {m}")
        print()

    if failed:
        print(f"❌ Failed to parse ({len(failed)}):")
        for name, err in failed:
            print(f"   - {name}: {err}")
        print()

    # Build table
    datasets = sorted(set(r["dataset"] for r in rows))
    presses = sorted(set(r["press"] for r in rows))
    crs = sorted(set(r["compression_ratio"] for r in rows), key=float)

    # Pivot table: rows = press × cr, columns = dataset
    header = ["press", "ratio"] + datasets
    if any(r["time_min"] for r in rows):
        header.append("avg_time(min)")

    # Index results for quick lookup
    result_map = {}
    for r in rows:
        result_map[(r["dataset"], r["press"], r["compression_ratio"])] = r

    table_rows = []
    for press in presses:
        for cr in crs:
            row_data = [press, cr]
            times = []
            has_any = False
            for ds in datasets:
                r = result_map.get((ds, press, cr))
                if r and r["score"] is not None:
                    # Format score as percentage if < 1, otherwise as-is
                    s = r["score"]
                    if isinstance(s, float) and 0 <= s <= 1:
                        row_data.append(f"{s*100:.1f}")
                    else:
                        row_data.append(f"{s:.2f}" if isinstance(s, float) else str(s))
                    has_any = True
                    if r["time_min"]:
                        times.append(r["time_min"])
                else:
                    row_data.append("-")

            if has_any:
                if any(r["time_min"] for r in rows):
                    avg_t = f"{sum(times)/len(times):.1f}" if times else "-"
                    row_data.append(avg_t)
                table_rows.append(row_data)

    # Output
    if args.format == "markdown" or args.format == "table":
        # Calculate column widths
        col_widths = [max(len(str(row[i])) for row in [header] + table_rows)
                      for i in range(len(header))]

        def fmt_row(row):
            return "| " + " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)) + " |"

        print(fmt_row(header))
        print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
        prev_press = None
        for row in table_rows:
            if prev_press and row[0] != prev_press:
                print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
            print(fmt_row(row))
            prev_press = row[0]
        print()

    # CSV output
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(table_rows)
        print(f"CSV saved to {args.csv}")

    # Also dump raw per-experiment details
    print(f"\n{'='*80}")
    print("  Detailed Results")
    print(f"{'='*80}\n")
    for r in rows:
        score_str = f"{r['score']}" if r['score'] is not None else "N/A"
        time_str = f"{r['time_min']:.1f}min" if r['time_min'] else ""
        gpu_str = f"{r['peak_gpu_gb']:.1f}GB" if r['peak_gpu_gb'] else ""
        prof = f"  [{time_str} {gpu_str}]".strip() if (time_str or gpu_str) else ""
        print(f"  {r['dataset']:30s} {r['press']:20s} cr={r['compression_ratio']}  → {score_str}{prof}")


if __name__ == "__main__":
    main()
