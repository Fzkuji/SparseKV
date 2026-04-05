#!/usr/bin/env python3
"""Summarize phase1 evaluation results into a table.

Usage:
    python scripts/summarize_results.py [--results_dir results/phase1_qwen3] [--csv results.csv]

Produces two tables:
  1. Summary: one column per benchmark group (longbench, ruler:4096, ruler:16384, etc.)
     - Groups like longbench aggregate all subsets into one avg score
     - ruler:4096 aggregates 13 subtasks into one avg score
  2. Detailed: every subset/subtask as its own column
"""

import argparse
import json
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path


def parse_result_dir_name(name: str):
    """Parse directory name like 'ruler__4096__Qwen--Qwen3-8B__snapkv__0.70'
    Returns (dataset, subset, model, press, compression_ratio) or None."""
    parts = name.split("__")
    if len(parts) == 5:
        dataset, subset, model, press, cr = parts
        return dataset, subset, model, press, cr
    elif len(parts) == 4:
        dataset, model, press, cr = parts
        return dataset, "", model, press, cr
    return None


def get_group(dataset, subset):
    """Map (dataset, subset) to a benchmark group for Table 1.
    
    Groups:
      - longbench:* → 'longbench'        (16 subsets aggregated)
      - ruler:4096:* → 'ruler:4096'      (13 subtasks aggregated)
      - ruler:16384:* → 'ruler:16384'
      - infinitebench:* → 'infinitebench'
      - longbench-v2:* → 'longbench-v2'
      - scbench:* → 'scbench'
      - others: dataset:subset as-is
    """
    ds_label = f"{dataset}:{subset}" if subset else dataset

    # Datasets where directory-level subsets should be grouped
    if dataset in ("longbench", "infinitebench", "scbench"):
        return dataset
    # ruler:4096, ruler:16384 — already grouped at directory level
    # (subtasks come from metrics_detail.json, not separate dirs)
    return ds_label


def get_detail_label(dataset, subset):
    """Full label for Table 2."""
    return f"{dataset}:{subset}" if subset else dataset


def read_score(metrics):
    """Extract a single score from metrics (float, int, or dict)."""
    if isinstance(metrics, (int, float)):
        return metrics
    if isinstance(metrics, dict):
        for key in ["score", "accuracy", "exact_match", "f1", "rouge_score", "avg_score"]:
            if key in metrics and isinstance(metrics[key], (int, float)):
                return metrics[key]
        subtask_scores = []
        for key, val in metrics.items():
            if isinstance(val, dict):
                for mk in ["string_match", "score", "accuracy", "f1", "exact_match"]:
                    if mk in val and isinstance(val[mk], (int, float)):
                        subtask_scores.append(val[mk])
                        break
            elif isinstance(val, (int, float)):
                subtask_scores.append(val)
        if subtask_scores:
            return sum(subtask_scores) / len(subtask_scores)
    return None


def read_subtask_scores(result_dir):
    """Read per-subtask scores from metrics_detail.json or metrics.json."""
    for fname in ["metrics_detail.json", "metrics.json"]:
        fpath = result_dir / fname
        if not fpath.exists():
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
            if isinstance(data, dict):
                subtasks = {}
                for key, val in data.items():
                    if isinstance(val, (int, float)):
                        subtasks[key] = val
                    elif isinstance(val, dict):
                        for mk in ["string_match", "score", "accuracy", "f1", "exact_match"]:
                            if mk in val and isinstance(val[mk], (int, float)):
                                subtasks[key] = val[mk]
                                break
                if len(subtasks) > 1:
                    return subtasks
        except Exception:
            pass
    return None


def format_score(s):
    """Format score: if 0-1 range (and not zero), show as percentage."""
    if s is None:
        return "-"
    if isinstance(s, float) and 0 < s < 1:
        return f"{s * 100:.1f}"
    return f"{s:.2f}" if isinstance(s, float) else str(s)


def print_table(header, table_rows):
    """Print a markdown table."""
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


def build_expected_jobs(model_tag="Qwen--Qwen3-8B"):
    """Build the full list of expected experiment directory names."""
    longbench_subsets = [
        "2wikimqa", "gov_report", "hotpotqa", "lcc", "multi_news",
        "multifieldqa_en", "musique", "narrativeqa", "passage_count",
        "passage_retrieval_en", "qasper", "qmsum", "repobench-p",
        "samsum", "trec", "triviaqa",
    ]
    scbench_tasks = [
        "scbench_choice_eng", "scbench_kv", "scbench_many_shot",
        "scbench_mf", "scbench_prefix_suffix", "scbench_qa_eng",
        "scbench_repoqa", "scbench_repoqa_and_kv", "scbench_summary",
        "scbench_summary_with_needles", "scbench_vt",
    ]
    ruler_lengths = ["4096", "16384"]
    other_datasets = [
        ("infinitebench", "longbook_qa_eng"),
        ("longbench-v2", ""),
    ]

    presses_with_ratios = [
        ("no_press", ["0.00"]),
        ("snapkv", ["0.30", "0.50", "0.70", "0.90", "0.95"]),
        ("streaming_llm", ["0.30", "0.50", "0.70", "0.90", "0.95"]),
        ("critical_snapkv", ["0.30", "0.50", "0.70", "0.90", "0.95"]),
        ("fastkvzip", ["0.30", "0.50", "0.70", "0.90", "0.95"]),
        # kvzip excluded from expected — too slow (~970min/job), existing results kept
    ]

    expected = set()
    for press, ratios in presses_with_ratios:
        for cr in ratios:
            # LongBench v1
            for subset in longbench_subsets:
                expected.add(f"longbench__{subset}__{model_tag}__{press}__{cr}")
            # SCBench
            for task in scbench_tasks:
                expected.add(f"scbench__{task}__{model_tag}__{press}__{cr}")
            # RULER
            for length in ruler_lengths:
                expected.add(f"ruler__{length}__{model_tag}__{press}__{cr}")
            # Other datasets
            for ds, sub in other_datasets:
                if sub:
                    expected.add(f"{ds}__{sub}__{model_tag}__{press}__{cr}")
                else:
                    expected.add(f"{ds}__{model_tag}__{press}__{cr}")

    return expected


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("--results_dir", default="results/phase1_qwen3",
                        help="Directory containing result subdirectories")
    parser.add_argument("--csv", default=None, help="Save detailed CSV")
    parser.add_argument("--model_tag", default="Qwen--Qwen3-8B",
                        help="Model tag for expected job enumeration")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Build expected jobs
    expected_jobs = build_expected_jobs(args.model_tag)

    # ======================================================================
    # Collect results
    # ======================================================================
    entries = []  # each entry: {group, detail_label, press, cr, score, subtasks, time_min, ...}
    missing_metrics = []  # dir exists but no metrics.json
    failed = []
    completed_dirs = set()

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir() or subdir.name == "logs":
            continue
        parsed = parse_result_dir_name(subdir.name)
        if not parsed:
            continue

        dataset, subset, model, press, cr = parsed
        metrics_path = subdir / "metrics.json"
        profiling_path = subdir / "profiling.json"

        if not metrics_path.exists():
            missing_metrics.append(subdir.name)
            continue

        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except Exception as e:
            failed.append((subdir.name, str(e)))
            continue

        score = read_score(metrics)
        subtasks = read_subtask_scores(subdir)

        time_min = None
        if profiling_path.exists():
            try:
                with open(profiling_path) as f:
                    prof = json.load(f)
                time_min = prof.get("total_time_minutes")
            except Exception:
                pass

        entries.append({
            "group": get_group(dataset, subset),
            "detail_label": get_detail_label(dataset, subset),
            "press": press,
            "cr": cr,
            "score": score,
            "subtasks": subtasks,
            "time_min": time_min,
        })
        completed_dirs.add(subdir.name)

    if not entries and not missing_metrics:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Compute not-yet-run jobs
    not_yet_run = sorted(expected_jobs - completed_dirs - set(missing_metrics))

    entries.sort(key=lambda e: (e["group"], e["detail_label"], e["press"], float(e["cr"])))

    presses = sorted(set(e["press"] for e in entries))
    crs = sorted(set(e["cr"] for e in entries), key=float)
    has_time = any(e["time_min"] for e in entries)

    # ======================================================================
    # Print header
    # ======================================================================
    total_expected = len(expected_jobs)
    total_completed = len(entries)
    total_in_progress = len(missing_metrics)
    total_not_run = len(not_yet_run)

    print(f"\n{'=' * 80}")
    print(f"  Results Summary: {results_dir}")
    print(f"  Expected: {total_expected}  |  Completed: {total_completed}  |  "
          f"In-progress: {total_in_progress}  |  Not started: {total_not_run}  |  "
          f"Failed: {len(failed)}")
    print(f"  Progress: {total_completed}/{total_expected} ({100*total_completed/total_expected:.1f}%)")
    print(f"{'=' * 80}\n")

    if missing_metrics:
        print(f"⏳ In-progress (dir exists, no metrics.json) ({len(missing_metrics)}):")
        for m in sorted(missing_metrics):
            print(f"   - {m}")
        print()
    if failed:
        print(f"❌ Failed to parse ({len(failed)}):")
        for name, err in failed:
            print(f"   - {name}: {err}")
        print()
    if not_yet_run:
        # Group by benchmark for cleaner display
        not_run_by_group = defaultdict(list)
        for name in not_yet_run:
            parsed = parse_result_dir_name(name)
            if parsed:
                ds, sub, _, press, cr = parsed
                group = get_group(ds, sub)
                not_run_by_group[group].append((press, cr))
            else:
                not_run_by_group["other"].append(name)

        print(f"📋 Not started ({len(not_yet_run)}):")
        for group in sorted(not_run_by_group):
            items = not_run_by_group[group]
            # Summarize by press, deduplicate ratios
            by_press = defaultdict(set)
            for press, cr in items:
                by_press[press].add(cr)
            parts = []
            for press in sorted(by_press):
                n = len([x for x in not_yet_run
                         if parse_result_dir_name(x) and
                         get_group(*parse_result_dir_name(x)[:2]) == group and
                         parse_result_dir_name(x)[3] == press])
                crs_list = ", ".join(sorted(by_press[press], key=float))
                parts.append(f"{press}[{n}]({crs_list})")
            print(f"   {group}: {', '.join(parts)}")
        print()

    # ======================================================================
    # Table 1: Summary — one column per group, scores averaged
    # ======================================================================
    # Build: group_scores[(group, press, cr)] = [score1, score2, ...]
    group_scores = defaultdict(list)
    group_times = defaultdict(list)
    for e in entries:
        if e["score"] is not None:
            group_scores[(e["group"], e["press"], e["cr"])].append(e["score"])
        if e["time_min"]:
            group_times[(e["group"], e["press"], e["cr"])].append(e["time_min"])

    groups = sorted(set(e["group"] for e in entries))

    print(f"{'=' * 80}")
    print(f"  Table 1: Summary (one column per benchmark, subsets averaged)")
    print(f"{'=' * 80}\n")

    header1 = ["press", "ratio"] + groups
    if has_time:
        header1.append("avg_time(min)")

    rows1 = []
    for press in presses:
        for cr in crs:
            row = [press, cr]
            any_val = False
            all_times = []
            for g in groups:
                scores = group_scores.get((g, press, cr), [])
                if scores:
                    avg = sum(scores) / len(scores)
                    row.append(format_score(avg))
                    any_val = True
                    all_times.extend(group_times.get((g, press, cr), []))
                else:
                    row.append("-")
            if any_val:
                if has_time:
                    row.append(f"{sum(all_times) / len(all_times):.1f}" if all_times else "-")
                rows1.append(row)

    print_table(header1, rows1)

    # ======================================================================
    # Table 2: All subtasks expanded
    # ======================================================================
    # For each group, collect all possible detail columns:
    #   - If group has multiple detail_labels (longbench → 16 subsets), list them
    #   - If entries have subtasks (ruler → 13 tasks), list them
    #   - Show group avg first, then individual items

    print(f"{'=' * 80}")
    print(f"  Table 2: All subtasks expanded")
    print(f"{'=' * 80}\n")

    # Build expanded columns: (display_name, type, group, detail_label_or_subtask_key)
    expanded_cols = []

    for g in groups:
        # Collect detail_labels in this group
        detail_labels = sorted(set(
            e["detail_label"] for e in entries if e["group"] == g
        ))
        # Collect subtask keys across all entries in this group
        all_subtask_keys = set()
        for e in entries:
            if e["group"] == g and e["subtasks"]:
                all_subtask_keys.update(e["subtasks"].keys())
        all_subtask_keys = sorted(all_subtask_keys)

        has_multiple_details = len(detail_labels) > 1
        has_subtasks = len(all_subtask_keys) > 0

        if has_multiple_details or has_subtasks:
            # Add group avg column
            expanded_cols.append((f"{g}(avg)", "group_avg", g, None, None))

        if has_multiple_details:
            for dl in detail_labels:
                # Each detail_label is a subset
                expanded_cols.append((dl, "detail", g, dl, None))
                # If this detail also has subtasks, expand them
                dl_subtasks = set()
                for e in entries:
                    if e["group"] == g and e["detail_label"] == dl and e["subtasks"]:
                        dl_subtasks.update(e["subtasks"].keys())
                for sk in sorted(dl_subtasks):
                    expanded_cols.append((f"{dl}:{sk}", "subtask_of_detail", g, dl, sk))
        elif has_subtasks:
            # Single detail_label but has subtasks (e.g. ruler:4096)
            dl = detail_labels[0]
            for sk in all_subtask_keys:
                expanded_cols.append((f"{g}:{sk}", "subtask", g, dl, sk))
        else:
            # No expansion needed
            if not (has_multiple_details or has_subtasks):
                expanded_cols.append((g, "simple", g, detail_labels[0] if detail_labels else g, None))

    # Index entries for lookup: (detail_label, press, cr) → entry
    entry_map = {}
    for e in entries:
        entry_map[(e["detail_label"], e["press"], e["cr"])] = e

    header2 = ["press", "ratio"] + [c[0] for c in expanded_cols]
    if has_time:
        header2.append("avg_time(min)")

    rows2 = []
    for press in presses:
        for cr in crs:
            row = [press, cr]
            any_val = False
            all_times = []

            for col_name, col_type, g, dl, sk in expanded_cols:
                if col_type == "group_avg":
                    scores = group_scores.get((g, press, cr), [])
                    if scores:
                        row.append(format_score(sum(scores) / len(scores)))
                        any_val = True
                        all_times.extend(group_times.get((g, press, cr), []))
                    else:
                        row.append("-")

                elif col_type in ("detail", "simple"):
                    e = entry_map.get((dl, press, cr))
                    if e and e["score"] is not None:
                        row.append(format_score(e["score"]))
                        any_val = True
                    else:
                        row.append("-")

                elif col_type == "subtask":
                    e = entry_map.get((dl, press, cr))
                    if e and e["subtasks"] and sk in e["subtasks"]:
                        row.append(format_score(e["subtasks"][sk]))
                        any_val = True
                    else:
                        row.append("-")

                elif col_type == "subtask_of_detail":
                    e = entry_map.get((dl, press, cr))
                    if e and e["subtasks"] and sk in e["subtasks"]:
                        row.append(format_score(e["subtasks"][sk]))
                        any_val = True
                    else:
                        row.append("-")
                else:
                    row.append("-")

            if any_val:
                if has_time:
                    row.append(f"{sum(all_times) / len(all_times):.1f}" if all_times else "-")
                rows2.append(row)

    print_table(header2, rows2)

    # ======================================================================
    # Detailed per-experiment list
    # ======================================================================
    print(f"\n{'=' * 80}")
    print("  Detailed Results (per experiment)")
    print(f"{'=' * 80}\n")
    for e in entries:
        score_str = format_score(e["score"]) if e["score"] is not None else "N/A"
        time_str = f"{e['time_min']:.1f}min" if e["time_min"] else ""
        prof = f"  [{time_str}]" if time_str else ""
        print(f"  {e['detail_label']:30s} {e['press']:20s} cr={e['cr']}  → {score_str}{prof}")

    # CSV
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header2)
            writer.writerows(rows2)
        print(f"\nCSV saved to {args.csv}")


if __name__ == "__main__":
    main()
