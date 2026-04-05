#!/usr/bin/env python3
"""Focused evaluation: run specific RULER tasks with specific presses."""

import json
import sys
import torch
import logging
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline

from kvpress import ObservedAttentionPress, SnapKVPress
from kvpress.presses.tova_press import TOVAPress
from kvpress.presses.criticalkv_press import CriticalKVPress

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# RULER scorer (exact match)
sys.path.insert(0, str(Path.home() / "kvpress/evaluation"))
from evaluate_registry import SCORER_REGISTRY
scorer = SCORER_REGISTRY["ruler"]


def evaluate_press(pipe, df_task, press, task_name):
    """Evaluate a single press on a single task."""
    df = df_task.copy()
    df["predicted_answer"] = None

    df_grouped = df.groupby("context")
    for context, df_group in tqdm(df_grouped, total=df["context"].nunique(),
                                   desc=f"{task_name}"):
        questions = df_group["question"].to_list()
        max_new_tokens = df_group["max_new_tokens"].iloc[0]
        answer_prefix = df_group["answer_prefix"].iloc[0]

        output = pipe(
            context,
            questions=questions,
            answer_prefix=answer_prefix,
            press=press,
            max_new_tokens=max_new_tokens,
        )
        df.loc[df_group.index, "predicted_answer"] = output["answers"]
        torch.cuda.empty_cache()

    score = scorer(df)
    return score


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+",
                       default=["niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
                                "niah_single_1", "niah_single_3"])
    parser.add_argument("--presses", type=str, nargs="+",
                       default=["observed_attention", "tova", "snapkv", "critical_snapkv"])
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="./results/focused")
    args = parser.parse_args()

    # Load dataset
    logger.info("Loading RULER dataset (4096)...")
    ds = load_dataset("simonjegou/ruler", data_dir="4096", split="test")
    df_all = ds.to_pandas()
    logger.info(f"Total samples: {len(df_all)}")

    # Filter tasks
    df_filtered = df_all[df_all["task"].isin(args.tasks)]
    logger.info(f"Filtered to {len(df_filtered)} samples across tasks: {args.tasks}")

    # Build presses
    press_map = {
        "observed_attention": ObservedAttentionPress(compression_ratio=args.compression_ratio),
        "tova": TOVAPress(compression_ratio=args.compression_ratio),
        "snapkv": SnapKVPress(compression_ratio=args.compression_ratio),
        "critical_snapkv": CriticalKVPress(
            press=SnapKVPress(compression_ratio=args.compression_ratio)
        ),
    }

    results = {}

    for press_name in args.presses:
        press = press_map[press_name]
        logger.info("\n" + "="*60)
        logger.info(f"Press: {press_name} (compression_ratio={args.compression_ratio})")
        logger.info("="*60)

        # Determine attention implementation
        needs_eager = press_name in ("observed_attention", "tova")
        attn_impl = "eager" if needs_eager else "flash_attention_2"

        logger.info(f"Loading model with attn_implementation={attn_impl}...")
        pipe = pipeline(
            "kv-press-text-generation",
            model=args.model,
            model_kwargs={"attn_implementation": attn_impl, "dtype": "auto"},
            device_map="auto",
            trust_remote_code=True,
        )
        pipe.model.eval()

        results[press_name] = {}
        for task_name in args.tasks:
            df_task = df_filtered[df_filtered["task"] == task_name]
            logger.info(f"\nTask: {task_name} ({len(df_task)} samples)")
            score = evaluate_press(pipe, df_task, press, task_name)
            results[press_name][task_name] = score
            logger.info(f"  Score: {score}")

        # Free model
        del pipe
        torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "="*80)
    print(f"RESULTS (compression_ratio={args.compression_ratio})")
    print("="*80)
    header = f"{Press:<25}" + "".join(f"{t:<20}" for t in args.tasks)
    print(header)
    print("-"*len(header))
    for press_name in args.presses:
        row = f"{press_name:<25}"
        for task_name in args.tasks:
            val = results[press_name].get(task_name, {})
            # Score might be a dict with task-specific metrics
            if isinstance(val, dict):
                # Get the value for this specific task
                v = val.get(task_name, val)
                if isinstance(v, (int, float)):
                    row += f"{v:<20.1f}"
                else:
                    row += f"{str(v):<20}"
            elif isinstance(val, (int, float)):
                row += f"{val:<20.1f}"
            else:
                row += f"{str(val):<20}"
        print(row)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "focused_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_dir / focused_results.json}")


if __name__ == "__main__":
    main()
