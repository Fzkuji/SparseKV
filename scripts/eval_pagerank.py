#!/usr/bin/env python3
"""Focused evaluation: test PageRankPress on RULER NIAH tasks."""

import json
import sys
import torch
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline

from kvpress import PageRankPress, ObservedAttentionPress
from kvpress.presses.tova_press import TOVAPress

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path.home() / "kvpress/evaluation"))
from evaluate_registry import SCORER_REGISTRY
scorer = SCORER_REGISTRY["ruler"]


def evaluate_press(pipe, df_task, press, task_name):
    """Evaluate a single press on a single task."""
    df = df_task.copy()
    df["predicted_answer"] = None
    df_grouped = df.groupby("context")
    for context, df_group in tqdm(df_grouped, total=df["context"].nunique(), desc=task_name):
        questions = df_group["question"].to_list()
        max_new_tokens = df_group["max_new_tokens"].iloc[0]
        answer_prefix = df_group["answer_prefix"].iloc[0]
        output = pipe(
            context, questions=questions, answer_prefix=answer_prefix,
            press=press, max_new_tokens=max_new_tokens,
        )
        df.loc[df_group.index, "predicted_answer"] = output["answers"]
        torch.cuda.empty_cache()
    return scorer(df)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+",
                       default=["niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
                                "niah_single_1", "niah_single_3"])
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./results/pagerank")
    parser.add_argument("--model_label", type=str, default="model")
    args = parser.parse_args()

    logger.info("Loading RULER dataset (4096)...")
    ds = load_dataset("simonjegou/ruler", data_dir="4096", split="test")
    df_all = ds.to_pandas()
    df_filtered = df_all[df_all["task"].isin(args.tasks)]
    logger.info(f"Filtered to {len(df_filtered)} samples across tasks: {args.tasks}")

    press = PageRankPress(
        compression_ratio=args.compression_ratio,
        damping=args.damping,
        num_iterations=args.num_iterations,
    )
    logger.info(f"PageRankPress: damping={args.damping}, iterations={args.num_iterations}")
    logger.info(f"Loading model with eager attention: {args.model}")

    pipe = pipeline(
        "kv-press-text-generation",
        model=args.model,
        model_kwargs={"attn_implementation": "eager", "dtype": "auto"},
        device_map="auto",
        trust_remote_code=True,
    )
    pipe.model.eval()

    results = {}
    for task_name in args.tasks:
        df_task = df_filtered[df_filtered["task"] == task_name]
        logger.info(f"Task: {task_name} ({len(df_task)} samples)")
        score = evaluate_press(pipe, df_task, press, task_name)
        results[task_name] = score
        logger.info(f"  Score: {score}")

    print("\n" + "="*60)
    print(f"PageRank Results ({args.model_label}, damping={args.damping}, iter={args.num_iterations})")
    print("="*60)
    for task, score in results.items():
        print(f"  {task}: {score}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"pagerank_{args.model_label}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
