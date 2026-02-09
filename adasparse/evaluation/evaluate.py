# SPDX-License-Identifier: Apache-2.0

"""
Evaluation script for AdaSparseKV.

Evaluates trained models with various KV cache eviction methods using kvpress,
and computes sparsity/stability metrics.

Usage:
    adasparse-eval --config configs/eval/full_eval.yaml

    python -m adasparse.evaluation.evaluate \
        --model_path ./output/block_dropout_8b \
        --press_names snapkv expected_attention streaming_llm \
        --compression_ratios 0.3 0.5 0.7
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, pipeline

from adasparse.evaluation.sparsity import compute_sparsity_metrics
from adasparse.evaluation.stability import compute_stability_metrics

logger = logging.getLogger(__name__)

# Map press names to kvpress classes
KVPRESS_REGISTRY: dict = {}


def _load_kvpress_registry():
    """Lazily load kvpress press classes."""
    global KVPRESS_REGISTRY
    if KVPRESS_REGISTRY:
        return

    try:
        from kvpress import (
            ExpectedAttentionPress,
            KnormPress,
            ObservedAttentionPress,
            RandomPress,
            SnapKVPress,
            StreamingLLMPress,
            TOVAPress,
        )

        KVPRESS_REGISTRY.update({
            "snapkv": SnapKVPress,
            "expected_attention": ExpectedAttentionPress,
            "streaming_llm": StreamingLLMPress,
            "observed_attention": ObservedAttentionPress,
            "knorm": KnormPress,
            "tova": TOVAPress,
            "random": RandomPress,
        })
    except ImportError:
        logger.warning("kvpress not installed. Only sparsity/stability metrics available.")

    # Also try to import PyramidKV and others
    try:
        from kvpress import PyramidKVPress

        KVPRESS_REGISTRY["pyramidkv"] = PyramidKVPress
    except ImportError:
        pass


def evaluate_with_press(
    model_path: str,
    press_name: str,
    compression_ratio: float,
    texts: list[str],
    questions: list[str],
    device: str = "auto",
) -> dict:
    """
    Evaluate a model with a specific kvpress press.

    Returns perplexity-like metrics and generation quality.
    """
    _load_kvpress_registry()

    press_cls = KVPRESS_REGISTRY.get(press_name)
    if press_cls is None:
        raise ValueError(f"Unknown press: {press_name}. Available: {list(KVPRESS_REGISTRY.keys())}")

    pipe = pipeline(
        "kv-press-text-generation",
        model=model_path,
        device_map=device,
        torch_dtype="auto",
    )

    press = press_cls(compression_ratio=compression_ratio)

    results = []
    for text, question in zip(texts, questions):
        output = pipe(text, question=question, press=press)
        results.append({"answer": output["answer"]})

    return {
        "press": press_name,
        "compression_ratio": compression_ratio,
        "num_examples": len(results),
        "results": results,
    }


@torch.no_grad()
def evaluate_sparsity(
    model_path: str,
    texts: list[str],
    tokenizer_path: str | None = None,
    max_length: int = 2048,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Compute sparsity and stability metrics for a model.

    Parameters
    ----------
    model_path : str
        Path to model or HuggingFace model name.
    texts : list[str]
        Input texts to evaluate on.
    max_length : int
        Maximum sequence length for evaluation.
    device : str
        Device to run on.

    Returns
    -------
    dict
        Aggregated sparsity and stability metrics.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    all_sparsity = []
    all_stability = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**inputs, output_attentions=True)

        for layer_attn in outputs.attentions:
            sparsity = compute_sparsity_metrics(layer_attn)
            all_sparsity.append(sparsity)

            stability = compute_stability_metrics(layer_attn)
            all_stability.append(stability)

    # Average across layers and examples
    avg_sparsity = {}
    avg_stability = {}

    if all_sparsity:
        for key in all_sparsity[0]:
            avg_sparsity[key] = sum(s[key] for s in all_sparsity) / len(all_sparsity)

    if all_stability:
        for key in all_stability[0]:
            avg_stability[key] = sum(s[key] for s in all_stability) / len(all_stability)

    return {**avg_sparsity, **avg_stability}


def main():
    parser = argparse.ArgumentParser(description="AdaSparseKV Evaluation")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--baseline_model", type=str, default=None, help="Original model for comparison")
    parser.add_argument("--press_names", nargs="+", default=["snapkv", "streaming_llm"])
    parser.add_argument("--compression_ratios", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--eval_sparsity", action="store_true", help="Compute sparsity metrics")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load config if provided
    config = vars(args)
    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
        config.update(yaml_config)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample evaluation texts (replace with actual benchmark loading)
    eval_texts = [
        "This is a placeholder for evaluation text. In practice, load from RULER/LongBench/etc."
    ]

    results = {}

    # Sparsity evaluation
    if config.get("eval_sparsity") and config.get("model_path"):
        logger.info("Computing sparsity metrics...")
        sparsity = evaluate_sparsity(config["model_path"], eval_texts)
        results["sparsity"] = sparsity
        logger.info(f"Sparsity metrics: {sparsity}")

        # Compare with baseline
        if config.get("baseline_model"):
            logger.info("Computing baseline sparsity metrics...")
            baseline_sparsity = evaluate_sparsity(config["baseline_model"], eval_texts)
            results["baseline_sparsity"] = baseline_sparsity
            logger.info(f"Baseline sparsity: {baseline_sparsity}")

    # Press evaluation
    if config.get("model_path"):
        for press_name in config["press_names"]:
            for ratio in config["compression_ratios"]:
                logger.info(f"Evaluating {press_name} @ {ratio:.1%} compression...")
                try:
                    eval_result = evaluate_with_press(
                        config["model_path"],
                        press_name,
                        ratio,
                        eval_texts,
                        ["Summarize the text."] * len(eval_texts),
                        device=config.get("device", "auto"),
                    )
                    key = f"{press_name}_cr{ratio}"
                    results[key] = eval_result
                    logger.info(f"  Done: {len(eval_result['results'])} examples")
                except Exception as e:
                    logger.error(f"  Failed: {e}")

    # Save results
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
