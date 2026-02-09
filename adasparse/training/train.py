# SPDX-License-Identifier: Apache-2.0

"""
Main training script for AdaSparseKV.

Usage:
    adasparse-train --config configs/train/block_dropout_8b.yaml

    # Or with command-line overrides:
    python -m adasparse.training.train \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
        --press_type block_dropout \
        --drop_ratio 0.3 \
        --output_dir ./output/block_dropout_8b
"""

import argparse
import logging
import sys

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from adasparse.presses.adaptive_dropout_press import AdaptiveBlockDropoutPress
from adasparse.presses.block_dropout_press import BlockDropoutPress
from adasparse.presses.eviction_aug_press import EvictionAugPress
from adasparse.presses.soft_threshold_press import SoftThresholdPress
from adasparse.presses.sparse_reg_press import SparseRegPress
from adasparse.presses.variable_block_press import VariableBlockDropoutPress
from adasparse.training.curriculum import CosineCurriculum, LinearCurriculum, StepCurriculum
from adasparse.training.data import DataConfig, create_data_collator, load_dataset_for_training
from adasparse.training.trainer import AdaSparseTrainer

logger = logging.getLogger(__name__)

PRESS_REGISTRY = {
    "block_dropout": BlockDropoutPress,
    "adaptive_block_dropout": AdaptiveBlockDropoutPress,
    "variable_block_dropout": VariableBlockDropoutPress,
    "soft_threshold": SoftThresholdPress,
    "sparse_reg": SparseRegPress,
}

CURRICULUM_REGISTRY = {
    "linear": LinearCurriculum,
    "step": StepCurriculum,
    "cosine": CosineCurriculum,
}


def create_press(config: dict) -> tuple[BlockDropoutPress, SparseRegPress | None]:
    """Create press and optional regularization press from config."""
    press_type = config.get("press_type", "block_dropout")
    press_cls = PRESS_REGISTRY.get(press_type)

    if press_cls is None:
        raise ValueError(f"Unknown press type: {press_type}. Available: {list(PRESS_REGISTRY.keys())}")

    # Build press kwargs from config
    press_kwargs = {}
    if press_type == "block_dropout":
        press_kwargs = {
            "block_size": config.get("block_size", 64),
            "drop_ratio": config.get("drop_ratio", 0.3),
            "protect_start": config.get("protect_start", 4),
            "protect_recent": config.get("protect_recent", 64),
            "training": True,
        }
    elif press_type == "adaptive_block_dropout":
        press_kwargs = {
            "block_size": config.get("block_size", 64),
            "base_drop_ratio": config.get("drop_ratio", 0.3),
            "protect_start": config.get("protect_start", 4),
            "protect_recent": config.get("protect_recent", 64),
            "sink_weight": config.get("sink_weight", 0.3),
            "temperature": config.get("temperature", 1.0),
            "training": True,
        }
    elif press_type == "variable_block_dropout":
        press_kwargs = {
            "min_block_size": config.get("min_block_size", 32),
            "max_block_size": config.get("max_block_size", 128),
            "drop_ratio": config.get("drop_ratio", 0.3),
            "protect_start": config.get("protect_start", 4),
            "protect_recent": config.get("protect_recent", 64),
            "training": True,
        }
    elif press_type == "soft_threshold":
        press_kwargs = {
            "threshold": config.get("threshold", 0.01),
            "temperature": config.get("temperature", 10.0),
            "training": True,
        }

    press = press_cls(**press_kwargs)

    # Optional regularization press
    reg_press = None
    if config.get("sparse_reg_weight", 0) > 0:
        reg_press = SparseRegPress(
            reg_type=config.get("sparse_reg_type", "entropy"),
            reg_weight=config.get("sparse_reg_weight", 0.01),
            training=True,
        )

    return press, reg_press


def create_curriculum(config: dict) -> LinearCurriculum | None:
    """Create curriculum schedule from config."""
    if not config.get("use_curriculum", True):
        return None

    curriculum_type = config.get("curriculum_type", "linear")
    curriculum_cls = CURRICULUM_REGISTRY.get(curriculum_type)

    if curriculum_cls is None:
        raise ValueError(f"Unknown curriculum: {curriculum_type}")

    return curriculum_cls(
        start_ratio=config.get("curriculum_start_ratio", 0.0),
        end_ratio=config.get("curriculum_end_ratio", 0.5),
        warmup_steps=config.get("curriculum_warmup_steps", 1000),
    )


def load_config(config_path: str) -> dict:
    """Load training config from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Flatten nested config for easier access
    flat = {}
    for section in config.values():
        if isinstance(section, dict):
            flat.update(section)
        else:
            pass
    return {**config, **flat}


def main():
    parser = argparse.ArgumentParser(description="AdaSparseKV Training")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--press_type", type=str, default=None)
    parser.add_argument("--drop_ratio", type=float, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--dataset_name", type=str, default="allenai/c4")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--objective_type", type=str, default=None,
                        choices=["standard_lm", "reconstruction", "sparse_lm", "mixed"],
                        help="Training objective type")

    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        config = load_config(args.config)

    # CLI overrides
    for key, val in vars(args).items():
        if val is not None and key != "config":
            config[key] = val

    # Defaults
    config.setdefault("model_name_or_path", "meta-llama/Llama-3.1-8B-Instruct")
    config.setdefault("output_dir", "./output")
    config.setdefault("press_type", "block_dropout")
    config.setdefault("drop_ratio", 0.3)
    config.setdefault("block_size", 64)
    config.setdefault("max_seq_length", 4096)
    config.setdefault("learning_rate", 1e-5)
    config.setdefault("num_train_epochs", 1)
    config.setdefault("per_device_train_batch_size", 4)
    config.setdefault("gradient_accumulation_steps", 8)

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Training config: {config}")

    # Load model and tokenizer
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(config.get("dtype", "bfloat16"), torch.bfloat16)

    logger.info(f"Loading model: {config['model_name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",  # Need attention weights for adaptive press
    )

    # Create press and curriculum
    press, reg_press = create_press(config)
    curriculum = create_curriculum(config)

    logger.info(f"Press: {press}")
    logger.info(f"Curriculum: {curriculum}")

    # Load dataset
    data_config = DataConfig(
        dataset_name=config.get("dataset_name", "allenai/c4"),
        max_seq_length=config.get("max_seq_length", 4096),
        streaming=config.get("streaming", True),
    )
    train_dataset = load_dataset_for_training(data_config, tokenizer)
    logger.info(f"Dataset loaded: {len(train_dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config.get("warmup_steps", 100),
        weight_decay=config.get("weight_decay", 0.01),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 3),
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        dataloader_pin_memory=True,
        report_to=config.get("report_to", "tensorboard"),
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = AdaSparseTrainer(
        press=press,
        curriculum=curriculum,
        reg_press=reg_press,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=create_data_collator(tokenizer),
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {config['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(config["output_dir"])

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
