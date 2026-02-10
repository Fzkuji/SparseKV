#!/usr/bin/env python3
"""
Launch EIT (Eviction-Invariant Training).

Usage:
    python scripts/train_eit.py --model Qwen/Qwen3-8B --output_dir ./output/qwen3_eit
    python scripts/train_eit.py --model meta-llama/Llama-3.1-8B-Instruct --output_dir ./output/llama_eit
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sparsekv.training import EITTrainer, EITConfig, EvictionConfig, LossConfig, SchedulerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple dataset that tokenizes text samples to fixed length."""
    
    def __init__(self, tokenizer, dataset_name, subset, split, max_seq_len, num_samples):
        logger.info(f"Loading dataset: {dataset_name} ({subset}), split={split}")
        
        ds = load_dataset(dataset_name, subset, split=split, streaming=True)
        
        self.samples = []
        for i, example in enumerate(ds):
            if i >= num_samples * 2:  # Load extra to account for short texts
                break
            
            text = example.get("text", "")
            if len(text) < 100:
                continue
            
            tokens = tokenizer(
                text,
                max_length=max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            
            self.samples.append({
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            })
            
            if len(self.samples) >= num_samples:
                break
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    parser = argparse.ArgumentParser(description="EIT Training")
    
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="./output/eit")
    
    # Training
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--num_train_samples", type=int, default=10000)
    parser.add_argument("--num_val_samples", type=int, default=500)
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--no_lora", action="store_true")
    
    # EIT
    parser.add_argument("--lambda_eit", type=float, default=1.0)
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "cosine", "kl", "huber"])
    parser.add_argument("--layer_weighting", type=str, default="uniform", choices=["uniform", "linear", "learned"])
    
    # Eviction
    parser.add_argument("--initial_keep_ratio", type=float, default=0.9)
    parser.add_argument("--min_keep_ratio", type=float, default=0.3)
    parser.add_argument("--scheduler_mode", type=str, default="curriculum", choices=["fixed", "curriculum", "adaptive"])
    parser.add_argument("--adaptive_heads", action="store_true", default=True)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=64)
    
    # Data
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_subset", type=str, default="sample-10BT")
    
    args = parser.parse_args()
    
    # Build config
    config = EITConfig(
        model_name=args.model,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        num_train_samples=args.num_train_samples,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        dataset_subset=args.dataset_subset,
        eviction=EvictionConfig(
            keep_ratio=args.initial_keep_ratio,
            sink_size=args.sink_size,
            recent_size=args.recent_size,
            adaptive_heads=args.adaptive_heads,
        ),
        loss=LossConfig(
            lambda_eit=args.lambda_eit,
            loss_type=args.loss_type,
            layer_weighting=args.layer_weighting,
        ),
        scheduler=SchedulerConfig(
            initial_keep_ratio=args.initial_keep_ratio,
            min_keep_ratio=args.min_keep_ratio,
            mode=args.scheduler_mode,
        ),
    )
    
    # Initialize trainer
    trainer = EITTrainer(config)
    
    # Load data
    tokenizer = trainer.tokenizer
    
    train_dataset = TextDataset(
        tokenizer, args.dataset, args.dataset_subset,
        "train", args.max_seq_len, args.num_train_samples,
    )
    val_dataset = TextDataset(
        tokenizer, args.dataset, args.dataset_subset,
        "train", args.max_seq_len, args.num_val_samples,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
    )
    
    # Train
    trainer.train(train_loader, val_loader)
    
    logger.info(f"Done! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
