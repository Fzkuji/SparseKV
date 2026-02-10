#!/usr/bin/env python3
"""
Train SparseKV: Anchor-aware KV dropout training.

Usage:
    python scripts/train_sparsekv.py --model Qwen/Qwen3-8B --output_dir ./output/qwen3_sparsekv
    python scripts/train_sparsekv.py --model meta-llama/Llama-3.1-8B-Instruct --output_dir ./output/llama_sparsekv
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from sparsekv.training import SparseKVTrainer, TrainConfig, AnchorConfig, SchedulerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log"),
    ],
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Tokenized text dataset."""
    
    def __init__(self, tokenizer, dataset_name, subset, split, max_seq_len, num_samples):
        logger.info(f"Loading dataset: {dataset_name} ({subset})")
        ds = load_dataset(dataset_name, subset, split=split, streaming=True)
        
        self.samples = []
        for i, example in enumerate(ds):
            if i >= num_samples * 3:
                break
            text = example.get("text", "")
            if len(text) < 200:
                continue
            
            tokens = tokenizer(
                text, max_length=max_seq_len, truncation=True,
                padding="max_length", return_tensors="pt",
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
    parser = argparse.ArgumentParser(description="SparseKV Training")
    
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="./output/sparsekv")
    
    # Training
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--num_train_samples", type=int, default=10000)
    parser.add_argument("--num_val_samples", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=64)
    
    # SparseKV
    parser.add_argument("--lambda_kl", type=float, default=1.0)
    parser.add_argument("--initial_keep_ratio", type=float, default=0.9)
    parser.add_argument("--min_keep_ratio", type=float, default=0.3)
    parser.add_argument("--scheduler_mode", type=str, default="curriculum",
                        choices=["fixed", "curriculum", "adaptive"])
    
    # Anchor
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=64)
    parser.add_argument("--no_punctuation", action="store_true")
    
    # Data
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_subset", type=str, default="sample-10BT")
    
    args = parser.parse_args()
    
    config = TrainConfig(
        model_name=args.model,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        output_dir=args.output_dir,
        lambda_kl=args.lambda_kl,
        dataset_name=args.dataset,
        dataset_subset=args.dataset_subset,
        anchor=AnchorConfig(
            sink_size=args.sink_size,
            recent_size=args.recent_size,
            use_punctuation=not args.no_punctuation,
        ),
        scheduler=SchedulerConfig(
            initial_keep_ratio=args.initial_keep_ratio,
            min_keep_ratio=args.min_keep_ratio,
            mode=args.scheduler_mode,
        ),
    )
    
    trainer = SparseKVTrainer(config)
    
    # Load data
    train_dataset = TextDataset(
        trainer.tokenizer, args.dataset, args.dataset_subset,
        "train", args.max_seq_len, args.num_train_samples,
    )
    val_dataset = TextDataset(
        trainer.tokenizer, args.dataset, args.dataset_subset,
        "train", args.max_seq_len, args.num_val_samples,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Merge LoRA and save final model
    trainer.merge_and_save(os.path.join(args.output_dir, "merged"))
    
    logger.info("Done!")


if __name__ == "__main__":
    import os
    main()
