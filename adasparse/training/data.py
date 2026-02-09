# SPDX-License-Identifier: Apache-2.0

"""
Data loading utilities for AdaSparseKV training.

Supports loading from HuggingFace datasets with proper tokenization
and sequence packing for efficient long-context training.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class DataConfig:
    """Configuration for data loading."""

    dataset_name: str = "allenai/c4"
    dataset_config: str | None = "en"
    split: str = "train"
    max_seq_length: int = 4096
    streaming: bool = True
    num_proc: int = 4
    seed: int = 42


class PackedDataset(Dataset):
    """
    Packed dataset that concatenates and chunks texts into fixed-length sequences.

    This is the standard approach for efficient LM training: concatenate all texts
    with EOS separators and chunk into max_seq_length sequences.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Tokenize and pack
        self.input_ids = self._pack(texts)

    def _pack(self, texts: list[str]) -> list[torch.Tensor]:
        """Tokenize texts, concatenate, and chunk into fixed-length sequences."""
        all_ids: list[int] = []

        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            if self.tokenizer.eos_token_id is not None:
                all_ids.append(self.tokenizer.eos_token_id)

        # Chunk into sequences
        chunks = []
        for i in range(0, len(all_ids) - self.max_seq_length, self.max_seq_length):
            chunk = torch.tensor(all_ids[i : i + self.max_seq_length], dtype=torch.long)
            chunks.append(chunk)

        return chunks

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids = self.input_ids[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }


def load_dataset_for_training(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """
    Load and prepare a dataset for training.

    Parameters
    ----------
    config : DataConfig
        Data configuration.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer for encoding texts.

    Returns
    -------
    Dataset
        A packed dataset ready for training.
    """
    from datasets import load_dataset

    # Load dataset
    kwargs: dict[str, Any] = {"streaming": config.streaming}
    if config.dataset_config:
        kwargs["name"] = config.dataset_config

    raw_dataset = load_dataset(config.dataset_name, split=config.split, **kwargs)

    if config.streaming:
        # For streaming datasets, collect a buffer of texts
        texts = []
        target_tokens = 100_000  # Collect ~100K tokens worth
        token_count = 0
        for example in raw_dataset:
            text = example.get("text", "")
            if text:
                texts.append(text)
                token_count += len(text.split()) * 1.3  # rough estimate
                if token_count >= target_tokens:
                    break
    else:
        texts = [ex["text"] for ex in raw_dataset if ex.get("text")]

    return PackedDataset(texts, tokenizer, config.max_seq_length)


def create_data_collator(tokenizer: PreTrainedTokenizerBase):
    """Create a simple data collator that pads sequences."""

    def collate_fn(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch = {
            "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
            "labels": torch.stack([ex["labels"] for ex in examples]),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in examples]),
        }
        return batch

    return collate_fn
