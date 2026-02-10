# SPDX-License-Identifier: Apache-2.0

"""
BlockDropoutPress: Fixed-size block-wise KV cache dropout.

During training, randomly drops contiguous blocks of KV cache entries to teach the model
robustness to missing context. Sink tokens (initial tokens) and recent tokens are protected.
"""

import random
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.base_press import BasePress


@dataclass
class BlockDropoutPress(BasePress):
    """
    Fixed-size block-wise dropout on KV cache.

    Splits the KV cache into non-overlapping blocks and randomly drops a fraction of them.
    Designed for use during training to make models robust to KV cache eviction.

    Parameters
    ----------
    block_size : int
        Size of each block in tokens.
    drop_ratio : float
        Fraction of blocks to drop (0.0 = no dropout, 1.0 = drop all).
    protect_start : int
        Number of initial tokens (sink tokens) to always keep.
    protect_recent : int
        Number of recent tokens to always keep.
    training : bool
        If False, acts as pass-through (no dropout applied).
    """

    block_size: int = 64
    drop_ratio: float = 0.3
    protect_start: int = 4
    protect_recent: int = 64
    training: bool = True

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply block-wise dropout to KV cache."""
        if not self.training or self.drop_ratio == 0:
            return keys, values

        batch_size, num_heads, seq_len, head_dim = keys.shape

        # Generate block dropout mask
        mask = self._generate_block_mask(seq_len, keys.device)

        # Apply mask: keep only non-dropped positions
        kept_indices = mask.nonzero(as_tuple=True)[0]

        if len(kept_indices) == 0:
            return keys, values

        # Gather kept positions
        idx = kept_indices.view(1, 1, -1, 1).expand(batch_size, num_heads, -1, head_dim)
        keys = keys.gather(2, idx).contiguous()
        values = values.gather(2, idx).contiguous()

        return keys, values

    def _generate_block_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate a binary mask where 1 = keep, 0 = drop.

        The droppable region is [protect_start, seq_len - protect_recent).
        This region is split into blocks, and a fraction of blocks are dropped.
        """
        mask = torch.ones(seq_len, device=device, dtype=torch.bool)

        droppable_start = self.protect_start
        droppable_end = seq_len - self.protect_recent

        if droppable_end <= droppable_start:
            return mask

        droppable_len = droppable_end - droppable_start
        n_blocks = droppable_len // self.block_size

        if n_blocks <= 0:
            return mask

        n_drop = max(1, int(n_blocks * self.drop_ratio))
        n_drop = min(n_drop, n_blocks)

        drop_block_indices = random.sample(range(n_blocks), n_drop)

        for b in drop_block_indices:
            start = droppable_start + b * self.block_size
            end = min(start + self.block_size, droppable_end)
            mask[start:end] = False

        return mask

    def set_training(self, mode: bool = True) -> "BlockDropoutPress":
        """Toggle training mode."""
        self.training = mode
        return self
