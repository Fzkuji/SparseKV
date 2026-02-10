# SPDX-License-Identifier: Apache-2.0

"""
VariableBlockDropoutPress: Variable-length block dropout on KV cache.

Instead of fixed-size blocks, randomly generates blocks of varying sizes,
providing more diverse dropout patterns during training.
"""

import random
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress


@dataclass
class VariableBlockDropoutPress(BasePress):
    """
    Variable-length block-wise dropout on KV cache.

    Parameters
    ----------
    min_block_size : int
        Minimum block size in tokens.
    max_block_size : int
        Maximum block size in tokens.
    drop_ratio : float
        Fraction of blocks to drop.
    protect_start : int
        Number of initial (sink) tokens to protect.
    protect_recent : int
        Number of recent tokens to protect.
    training : bool
        If False, acts as pass-through.
    """

    min_block_size: int = 32
    max_block_size: int = 128
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
        if not self.training or self.drop_ratio == 0:
            return keys, values

        batch_size, num_heads, seq_len, head_dim = keys.shape
        mask = self._generate_variable_mask(seq_len, keys.device)

        kept_indices = mask.nonzero(as_tuple=True)[0]
        if len(kept_indices) == 0:
            return keys, values

        idx = kept_indices.view(1, 1, -1, 1).expand(batch_size, num_heads, -1, head_dim)
        keys = keys.gather(2, idx).contiguous()
        values = values.gather(2, idx).contiguous()

        return keys, values

    def _generate_variable_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate mask with variable-sized blocks."""
        mask = torch.ones(seq_len, device=device, dtype=torch.bool)

        droppable_start = self.protect_start
        droppable_end = seq_len - self.protect_recent
        if droppable_end <= droppable_start:
            return mask

        # Generate random-sized blocks
        blocks: list[tuple[int, int]] = []
        pos = droppable_start
        while pos < droppable_end:
            block_size = random.randint(self.min_block_size, self.max_block_size)
            end = min(pos + block_size, droppable_end)
            blocks.append((pos, end))
            pos = end

        if len(blocks) == 0:
            return mask

        n_drop = max(1, int(len(blocks) * self.drop_ratio))
        n_drop = min(n_drop, len(blocks))
        drop_indices = random.sample(range(len(blocks)), n_drop)

        for i in drop_indices:
            start, end = blocks[i]
            mask[start:end] = False

        return mask

    def set_training(self, mode: bool = True) -> "VariableBlockDropoutPress":
        self.training = mode
        return self
