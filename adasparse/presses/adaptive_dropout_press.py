# SPDX-License-Identifier: Apache-2.0

"""
AdaptiveBlockDropoutPress: Attention-density-aware adaptive block dropout.

The key innovation: blocks with lower internal attention density (i.e., tokens that
mostly attend to sink tokens rather than each other) are more likely to be dropped.
This mimics what happens during real eviction — unimportant context is removed first.
"""

import random
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress


@dataclass
class AdaptiveBlockDropoutPress(BasePress):
    """
    Adaptive block-wise dropout based on attention density and sink-awareness.

    Each block's drop probability is determined by:
    1. Intra-block attention density: low density → high drop prob
    2. Sink attention ratio: high sink ratio → high drop prob (content is less important)

    Parameters
    ----------
    block_size : int
        Size of each block in tokens.
    base_drop_ratio : float
        Base fraction of blocks to drop.
    protect_start : int
        Number of initial tokens (sink tokens) to always keep.
    protect_recent : int
        Number of recent tokens to always keep.
    sink_weight : float
        Weight for sink attention ratio in drop probability computation.
    temperature : float
        Temperature for converting density to drop probability (higher = sharper).
    training : bool
        If False, acts as pass-through.
    """

    block_size: int = 64
    base_drop_ratio: float = 0.3
    protect_start: int = 4
    protect_recent: int = 64
    sink_weight: float = 0.3
    temperature: float = 1.0
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
        """Apply adaptive block dropout based on attention patterns."""
        if not self.training or self.base_drop_ratio == 0:
            return keys, values

        batch_size, num_heads, seq_len, head_dim = keys.shape

        # If attention weights are available, use adaptive dropout
        if attentions is not None:
            mask = self._adaptive_block_mask(attentions, seq_len, keys.device)
        else:
            # Fallback to uniform random dropout
            mask = self._uniform_block_mask(seq_len, keys.device)

        kept_indices = mask.nonzero(as_tuple=True)[0]
        if len(kept_indices) == 0:
            return keys, values

        idx = kept_indices.view(1, 1, -1, 1).expand(batch_size, num_heads, -1, head_dim)
        keys = keys.gather(2, idx).contiguous()
        values = values.gather(2, idx).contiguous()

        return keys, values

    def _compute_block_info(
        self,
        attentions: torch.Tensor,
        blocks: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-block attention density and sink ratio.

        Parameters
        ----------
        attentions : torch.Tensor
            Attention weights (batch, num_heads, seq_len, seq_len).
        blocks : list of (start, end) tuples
            Block boundaries.

        Returns
        -------
        densities : torch.Tensor of shape (n_blocks,)
            Intra-block attention density for each block.
        sink_ratios : torch.Tensor of shape (n_blocks,)
            Fraction of attention mass going to sink tokens.
        """
        # Average over batch and heads for block-level decisions
        attn = attentions.mean(dim=(0, 1))  # (seq_len, seq_len)

        densities = []
        sink_ratios = []

        for start, end in blocks:
            block_len = end - start
            if block_len == 0:
                densities.append(0.0)
                sink_ratios.append(1.0)
                continue

            # Intra-block attention: how much do tokens in this block attend to each other
            intra_attn = attn[start:end, start:end].sum().item() / block_len
            densities.append(intra_attn)

            # Sink ratio: fraction of attention going to sink tokens
            total_attn = attn[start:end, :].sum().item()
            if total_attn > 0:
                sink_attn = attn[start:end, : self.protect_start].sum().item()
                sink_ratios.append(sink_attn / total_attn)
            else:
                sink_ratios.append(0.0)

        return (
            torch.tensor(densities, dtype=torch.float32),
            torch.tensor(sink_ratios, dtype=torch.float32),
        )

    def _adaptive_block_mask(
        self,
        attentions: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate mask with attention-density-aware drop probabilities."""
        mask = torch.ones(seq_len, device=device, dtype=torch.bool)

        droppable_start = self.protect_start
        droppable_end = seq_len - self.protect_recent
        if droppable_end <= droppable_start:
            return mask

        # Build block boundaries
        blocks = []
        pos = droppable_start
        while pos < droppable_end:
            end = min(pos + self.block_size, droppable_end)
            blocks.append((pos, end))
            pos = end

        if len(blocks) == 0:
            return mask

        # Compute per-block statistics
        densities, sink_ratios = self._compute_block_info(attentions, blocks)

        # Drop probability: low density → high drop prob, high sink ratio → high drop prob
        # Normalize densities to [0, 1]
        if densities.max() > densities.min():
            norm_density = (densities - densities.min()) / (densities.max() - densities.min())
        else:
            norm_density = torch.zeros_like(densities)

        # Base drop probability: inverse of normalized density
        drop_probs = (1 - norm_density) * self.temperature

        # Sink-aware adjustment
        drop_probs = drop_probs + self.sink_weight * sink_ratios

        # Normalize to achieve target drop ratio
        # Scale so that expected number of dropped blocks matches base_drop_ratio
        target_n_drop = int(len(blocks) * self.base_drop_ratio)
        if target_n_drop > 0 and drop_probs.sum() > 0:
            drop_probs = drop_probs / drop_probs.sum() * target_n_drop
            drop_probs = drop_probs.clamp(0, 1)

        # Sample drops
        for i, (start, end) in enumerate(blocks):
            if random.random() < drop_probs[i].item():
                mask[start:end] = False

        return mask

    def _uniform_block_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Fallback: uniform random block dropout."""
        mask = torch.ones(seq_len, device=device, dtype=torch.bool)

        droppable_start = self.protect_start
        droppable_end = seq_len - self.protect_recent
        if droppable_end <= droppable_start:
            return mask

        n_blocks = (droppable_end - droppable_start) // self.block_size
        if n_blocks <= 0:
            return mask

        n_drop = max(1, int(n_blocks * self.base_drop_ratio))
        drop_indices = random.sample(range(n_blocks), min(n_drop, n_blocks))

        for b in drop_indices:
            start = droppable_start + b * self.block_size
            end = min(start + self.block_size, droppable_end)
            mask[start:end] = False

        return mask

    def set_training(self, mode: bool = True) -> "AdaptiveBlockDropoutPress":
        """Toggle training mode."""
        self.training = mode
        return self
