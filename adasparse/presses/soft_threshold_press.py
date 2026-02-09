# SPDX-License-Identifier: Apache-2.0

"""
SoftThresholdPress: Sigmoid soft masking on attention weights.

Uses a differentiable sigmoid mask to suppress low-attention KV pairs,
encouraging the model to concentrate attention on fewer tokens.
The temperature can be annealed during training (soft → hard).
"""

from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress


@dataclass
class SoftThresholdPress(BasePress):
    """
    Sigmoid-based soft threshold masking on attention weights.

    Instead of hard top-k selection, applies mask = sigmoid(T * (attn - threshold))
    to attention weights. Fully differentiable, allowing gradients to flow through.

    At inference time, can switch to hard threshold for actual KV eviction.

    Parameters
    ----------
    threshold : float
        Attention threshold below which values are suppressed.
    temperature : float
        Sigmoid temperature (higher = sharper, more like hard threshold).
    compression_ratio : float
        Target compression ratio for hard-threshold mode (inference).
    training : bool
        If True, uses soft sigmoid mask. If False, uses hard threshold + KV pruning.
    """

    threshold: float = 0.01
    temperature: float = 10.0
    compression_ratio: float = 0.0
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
        if attentions is None:
            return keys, values

        if self.training:
            return self._soft_threshold(module, keys, values, attentions)
        else:
            return self._hard_threshold(module, keys, values, attentions)

    def _soft_threshold(
        self,
        module: nn.Module,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply soft sigmoid mask. This doesn't prune KV pairs but reweights them.

        Note: This modifies the effective attention pattern by scaling values.
        The actual KV cache size stays the same during training, but the model
        learns to concentrate attention because suppressed entries provide
        near-zero signal.
        """
        # attentions: (batch, num_heads, q_len, kv_len)
        # Compute per-KV importance: average attention received across queries and heads
        importance = attentions.mean(dim=(0, 2))  # (num_heads, kv_len)
        importance = importance.mean(dim=0)  # (kv_len,)

        # Soft mask
        mask = torch.sigmoid(self.temperature * (importance - self.threshold))
        # mask shape: (kv_len,)

        # Apply mask to values (soft pruning)
        # Reshape for broadcasting: (1, 1, kv_len, 1)
        mask = mask.view(1, 1, -1, 1)
        values = values * mask
        # Keys stay unchanged — they determine attention, values carry the signal

        return keys, values

    def _hard_threshold(
        self,
        module: nn.Module,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hard threshold + actual KV pruning for inference."""
        if self.compression_ratio == 0:
            return keys, values

        batch_size, num_heads, seq_len, head_dim = keys.shape

        # Compute importance
        importance = attentions.mean(dim=(0, 2))  # (num_heads, kv_len)
        importance = importance.mean(dim=0)  # (kv_len,)

        # Keep top tokens
        n_kept = int(seq_len * (1 - self.compression_ratio))
        kept_indices = importance.topk(n_kept).indices.sort().values

        idx = kept_indices.view(1, 1, -1, 1).expand(batch_size, num_heads, -1, head_dim)
        keys = keys.gather(2, idx).contiguous()
        values = values.gather(2, idx).contiguous()

        return keys, values

    def set_training(self, mode: bool = True) -> "SoftThresholdPress":
        self.training = mode
        return self

    def anneal_temperature(self, step: int, total_steps: int, start_temp: float = 1.0, end_temp: float = 50.0):
        """Anneal temperature from soft to hard over training."""
        progress = min(step / max(total_steps, 1), 1.0)
        self.temperature = start_temp + (end_temp - start_temp) * progress
