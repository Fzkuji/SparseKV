# SPDX-License-Identifier: Apache-2.0

"""
EvictionAugPress: Use existing eviction methods as data augmentation during training.

Wraps any kvpress ScorerPress to apply KV cache compression during training.
No gradient flows through the eviction selection — the model simply learns to be
robust to having parts of its cache removed by standard eviction methods.

Can randomly switch between different eviction methods each step for diversity.
"""

import random
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class EvictionAugPress(BasePress):
    """
    Data-augmentation press that applies existing eviction methods during training.

    Parameters
    ----------
    base_presses : list[ScorerPress]
        List of kvpress ScorerPress instances to randomly sample from.
        Example: [SnapKVPress(compression_ratio=0.5), KnormPress(compression_ratio=0.5)]
    compression_ratio : float
        Compression ratio applied by the selected press.
    random_select : bool
        If True, randomly select a press each call. If False, use the first press.
    training : bool
        If False, acts as pass-through.
    """

    base_presses: list = field(default_factory=list)
    compression_ratio: float = 0.5
    random_select: bool = True
    training: bool = True

    def __post_init__(self):
        # Set compression ratio on all base presses
        for press in self.base_presses:
            if isinstance(press, ScorerPress):
                press.compression_ratio = self.compression_ratio

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or not self.base_presses:
            return keys, values

        # Select a press
        if self.random_select and len(self.base_presses) > 1:
            press = random.choice(self.base_presses)
        else:
            press = self.base_presses[0]

        # Apply eviction without gradient through the selection
        with torch.no_grad():
            keys_compressed, values_compressed = press.compress(
                module, hidden_states, keys, values, attentions, kwargs
            )

        return keys_compressed.detach(), values_compressed.detach()

    def set_training(self, mode: bool = True) -> "EvictionAugPress":
        self.training = mode
        return self
