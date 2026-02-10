# SPDX-License-Identifier: Apache-2.0

"""
SparseRegPress: Attention sparsity regularization during training.

Adds entropy or L1 regularization on attention weights to encourage sparser patterns.
Does NOT prune the KV cache — instead, stores a regularization loss that should be
added to the training objective.
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.base_press import BasePress


@dataclass
class SparseRegPress(BasePress):
    """
    Sparsity regularization press.

    Does not modify the KV cache. Instead, computes and stores a regularization loss
    from the attention weights that should be added to the training loss.

    Parameters
    ----------
    reg_type : str
        Type of regularization: "entropy" (minimize entropy → concentrate attention)
        or "l1" (L1 penalty on attention weights → sparse attention).
    reg_weight : float
        Weight of the regularization loss.
    training : bool
        If False, acts as pass-through (no regularization computed).
    """

    reg_type: str = "entropy"
    reg_weight: float = 0.01
    training: bool = True
    _reg_losses: list = field(default_factory=list, repr=False)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute regularization loss but don't modify KV cache."""
        if self.training and attentions is not None:
            reg_loss = self._compute_reg_loss(attentions)
            self._reg_losses.append(reg_loss)

        # Pass through — no KV modification
        return keys, values

    def _compute_reg_loss(self, attentions: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity regularization loss from attention weights.

        Parameters
        ----------
        attentions : torch.Tensor
            Attention weights (batch, num_heads, q_len, kv_len).
        """
        if self.reg_type == "entropy":
            # Minimize entropy → attention becomes more concentrated
            eps = 1e-8
            entropy = -(attentions * (attentions + eps).log()).sum(dim=-1)  # (B, H, Q)
            return self.reg_weight * entropy.mean()

        elif self.reg_type == "l1":
            # L1 penalty → encourage zeros in attention
            return self.reg_weight * attentions.abs().mean()

        else:
            raise ValueError(f"Unknown reg_type: {self.reg_type}. Use 'entropy' or 'l1'.")

    def get_reg_loss(self) -> torch.Tensor:
        """Get accumulated regularization loss and reset."""
        if not self._reg_losses:
            return torch.tensor(0.0)
        total = torch.stack(self._reg_losses).sum()
        self._reg_losses.clear()
        return total

    def reset(self):
        """Clear accumulated losses."""
        self._reg_losses.clear()

    def set_training(self, mode: bool = True) -> "SparseRegPress":
        self.training = mode
        return self
