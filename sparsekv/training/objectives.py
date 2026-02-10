# SPDX-License-Identifier: Apache-2.0

"""
Training objectives for SparseKV.

Supports multiple training signals:
1. StandardLM: next-token prediction with dropout press (cache robustness)
2. SparseLM: LM loss + sparsity regularization (attention concentration)
3. Reconstruction: reconstruct dropped content from sparse cache
4. Mixed: configurable combination of all above
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from sparsekv.presses.sparse_reg_press import SparseRegPress


@dataclass
class StandardLMObjective:
    """
    Standard language modeling objective with KV cache dropout.

    The press handles the dropout — this objective simply computes cross-entropy
    on the model output. The model learns to predict correctly despite missing cache.
    """

    def compute(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute cross-entropy loss."""
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss


@dataclass
class SparseLMObjective:
    """
    Language modeling + sparsity regularization.

    Combines standard LM loss with entropy/L1 regularization on attention
    to encourage sparser attention patterns.
    """

    reg_press: SparseRegPress | None = None
    reg_weight: float = 0.01

    def compute(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute LM loss + sparsity regularization."""
        # Standard LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Sparsity regularization
        reg_loss = torch.tensor(0.0, device=lm_loss.device)
        if self.reg_press is not None:
            reg_loss = self.reg_press.get_reg_loss()
            if reg_loss.device != lm_loss.device:
                reg_loss = reg_loss.to(lm_loss.device)

        return lm_loss + self.reg_weight * reg_loss


@dataclass
class ReconstructionObjective:
    """
    Reconstruction objective: predict the dropped content.

    After KV cache dropout, the model should be able to reconstruct or
    summarize the dropped content from the remaining context.

    This is an auxiliary loss that teaches the model to maintain
    representations of the full context even with sparse cache.
    """

    recon_weight: float = 0.1

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        full_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute LM loss + reconstruction loss."""
        # Standard LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Reconstruction loss (MSE between sparse and full hidden states)
        recon_loss = torch.tensor(0.0, device=lm_loss.device)
        if hidden_states is not None and full_hidden_states is not None:
            # Align lengths if needed
            min_len = min(hidden_states.shape[1], full_hidden_states.shape[1])
            recon_loss = F.mse_loss(
                hidden_states[:, :min_len],
                full_hidden_states[:, :min_len].detach(),
            )

        return lm_loss + self.recon_weight * recon_loss


@dataclass
class MixedObjective:
    """
    Configurable combination of multiple training objectives.

    Parameters
    ----------
    lm_weight : float
        Weight for standard LM loss (with dropout).
    sparse_lm_weight : float
        Weight for sparse LM loss (LM + sparsity regularization).
    recon_weight : float
        Weight for reconstruction loss.
    reg_press : SparseRegPress, optional
        Press for computing sparsity regularization.
    """

    lm_weight: float = 1.0
    sparse_lm_weight: float = 0.5
    recon_weight: float = 0.1
    reg_press: SparseRegPress | None = None

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        full_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute mixed objective."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Base LM loss
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        total_loss = self.lm_weight * lm_loss

        # Sparsity regularization
        if self.sparse_lm_weight > 0 and self.reg_press is not None:
            reg_loss = self.reg_press.get_reg_loss()
            if reg_loss.device != lm_loss.device:
                reg_loss = reg_loss.to(lm_loss.device)
            total_loss = total_loss + self.sparse_lm_weight * reg_loss

        # Reconstruction loss
        if self.recon_weight > 0 and hidden_states is not None and full_hidden_states is not None:
            min_len = min(hidden_states.shape[1], full_hidden_states.shape[1])
            recon_loss = F.mse_loss(
                hidden_states[:, :min_len],
                full_hidden_states[:, :min_len].detach(),
            )
            total_loss = total_loss + self.recon_weight * recon_loss

        return total_loss
