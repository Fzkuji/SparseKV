# SPDX-License-Identifier: Apache-2.0

"""
AdaSparseTrainer: Custom HuggingFace Trainer with press integration.

Integrates KV cache dropout presses into the training loop, supports
curriculum learning, and handles multiple training objectives.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers import (
    DynamicCache,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from kvpress.presses.base_press import BasePress

from adasparse.presses.sparse_reg_press import SparseRegPress
from adasparse.training.curriculum import BaseCurriculum

logger = logging.getLogger(__name__)


@dataclass
class AdaSparseTrainingArguments(TrainingArguments):
    """Extended training arguments for AdaSparseKV."""

    # Press configuration
    press_type: str = "block_dropout"
    block_size: int = 64
    drop_ratio: float = 0.3
    protect_start: int = 4
    protect_recent: int = 64

    # Curriculum
    use_curriculum: bool = True
    curriculum_type: str = "linear"
    curriculum_start_ratio: float = 0.0
    curriculum_end_ratio: float = 0.5
    curriculum_warmup_steps: int = 1000

    # Objective
    objective_type: str = "standard_lm"
    sparse_reg_type: str = "entropy"
    sparse_reg_weight: float = 0.01
    recon_weight: float = 0.1

    # Mixed objective weights
    lm_weight: float = 1.0
    sparse_lm_weight: float = 0.5

    # Attention output (needed for adaptive press and regularization)
    output_attentions: bool = False


class AdaSparseTrainer(Trainer):
    """
    Custom Trainer that integrates KV cache dropout presses.

    During each forward pass, the press's forward hook automatically applies
    dropout to the KV cache. The trainer handles:
    - Press lifecycle (register/remove hooks)
    - Curriculum learning (adjusting dropout ratio over time)
    - Multiple training objectives
    - Logging press-specific metrics
    """

    def __init__(
        self,
        press: BasePress,
        curriculum: BaseCurriculum | None = None,
        reg_press: SparseRegPress | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.press = press
        self.curriculum = curriculum
        self.reg_press = reg_press
        self._press_hooks: list = []

    def _update_curriculum(self):
        """Update dropout ratio based on curriculum schedule."""
        if self.curriculum is None:
            return

        current_step = self.state.global_step
        total_steps = self.state.max_steps or self.args.max_steps

        new_ratio = self.curriculum.get_ratio(current_step, total_steps)

        # Update press dropout ratio
        if hasattr(self.press, "drop_ratio"):
            self.press.drop_ratio = new_ratio
        elif hasattr(self.press, "base_drop_ratio"):
            self.press.base_drop_ratio = new_ratio

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Compute loss with press-applied KV cache dropout.

        The press context manager registers forward hooks on all attention layers.
        During the forward pass, these hooks automatically apply dropout to the KV cache.
        """
        # Update curriculum
        self._update_curriculum()

        # Prepare inputs
        labels = inputs.pop("labels", None)
        if labels is None:
            labels = inputs["input_ids"].clone()

        objective_type = getattr(self.args, "objective_type", "standard_lm") or "standard_lm"

        if objective_type == "reconstruction":
            loss, outputs = self._compute_reconstruction_loss(model, inputs, labels)
        else:
            # Forward with press (hooks handle the dropout)
            with self.press(model):
                outputs = model(
                    **inputs,
                    use_cache=True,
                    past_key_values=DynamicCache(),
                    output_attentions=getattr(self.args, "output_attentions", False),
                )
            loss = self._compute_objective_loss(outputs.logits, labels)

        if return_outputs:
            return loss, outputs
        return loss

    def _compute_reconstruction_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ):
        """
        Reconstruction objective:
        1. Forward WITHOUT dropout → get full hidden states (detached, no grad)
        2. Forward WITH dropout → get sparse hidden states
        3. Loss = LM loss + MSE(sparse_hidden, full_hidden)
        """
        import torch.nn.functional as F

        # Step 1: full forward (no dropout, no grad)
        with torch.no_grad():
            full_outputs = model(**inputs, output_hidden_states=True)
            full_hidden = full_outputs.hidden_states[-1].detach()  # last layer

        # Step 2: forward with dropout
        with self.press(model):
            sparse_outputs = model(
                **inputs,
                use_cache=True,
                past_key_values=DynamicCache(),
                output_hidden_states=True,
            )

        sparse_hidden = sparse_outputs.hidden_states[-1]
        logits = sparse_outputs.logits

        # LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Reconstruction loss
        recon_weight = getattr(self.args, "recon_weight", 0.1) or 0.1
        recon_loss = F.mse_loss(sparse_hidden, full_hidden)

        total_loss = lm_loss + recon_weight * recon_loss

        return total_loss, sparse_outputs

    def _compute_objective_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss based on the configured objective."""
        import torch.nn.functional as F

        # Standard next-token prediction loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        total_loss = lm_loss

        # Add sparsity regularization if configured
        if self.reg_press is not None:
            reg_loss = self.reg_press.get_reg_loss()
            if reg_loss.device != lm_loss.device:
                reg_loss = reg_loss.to(lm_loss.device)
            total_loss = total_loss + reg_loss

        return total_loss

    def log(self, logs: dict[str, float], **kwargs):
        """Add press-specific metrics to logs."""
        # Log current dropout ratio
        if hasattr(self.press, "drop_ratio"):
            logs["press/drop_ratio"] = self.press.drop_ratio
        elif hasattr(self.press, "base_drop_ratio"):
            logs["press/drop_ratio"] = self.press.base_drop_ratio

        if self.curriculum:
            logs["press/curriculum_step"] = self.state.global_step

        super().log(logs, **kwargs)
