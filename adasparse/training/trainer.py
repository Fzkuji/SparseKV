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

        # Forward with press (hooks handle the dropout)
        with self.press(model):
            # Need use_cache=True for the press hooks to have KV cache to modify
            outputs = model(
                **inputs,
                use_cache=True,
                past_key_values=DynamicCache(),
                output_attentions=getattr(self.args, "output_attentions", False),
            )

        logits = outputs.logits

        # Compute loss based on objective type
        loss = self._compute_objective_loss(logits, labels)

        if return_outputs:
            return loss, outputs
        return loss

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
