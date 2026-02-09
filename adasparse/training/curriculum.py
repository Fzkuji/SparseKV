# SPDX-License-Identifier: Apache-2.0

"""
Curriculum learning schedules for gradually increasing dropout intensity.

Starting with no dropout and slowly increasing teaches the model progressively,
avoiding catastrophic forgetting while building robustness to KV cache eviction.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseCurriculum(ABC):
    """Base class for curriculum learning schedules."""

    start_ratio: float = 0.0
    end_ratio: float = 0.5
    warmup_steps: int = 1000

    @abstractmethod
    def get_ratio(self, step: int, total_steps: int) -> float:
        """Get the dropout ratio for the current step."""
        ...


@dataclass
class LinearCurriculum(BaseCurriculum):
    """Linearly increase dropout ratio from start to end."""

    def get_ratio(self, step: int, total_steps: int) -> float:
        if step >= total_steps:
            return self.end_ratio
        if step < self.warmup_steps:
            return self.start_ratio

        progress = (step - self.warmup_steps) / max(total_steps - self.warmup_steps, 1)
        return self.start_ratio + (self.end_ratio - self.start_ratio) * progress


@dataclass
class StepCurriculum(BaseCurriculum):
    """Step-wise increase in dropout ratio."""

    n_stages: int = 5

    def get_ratio(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return self.start_ratio

        progress = (step - self.warmup_steps) / max(total_steps - self.warmup_steps, 1)
        stage = min(int(progress * self.n_stages), self.n_stages - 1)
        stage_ratio = stage / (self.n_stages - 1)
        return self.start_ratio + (self.end_ratio - self.start_ratio) * stage_ratio


@dataclass
class CosineCurriculum(BaseCurriculum):
    """Cosine schedule for dropout ratio increase."""

    def get_ratio(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return self.start_ratio
        if step >= total_steps:
            return self.end_ratio

        progress = (step - self.warmup_steps) / max(total_steps - self.warmup_steps, 1)
        # Cosine from 0 to 1
        cosine_progress = 0.5 * (1 - math.cos(math.pi * progress))
        return self.start_ratio + (self.end_ratio - self.start_ratio) * cosine_progress
