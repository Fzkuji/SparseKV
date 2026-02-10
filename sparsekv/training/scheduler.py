"""
Module 3: Compression Scheduler

Determines how much to compress at each training step.
Strategy: start with mild compression, increase until performance degrades.

The scheduler monitors validation perplexity:
- If ppl doesn't increase → increase compression (lower keep_ratio)
- If ppl increases beyond threshold → hold current compression
- This auto-discovers the maximum safe compression ratio for each model
"""

import torch
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Configuration for compression scheduling."""
    initial_keep_ratio: float = 0.9       # Start with mild compression
    min_keep_ratio: float = 0.2           # Don't compress beyond this
    max_keep_ratio: float = 0.95          # Upper bound
    
    # Curriculum mode: linear decrease over training
    mode: str = "curriculum"              # "curriculum", "adaptive", "fixed"
    
    # Curriculum params
    warmup_fraction: float = 0.1          # Fraction of steps at initial ratio
    
    # Adaptive params
    ppl_threshold: float = 1.05           # Allow 5% ppl increase
    adjust_interval: int = 100            # Check every N steps
    step_size: float = 0.02               # How much to decrease keep_ratio per adjustment
    patience: int = 3                     # How many consecutive increases before stopping


class CompressionScheduler:
    """
    Manages the compression ratio throughout training.
    
    Three modes:
    - fixed: constant keep_ratio throughout training
    - curriculum: linear decrease from initial to min over training
    - adaptive: decrease keep_ratio when ppl is stable, stop when ppl degrades
    """
    
    def __init__(self, config: SchedulerConfig, total_steps: int):
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        self.current_keep_ratio = config.initial_keep_ratio
        
        # Adaptive mode state
        self._baseline_ppl: Optional[float] = None
        self._consecutive_increases = 0
        self._stopped = False
        self._best_keep_ratio = config.initial_keep_ratio
    
    def get_keep_ratio(self) -> float:
        """Get the current keep ratio."""
        return self.current_keep_ratio
    
    def step(self, val_ppl: Optional[float] = None):
        """
        Advance one step and update keep_ratio.
        
        Args:
            val_ppl: validation perplexity (required for adaptive mode)
        """
        self.current_step += 1
        
        if self.config.mode == "fixed":
            self.current_keep_ratio = self.config.initial_keep_ratio
            
        elif self.config.mode == "curriculum":
            self._curriculum_step()
            
        elif self.config.mode == "adaptive":
            if val_ppl is not None and self.current_step % self.config.adjust_interval == 0:
                self._adaptive_step(val_ppl)
            
        else:
            raise ValueError(f"Unknown scheduler mode: {self.config.mode}")
    
    def _curriculum_step(self):
        """Linear decrease from initial to min keep_ratio."""
        warmup_steps = int(self.total_steps * self.config.warmup_fraction)
        
        if self.current_step <= warmup_steps:
            # Warmup: stay at initial ratio
            self.current_keep_ratio = self.config.initial_keep_ratio
        else:
            # Linear decrease
            progress = (self.current_step - warmup_steps) / max(self.total_steps - warmup_steps, 1)
            progress = min(progress, 1.0)
            self.current_keep_ratio = (
                self.config.initial_keep_ratio 
                - progress * (self.config.initial_keep_ratio - self.config.min_keep_ratio)
            )
        
        self.current_keep_ratio = max(self.current_keep_ratio, self.config.min_keep_ratio)
    
    def _adaptive_step(self, val_ppl: float):
        """Decrease compression when ppl is stable, stop when degraded."""
        if self._stopped:
            return
        
        if self._baseline_ppl is None:
            # First measurement
            self._baseline_ppl = val_ppl
            logger.info(f"[Scheduler] Baseline ppl: {val_ppl:.4f}, keep_ratio: {self.current_keep_ratio:.3f}")
            return
        
        ppl_ratio = val_ppl / self._baseline_ppl
        
        if ppl_ratio <= self.config.ppl_threshold:
            # Performance is acceptable → increase compression
            self._consecutive_increases = 0
            self._best_keep_ratio = self.current_keep_ratio
            self.current_keep_ratio = max(
                self.current_keep_ratio - self.config.step_size,
                self.config.min_keep_ratio
            )
            self._baseline_ppl = val_ppl  # Update baseline
            logger.info(
                f"[Scheduler] ppl OK ({ppl_ratio:.3f}x), decreasing keep_ratio to {self.current_keep_ratio:.3f}"
            )
        else:
            # Performance degraded
            self._consecutive_increases += 1
            logger.info(
                f"[Scheduler] ppl increased ({ppl_ratio:.3f}x), patience {self._consecutive_increases}/{self.config.patience}"
            )
            
            if self._consecutive_increases >= self.config.patience:
                # Stop compressing, revert to best
                self.current_keep_ratio = self._best_keep_ratio
                self._stopped = True
                logger.info(
                    f"[Scheduler] Stopped. Final keep_ratio: {self.current_keep_ratio:.3f}"
                )
    
    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "current_keep_ratio": self.current_keep_ratio,
            "baseline_ppl": self._baseline_ppl,
            "consecutive_increases": self._consecutive_increases,
            "stopped": self._stopped,
            "best_keep_ratio": self._best_keep_ratio,
        }
    
    def load_state_dict(self, state: dict):
        self.current_step = state["current_step"]
        self.current_keep_ratio = state["current_keep_ratio"]
        self._baseline_ppl = state["baseline_ppl"]
        self._consecutive_increases = state["consecutive_increases"]
        self._stopped = state["stopped"]
        self._best_keep_ratio = state["best_keep_ratio"]
