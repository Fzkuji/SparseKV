"""
SparseKV Trainer — Eviction-Invariant Training

Training loop:
1. Full forward (no_grad, flash attention) → teacher logits
2. Create KV mask: anchors always kept + random sample of non-anchors
3. Evicted forward (with grad, SDPA + 4D mask) → student logits
4. Loss = CE(student, labels) + λ * KL(teacher, student)
5. Curriculum: gradually increase dropout ratio during training

The model learns to produce correct outputs even when most non-anchor
KV cache entries are missing, effectively learning to concentrate
information into anchor tokens.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from sparsekv.training.anchor import AnchorSelector, AnchorConfig
from sparsekv.training.kv_dropout import create_kv_dropout_mask, keep_mask_to_4d_attention_mask
from sparsekv.training.scheduler import CompressionScheduler, SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Full training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-8B"
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 4096
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    bf16: bool = True
    
    # Loss
    lambda_kl: float = 1.0               # Weight for KL divergence loss
    kl_temperature: float = 1.0          # Temperature for KL computation
    
    # Anchor
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    
    # Compression scheduler
    scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig(
        initial_keep_ratio=0.9,
        min_keep_ratio=0.3,
        mode="curriculum",
    ))
    
    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: Optional[str] = "sample-10BT"
    num_train_samples: int = 10000
    num_val_samples: int = 500
    
    # Logging & saving
    output_dir: str = "./output"
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100


class SparseKVTrainer:
    """
    Main trainer for SparseKV.
    
    Usage:
        config = TrainConfig(model_name="Qwen/Qwen3-8B")
        trainer = SparseKVTrainer(config)
        trainer.train(train_loader, val_loader)
    """
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model + tokenizer
        self._setup_model()
        
        # Anchor selector
        self.anchor_selector = AnchorSelector(config.anchor, self.tokenizer)
        logger.info(f"Anchor config: {self.anchor_selector.describe()}")
        
        # Scheduler initialized in train()
        self.scheduler: Optional[CompressionScheduler] = None
    
    def _setup_model(self):
        """Load model, apply LoRA."""
        logger.info(f"Loading model: {self.config.model_name}")
        dtype = torch.bfloat16 if self.config.bf16 else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use SDPA for evicted forward (supports 4D attention mask)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map="auto",
        )
        
        if self.config.use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=0.05,
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def _compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        keep_ratio: float,
    ) -> tuple[torch.Tensor, dict]:
        """
        Core loss computation:
        1. Full forward → teacher logits
        2. Evicted forward → student logits
        3. Loss = CE + λ * KL
        
        Returns:
            total_loss: scalar
            metrics: dict with loss components
        """
        B, L = input_ids.shape
        
        # ---- Step 1: Full forward (teacher, no grad) ----
        with torch.no_grad():
            full_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = full_outputs.logits  # (B, L, V)
        
        # ---- Step 2: Create KV dropout mask ----
        anchor_mask = self.anchor_selector.get_anchor_mask(input_ids)  # (B, L)
        keep_mask = create_kv_dropout_mask(anchor_mask, keep_ratio, L)  # (B, L)
        
        # Build 4D attention mask for SDPA
        attn_mask_4d = keep_mask_to_4d_attention_mask(
            keep_mask, L,
            dtype=teacher_logits.dtype,
        )  # (B, 1, L, L)
        
        # ---- Step 3: Evicted forward (student, with grad) ----
        evict_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask_4d,
            labels=labels,
        )
        
        # CE loss from evicted forward
        ce_loss = evict_outputs.loss
        student_logits = evict_outputs.logits  # (B, L, V)
        
        # ---- Step 4: KL divergence loss ----
        T = self.config.kl_temperature
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (T * T)  # Scale by T² for gradient magnitude
        
        # ---- Total loss ----
        total_loss = ce_loss + self.config.lambda_kl * kl_loss
        
        # Metrics
        anchor_ratio = anchor_mask.float().mean().item()
        kept_ratio = keep_mask.float().mean().item()
        
        metrics = {
            "ce_loss": ce_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "keep_ratio": keep_ratio,
            "actual_kept": kept_ratio,
            "anchor_ratio": anchor_ratio,
        }
        
        return total_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop."""
        total_steps = (
            len(train_loader) * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )
        
        # Initialize scheduler
        self.scheduler = CompressionScheduler(self.config.scheduler, total_steps)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps,
        )
        
        logger.info(f"Training: {total_steps} steps, {self.config.num_epochs} epoch(s)")
        logger.info(f"Compression: {self.config.scheduler.initial_keep_ratio} → {self.config.scheduler.min_keep_ratio}")
        
        global_step = 0
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get(
                    "attention_mask", torch.ones_like(input_ids)
                ).to(self.device)
                labels = input_ids.clone()
                
                # Get current keep ratio from scheduler
                keep_ratio = self.scheduler.get_keep_ratio()
                
                # Compute loss
                loss, metrics = self._compute_loss(
                    input_ids, attention_mask, labels, keep_ratio
                )
                
                # Backward
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    self.scheduler.step()
                    global_step += 1
                    
                    # Log
                    if global_step % self.config.log_interval == 0:
                        lr = lr_scheduler.get_last_lr()[0]
                        logger.info(
                            f"Step {global_step}/{total_steps} | "
                            f"CE: {metrics['ce_loss']:.4f} | "
                            f"KL: {metrics['kl_loss']:.4f} | "
                            f"Total: {metrics['total_loss']:.4f} | "
                            f"Keep: {metrics['keep_ratio']:.3f} | "
                            f"Anchor: {metrics['anchor_ratio']:.3f} | "
                            f"LR: {lr:.2e}"
                        )
                    
                    # Evaluate
                    if val_loader and global_step % self.config.eval_interval == 0:
                        val_ppl = self._evaluate(val_loader)
                        logger.info(f"Step {global_step} | Val PPL: {val_ppl:.2f}")
                        if self.config.scheduler.mode == "adaptive":
                            self.scheduler.step(val_ppl)
                    
                    # Save
                    if global_step % self.config.save_interval == 0:
                        self._save(global_step)
        
        # Final save
        self._save(global_step, final=True)
        logger.info("Training complete!")
    
    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> float:
        """Compute validation perplexity (full model, no eviction)."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get(
                "attention_mask", torch.ones_like(input_ids)
            ).to(self.device)
            labels = input_ids.clone()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            num_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
        
        self.model.train()
        return math.exp(total_loss / max(total_tokens, 1))
    
    def _save(self, step: int, final: bool = False):
        """Save checkpoint."""
        save_dir = Path(self.config.output_dir) / ("final" if final else f"checkpoint-{step}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model (LoRA adapter or full)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save training state
        state = {
            "step": step,
            "scheduler": self.scheduler.state_dict() if self.scheduler else {},
            "config": {
                k: v for k, v in self.config.__dict__.items()
                if not isinstance(v, (AnchorConfig, SchedulerConfig))
            },
            "anchor_config": self.config.anchor.__dict__,
            "scheduler_config": self.config.scheduler.__dict__,
        }
        torch.save(state, save_dir / "training_state.pt")
        
        logger.info(f"Saved to {save_dir}")
