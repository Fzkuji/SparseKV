"""
EIT (Eviction-Invariant Training) Trainer

Core training loop that combines the three modules:
1. Eviction Simulation (Module 1): determines which tokens to evict per layer
2. Invariance Loss (Module 2): measures discrepancy between full and evicted outputs
3. Compression Scheduler (Module 3): controls compression ratio over training

Training procedure:
1. Hook into each attention layer to capture:
   - Full attention output (O_full)
   - Attention weights (for eviction decisions)
2. After full forward pass, re-compute attention with evicted KV at each layer
3. Compute invariance loss + LM loss
4. Backward and update (LoRA) parameters
"""

import os
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from sparsekv.training.eviction_sim import AttentionScoreEviction, EvictionConfig
from sparsekv.training.loss import EvictionInvarianceLoss, LossConfig
from sparsekv.training.scheduler import CompressionScheduler, SchedulerConfig
from sparsekv.training.attention_hook import AttentionHook, compute_evicted_attention

logger = logging.getLogger(__name__)


@dataclass
class EITConfig:
    """Full configuration for EIT training."""
    # Model
    model_name: str = "Qwen/Qwen3-8B"
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 4096
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    
    # EIT modules
    eviction: EvictionConfig = field(default_factory=EvictionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_split: str = "train"
    dataset_subset: Optional[str] = "sample-10BT"
    num_train_samples: int = 10000
    
    # Logging & checkpointing
    output_dir: str = "./output"
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100
    
    # Attention capture
    capture_layers: Optional[list] = None  # None = all layers


class EITTrainer:
    """
    Main trainer for Eviction-Invariant Training.
    
    Usage:
        config = EITConfig(model_name="Qwen/Qwen3-8B")
        trainer = EITTrainer(config)
        trainer.train(train_dataloader, val_dataloader)
    """
    
    def __init__(self, config: EITConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self._setup_model()
        
        # Initialize EIT modules
        self.eviction_sim = AttentionScoreEviction(config.eviction)
        self.invariance_loss = EvictionInvarianceLoss(
            config.loss, self.num_layers
        ).to(self.device)
        
        # Scheduler will be initialized in train() when we know total steps
        self.compression_scheduler = None
    
    def _setup_model(self):
        """Load model and optionally apply LoRA."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        dtype = torch.bfloat16 if self.config.bf16 else (torch.float16 if self.config.fp16 else torch.float32)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            attn_implementation="eager",  # Need attention weights for capture
            device_map="auto",
        )
        
        # Apply LoRA BEFORE patching (so hooks see LoRA-modified projections)
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
        
        # Patch attention layers to capture Q, K, V
        self.hook = AttentionHook(enabled=True)
        self.hook.patch(self.model)
        self.num_layers = len(self.hook._get_layers(self.model))
    
    def _compute_eit_loss(
        self,
        keep_ratio: float,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute EIT invariance loss using captured Q, K, V from each layer.
        
        For each layer:
        1. Use captured attention weights to compute eviction mask (AdaKV-style)
        2. Recompute attention exactly with evicted KV using captured Q, K, V
        3. Compare full attention output vs evicted attention output
        
        Returns:
            eit_loss: scalar tensor
            metrics: dict with per-layer loss values
        """
        full_outputs = []
        evict_outputs = []
        
        for l_idx in range(self.num_layers):
            if l_idx not in self.hook.captures:
                continue
            
            cap = self.hook.captures[l_idx]
            
            # Determine eviction mask using attention weights
            # If attention weights not available, compute from Q, K
            if cap.attn_weights is not None:
                attn_for_eviction = cap.attn_weights  # (B, H_q, L, L)
                # For GQA: average across query groups to get per-KV-head weights
                if cap.num_key_value_groups > 1:
                    B, H_q, L, _ = attn_for_eviction.shape
                    H_kv = H_q // cap.num_key_value_groups
                    attn_for_eviction = attn_for_eviction.view(
                        B, H_kv, cap.num_key_value_groups, L, L
                    ).mean(dim=2)  # (B, H_kv, L, L)
            else:
                # Compute attention weights from Q, K
                scale = 1.0 / math.sqrt(cap.query.shape[-1])
                q_for_score = cap.query
                k_for_score = cap.key
                # Expand K for GQA
                if cap.num_key_value_groups > 1:
                    B, H_kv, L, D = k_for_score.shape
                    k_expanded = k_for_score.unsqueeze(2).expand(
                        -1, -1, cap.num_key_value_groups, -1, -1
                    ).reshape(B, -1, L, D)
                    scores = torch.matmul(q_for_score, k_expanded.transpose(-2, -1)) * scale
                    # Average back to H_kv
                    scores = scores.view(B, H_kv, cap.num_key_value_groups, L, L).mean(dim=2)
                else:
                    scores = torch.matmul(q_for_score, k_for_score.transpose(-2, -1)) * scale
                
                # Apply causal mask
                causal = torch.triu(torch.ones(L, L, device=scores.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn_for_eviction = F.softmax(scores, dim=-1)
            
            # Compute eviction mask: (B, H_kv, L)
            eviction_mask = self.eviction_sim(attn_for_eviction, keep_ratio=keep_ratio)
            
            # Recompute attention with evicted KV
            evict_attn_out = compute_evicted_attention(
                query=cap.query,
                key=cap.key,
                value=cap.value,
                eviction_mask=eviction_mask,
                num_kv_groups=cap.num_key_value_groups,
                causal=True,
            )  # (B, H_q, L, D)
            
            # Reshape evicted output to match full output shape (B, L, H_q * D)
            B, H_q, L, D = evict_attn_out.shape
            evict_out_flat = evict_attn_out.transpose(1, 2).reshape(B, L, H_q * D)
            
            # Full attention output from hook (B, L, hidden_dim) — after o_proj
            # We need pre-o_proj full output for fair comparison
            # Reshape captured Q,K,V full attention for comparison
            full_attn_out = compute_evicted_attention(
                query=cap.query,
                key=cap.key,
                value=cap.value,
                eviction_mask=torch.ones_like(eviction_mask),  # Keep all
                num_kv_groups=cap.num_key_value_groups,
                causal=True,
            )
            full_out_flat = full_attn_out.transpose(1, 2).reshape(B, L, H_q * D)
            
            full_outputs.append(full_out_flat)
            evict_outputs.append(evict_out_flat)
        
        if not full_outputs:
            return torch.tensor(0.0, device=self.device), {}
        
        return self.invariance_loss(full_outputs, evict_outputs)
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """
        Main training loop.
        """
        total_steps = (
            len(train_dataloader) * self.config.num_epochs 
            // self.config.gradient_accumulation_steps
        )
        
        # Initialize compression scheduler
        self.compression_scheduler = CompressionScheduler(
            self.config.scheduler, total_steps
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        
        # LR scheduler
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps,
        )
        
        logger.info(f"Starting EIT training: {total_steps} steps, {self.config.num_epochs} epochs")
        
        global_step = 0
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
                labels = input_ids.clone()
                
                # Clear captured data from previous step
                self.hook.clear()
                
                # Forward pass (patched attention captures Q, K, V per layer)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                lm_loss = outputs.loss
                
                # Compute EIT loss
                keep_ratio = self.compression_scheduler.get_keep_ratio()
                eit_loss, eit_metrics = self._compute_eit_loss(keep_ratio)
                
                # Total loss
                total_loss = lm_loss + eit_loss
                
                # Backward
                total_loss = total_loss / self.config.gradient_accumulation_steps
                total_loss.backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update compression scheduler
                    self.compression_scheduler.step()
                    global_step += 1
                    
                    # Logging
                    if global_step % self.config.log_interval == 0:
                        logger.info(
                            f"Step {global_step}/{total_steps} | "
                            f"LM Loss: {lm_loss.item():.4f} | "
                            f"EIT Loss: {eit_loss.item():.4f} | "
                            f"Keep Ratio: {keep_ratio:.3f} | "
                            f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                        )
                    
                    # Evaluation
                    if val_dataloader and global_step % self.config.eval_interval == 0:
                        val_ppl = self._evaluate(val_dataloader)
                        self.compression_scheduler.step(val_ppl)
                        logger.info(f"Step {global_step} | Val PPL: {val_ppl:.4f}")
                    
                    # Save checkpoint
                    if global_step % self.config.save_interval == 0:
                        self._save_checkpoint(global_step)
        
        # Final save
        self._save_checkpoint(global_step, final=True)
        logger.info("Training complete!")
    
    @torch.no_grad()
    def _evaluate(self, val_dataloader: DataLoader) -> float:
        """Compute validation perplexity."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
            labels = input_ids.clone()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            total_loss += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
        
        self.model.train()
        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)
    
    def _save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint."""
        save_dir = Path(self.config.output_dir) / (f"checkpoint-{step}" if not final else "final")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.use_lora:
            self.model.save_pretrained(save_dir)
        else:
            self.model.save_pretrained(save_dir)
        
        self.tokenizer.save_pretrained(save_dir)
        
        # Save scheduler state
        torch.save(
            self.compression_scheduler.state_dict(),
            save_dir / "scheduler_state.pt"
        )
        
        # Save config
        with open(save_dir / "eit_config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved to {save_dir}")
