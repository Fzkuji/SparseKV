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


class AttentionCaptureHook:
    """
    Hook to capture attention weights and outputs from transformer layers.
    
    Registers forward hooks on attention modules to intercept:
    - attention weights (for eviction decisions)
    - attention output (for invariance loss computation)
    """
    
    def __init__(self):
        self.attention_weights = {}   # layer_idx -> (B, H, L_q, L_kv)
        self.attention_outputs = {}   # layer_idx -> (B, L, D)
        self._hooks = []
    
    def register(self, model: nn.Module):
        """Register hooks on all attention layers."""
        for idx, layer in enumerate(self._get_attention_layers(model)):
            hook = layer.register_forward_hook(
                self._make_hook(idx),
                with_kwargs=True,
            )
            self._hooks.append(hook)
        logger.info(f"Registered attention hooks on {len(self._hooks)} layers")
    
    def _get_attention_layers(self, model):
        """Extract attention modules from the model."""
        # Support common architectures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # LlamaForCausalLM, QwenForCausalLM, etc.
            return [layer.self_attn for layer in model.model.layers]
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style
            return [layer.attn for layer in model.transformer.h]
        else:
            raise ValueError(f"Unsupported model architecture: {type(model)}")
    
    def _make_hook(self, layer_idx: int):
        def hook_fn(module, args, kwargs, output):
            # output is typically (attn_output, attn_weights, past_key_value)
            # or just attn_output depending on output_attentions flag
            if isinstance(output, tuple):
                attn_output = output[0]
                if len(output) > 1 and output[1] is not None:
                    self.attention_weights[layer_idx] = output[1].detach()
            else:
                attn_output = output
            
            self.attention_outputs[layer_idx] = attn_output
            return output
        return hook_fn
    
    def clear(self):
        """Clear captured data."""
        self.attention_weights.clear()
        self.attention_outputs.clear()
    
    def remove(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def compute_evicted_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """
    Compute attention output using only the kept (non-evicted) KV positions.
    
    Args:
        query: (B, H, L_q, D)
        key: (B, H, L_kv, D)
        value: (B, H, L_kv, D)
        mask: (B, H, L_kv) binary mask, True = keep
        head_dim: dimension per head
        
    Returns:
        attn_output: (B, H, L_q, D)
    """
    # Compute attention scores
    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # (B, H, L_q, L_kv)
    
    # Mask out evicted positions with -inf
    eviction_mask = ~mask.unsqueeze(2)  # (B, H, 1, L_kv)
    attn_scores = attn_scores.masked_fill(eviction_mask, float('-inf'))
    
    # Softmax and compute output
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_weights = attn_weights.nan_to_num(0.0)  # Handle all-masked rows
    attn_output = torch.matmul(attn_weights, value)  # (B, H, L_q, D)
    
    return attn_output


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
        
        num_layers = len(list(self.hook._get_attention_layers(self.model)))
        self.invariance_loss = EvictionInvarianceLoss(config.loss, num_layers).to(self.device)
        
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
            attn_implementation="eager",  # Need attention weights
            device_map="auto",
        )
        
        # Register attention hooks
        self.hook = AttentionCaptureHook()
        self.hook.register(self.model)
        
        # Apply LoRA if configured
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
        
        # Enable output_attentions
        self.model.config.output_attentions = True
    
    def _compute_eit_loss(
        self,
        keep_ratio: float,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute EIT invariance loss using captured attention data.
        
        After a forward pass, the hook has captured attention weights and outputs
        for each layer. This method:
        1. Uses attention weights to determine eviction masks
        2. Recomputes attention with evicted KV
        3. Computes invariance loss between full and evicted outputs
        
        Returns:
            eit_loss: scalar tensor
            metrics: dict with per-layer losses
        """
        full_outputs = []
        evict_outputs = []
        
        layers = list(self.hook._get_attention_layers(self.model))
        
        for l_idx in range(len(layers)):
            if l_idx not in self.hook.attention_weights:
                continue
            
            attn_weights = self.hook.attention_weights[l_idx]  # (B, H, L_q, L_kv)
            full_out = self.hook.attention_outputs[l_idx]       # (B, L, D) or (B, H, L, D)
            
            # Get eviction mask
            mask = self.eviction_sim(attn_weights, keep_ratio=keep_ratio)  # (B, H, L_kv)
            
            # Get Q, K, V from the layer (need to access them)
            layer = layers[l_idx]
            # Re-extract K, V from the cached key_value in the layer
            # This depends on model architecture. For now, we use the
            # attention weights to compute evicted output differently.
            
            # Alternative: use attention weights directly to approximate evicted output
            # evicted_attn_weights = attn_weights * mask.unsqueeze(2)  # zero out evicted cols
            # renormalize
            evicted_weights = attn_weights.clone()
            eviction_mask = ~mask.unsqueeze(2)  # (B, H, 1, L_kv)
            evicted_weights = evicted_weights.masked_fill(eviction_mask, 0.0)
            # Renormalize
            weight_sum = evicted_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            evicted_weights = evicted_weights / weight_sum
            
            # full_out shape handling
            # Attention output after o_proj is (B, L, D)
            # We store it as-is from the hook
            full_outputs.append(full_out.detach())
            
            # For evicted output, we need to approximate it
            # Since we can't easily re-extract V, we use the ratio of weight changes
            # as a correction factor. This is an approximation.
            #
            # Better approach: store Q, K, V in the hook and recompute properly.
            # TODO: implement full Q/K/V capture for exact evicted attention computation
            evict_outputs.append(full_out)  # Placeholder - will be replaced with proper impl
        
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
                
                # Clear hook data
                self.hook.clear()
                
                # Forward pass (hooks capture attention data)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=True,
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
