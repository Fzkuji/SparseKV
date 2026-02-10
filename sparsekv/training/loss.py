"""
Module 2: Eviction Invariance Loss

Core innovation: Layer-wise distillation loss that forces the model to produce
similar attention outputs with and without evicted tokens.

L_total = L_LM + λ · Σ_l w_l · L_inv^l

Where L_inv^l measures the discrepancy between full and evicted attention outputs
at layer l.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class LossConfig:
    """Configuration for EIT loss."""
    lambda_eit: float = 1.0           # Weight for invariance loss
    loss_type: str = "mse"            # "mse", "cosine", "kl", "huber"
    layer_weighting: str = "uniform"  # "uniform", "linear", "learned"
    normalize: bool = True            # Normalize outputs before computing loss


class EvictionInvarianceLoss(nn.Module):
    """
    Computes the eviction invariance loss between full and evicted attention outputs.
    
    For each layer l:
        L_inv^l = distance(O_l^full, O_l^evict)
    
    Where O_l^full is the attention output with full KV cache,
    and O_l^evict is the attention output with evicted KV cache.
    """
    
    def __init__(self, config: LossConfig, num_layers: int):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        
        # Layer weights
        if config.layer_weighting == "learned":
            self.layer_weights = nn.Parameter(torch.ones(num_layers))
        elif config.layer_weighting == "linear":
            # Deeper layers get more weight
            weights = torch.linspace(0.5, 1.5, num_layers)
            self.register_buffer("layer_weights", weights)
        else:  # uniform
            self.register_buffer("layer_weights", torch.ones(num_layers))
    
    def compute_layer_loss(
        self,
        full_output: torch.Tensor,
        evict_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute invariance loss for a single layer.
        
        Args:
            full_output: (B, L, D) attention output with full KV
            evict_output: (B, L, D) attention output with evicted KV
            
        Returns:
            loss: scalar
        """
        if self.config.normalize:
            full_output = F.normalize(full_output, dim=-1)
            evict_output = F.normalize(evict_output, dim=-1)
        
        if self.config.loss_type == "mse":
            return F.mse_loss(evict_output, full_output.detach())
        
        elif self.config.loss_type == "huber":
            return F.smooth_l1_loss(evict_output, full_output.detach())
        
        elif self.config.loss_type == "cosine":
            # 1 - cosine_similarity, averaged over all positions
            cos_sim = F.cosine_similarity(
                evict_output, full_output.detach(), dim=-1
            )  # (B, L)
            return (1 - cos_sim).mean()
        
        elif self.config.loss_type == "kl":
            # Treat attention outputs as logits, compute KL divergence
            full_log_prob = F.log_softmax(full_output.detach(), dim=-1)
            evict_log_prob = F.log_softmax(evict_output, dim=-1)
            return F.kl_div(evict_log_prob, full_log_prob, log_target=True, reduction="batchmean")
        
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def forward(
        self,
        full_outputs: list[torch.Tensor],
        evict_outputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute total eviction invariance loss across all layers.
        
        Args:
            full_outputs: list of (B, L, D) tensors, one per layer
            evict_outputs: list of (B, L, D) tensors, one per layer
            
        Returns:
            total_loss: scalar, λ * Σ_l w_l * L_inv^l
            layer_losses: dict with per-layer loss values for logging
        """
        assert len(full_outputs) == len(evict_outputs) == self.num_layers
        
        # Normalize layer weights
        weights = F.softmax(self.layer_weights, dim=0) if self.config.layer_weighting == "learned" \
            else self.layer_weights / self.layer_weights.sum()
        
        total_loss = torch.tensor(0.0, device=full_outputs[0].device)
        layer_losses = {}
        
        for l, (full_out, evict_out) in enumerate(zip(full_outputs, evict_outputs)):
            l_loss = self.compute_layer_loss(full_out, evict_out)
            total_loss = total_loss + weights[l] * l_loss
            layer_losses[f"layer_{l}"] = l_loss.item()
        
        total_loss = self.config.lambda_eit * total_loss
        layer_losses["total"] = total_loss.item()
        
        return total_loss, layer_losses
