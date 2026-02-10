"""
Module 1: Eviction Simulation

Simulates KV cache eviction during training.
Uses AdaKV-style per-layer per-head adaptive eviction:
- Each head gets a budget based on its attention distribution
- Heads with concentrated attention can be compressed more
- Heads with dispersed attention retain more tokens
- Safeguard ensures minimum retention per head
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvictionConfig:
    """Configuration for eviction simulation."""
    keep_ratio: float = 0.5           # Global target: fraction of tokens to keep
    sink_size: int = 4                # Number of sink tokens to always keep
    recent_size: int = 64             # Number of recent tokens to always keep
    alpha_safeguard: float = 0.20     # Min fraction each head must retain (AdaKV)
    adaptive_heads: bool = True       # Per-head adaptive budget (AdaKV-style)


class AttentionScoreEviction(nn.Module):
    """
    Eviction simulation based on attention scores.
    
    Given attention weights from a layer, determines which KV positions to evict.
    Returns a binary mask indicating which positions to keep.
    
    Supports two modes:
    - Uniform: each head keeps the same number of tokens
    - Adaptive (AdaKV-style): budget allocated across heads based on attention entropy
    """
    
    def __init__(self, config: EvictionConfig):
        super().__init__()
        self.config = config
    
    def compute_head_budgets(
        self,
        attn_weights: torch.Tensor,
        total_budget: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        AdaKV-style: allocate per-head budgets based on attention distribution.
        
        Heads with higher entropy (more dispersed attention) get more budget.
        Heads with lower entropy (concentrated attention) get less budget.
        
        Args:
            attn_weights: (B, H, L_q, L_kv) attention weights
            total_budget: total number of tokens to keep across all heads
            seq_len: sequence length
            
        Returns:
            head_budgets: (B, H) number of tokens each head should keep
        """
        B, H, L_q, L_kv = attn_weights.shape
        
        # Compute attention entropy per head (average over query positions)
        # Higher entropy = more dispersed = needs more tokens
        eps = 1e-8
        entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)  # (B, H, L_q)
        head_entropy = entropy.mean(dim=-1)  # (B, H)
        
        # Normalize entropy to get allocation weights
        alloc_weights = head_entropy / (head_entropy.sum(dim=-1, keepdim=True) + eps)  # (B, H)
        
        # Allocate budget proportionally
        head_budgets = (alloc_weights * total_budget).round().long()  # (B, H)
        
        # Apply safeguard: minimum per-head budget
        min_budget = max(int(seq_len * (1 - self.config.keep_ratio) * self.config.alpha_safeguard), 1)
        # Wait, min_budget should be based on keep ratio
        min_budget = max(int(seq_len * self.config.keep_ratio * self.config.alpha_safeguard), 1)
        head_budgets = head_budgets.clamp(min=min_budget)
        
        # Adjust to match total budget (redistribute excess/deficit)
        current_total = head_budgets.sum(dim=-1, keepdim=True)  # (B, 1)
        diff = total_budget - current_total  # (B, 1)
        # Distribute diff evenly across heads
        per_head_adj = diff // H
        head_budgets = head_budgets + per_head_adj
        # Handle remainder
        remainder = diff - per_head_adj * H
        # Add 1 to first `remainder` heads
        for b in range(B):
            r = remainder[b, 0].item()
            if r > 0:
                head_budgets[b, :r] += 1
            elif r < 0:
                head_budgets[b, :(-r)] -= 1
        
        head_budgets = head_budgets.clamp(min=1, max=seq_len)
        return head_budgets
    
    def forward(
        self,
        attn_weights: torch.Tensor,
        keep_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute eviction mask based on attention scores.
        
        Args:
            attn_weights: (B, H, L_q, L_kv) attention weights from current layer
            keep_ratio: override config keep_ratio (for curriculum scheduling)
            
        Returns:
            mask: (B, H, L_kv) binary mask, 1 = keep, 0 = evict
        """
        B, H, L_q, L_kv = attn_weights.shape
        kr = keep_ratio if keep_ratio is not None else self.config.keep_ratio
        device = attn_weights.device
        
        # Compute importance scores: sum of attention weights across all query positions
        scores = attn_weights.sum(dim=2)  # (B, H, L_kv)
        
        # Always keep sink tokens and recent tokens
        mask = torch.zeros(B, H, L_kv, device=device, dtype=torch.bool)
        
        sink = self.config.sink_size
        recent = self.config.recent_size
        if sink > 0:
            mask[:, :, :sink] = True
        if recent > 0:
            mask[:, :, -recent:] = True
        
        # Budget for middle tokens
        n_protected = min(sink + recent, L_kv)
        middle_len = L_kv - n_protected
        
        if middle_len <= 0:
            # Sequence too short, keep everything
            return torch.ones(B, H, L_kv, device=device, dtype=torch.bool)
        
        total_keep = int(L_kv * kr)
        middle_budget = max(total_keep - n_protected, 0)
        
        if self.config.adaptive_heads:
            # AdaKV-style: per-head adaptive budget
            total_middle_budget = middle_budget * H
            head_budgets = self.compute_head_budgets(
                attn_weights, total_middle_budget, middle_len
            )  # (B, H)
            
            # Extract middle scores
            middle_scores = scores[:, :, sink:(L_kv - recent if recent > 0 else L_kv)]  # (B, H, middle_len)
            
            # Per-head top-k with different k per head
            for b in range(B):
                for h in range(H):
                    k = min(head_budgets[b, h].item(), middle_len)
                    if k > 0:
                        topk_idx = middle_scores[b, h].topk(k).indices
                        mask[b, h, sink + topk_idx] = True
        else:
            # Uniform: each head keeps same number
            middle_scores = scores[:, :, sink:(L_kv - recent if recent > 0 else L_kv)]
            if middle_budget > 0:
                k = min(middle_budget, middle_len)
                topk_idx = middle_scores.topk(k, dim=-1).indices  # (B, H, k)
                # Convert to full sequence indices
                topk_idx = topk_idx + sink
                mask.scatter_(2, topk_idx, True)
        
        return mask


def apply_eviction_mask(
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply eviction mask by setting evicted positions to zero.
    
    Instead of physically removing tokens (which changes sequence length),
    we zero out evicted positions. This allows the evicted attention to be
    computed with the same tensor shapes, making it differentiable.
    
    For attention computation, we also need to mask the attention logits
    to -inf for evicted positions. See compute_evicted_attention().
    
    Args:
        keys: (B, H, L, D)
        values: (B, H, L, D)
        mask: (B, H, L) binary mask, True = keep
        
    Returns:
        masked_keys: (B, H, L, D)
        masked_values: (B, H, L, D)
    """
    mask_expanded = mask.unsqueeze(-1)  # (B, H, L, 1)
    return keys * mask_expanded, values * mask_expanded
