"""
KV Dropout: Per-layer per-head KV cache eviction during training.

Mask shape: (B, num_layers, num_heads, L)
- Each layer and each head can have a different eviction pattern
- Anchor tokens are always kept across all layers/heads
- Non-anchor budget is allocated per-head (AdaKV-style: heads with more
  concentrated attention get less budget, dispersed heads get more)

Implementation: patches torch SDPA to inject layer-specific per-head masks.
A layer counter tracks which layer is currently being computed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def create_kv_dropout_mask(
    anchor_mask: torch.Tensor,
    keep_ratio: float,
    seq_len: int,
    num_layers: int = 1,
    num_heads: int = 1,
) -> torch.Tensor:
    """
    Create per-layer per-head KV dropout mask.
    
    Args:
        anchor_mask: (B, L) bool, True = anchor (always keep)
        keep_ratio: fraction of total tokens to keep
        seq_len: sequence length
        num_layers: number of attention layers
        num_heads: number of KV heads (num_key_value_heads, not num_attention_heads)
        
    Returns:
        mask: (B, num_layers, num_heads, L) bool, True = keep
    """
    B, L = anchor_mask.shape
    device = anchor_mask.device
    
    total_keep = max(int(L * keep_ratio), 1)
    
    # Expand anchor mask to all layers/heads: anchors always kept
    # (B, L) → (B, num_layers, num_heads, L)
    mask = anchor_mask.unsqueeze(1).unsqueeze(2).expand(B, num_layers, num_heads, L).clone()
    
    num_anchors = anchor_mask.sum(dim=-1)  # (B,)
    
    for b in range(B):
        n_anchors = num_anchors[b].item()
        remaining_budget = max(total_keep - n_anchors, 0)
        
        if remaining_budget <= 0:
            continue
        
        non_anchor_positions = (~anchor_mask[b]).nonzero(as_tuple=True)[0]
        n_non_anchor = len(non_anchor_positions)
        
        if n_non_anchor == 0:
            continue
        
        # Each layer-head combination gets its own random sample
        for l in range(num_layers):
            for h in range(num_heads):
                n_sample = min(remaining_budget, n_non_anchor)
                perm = torch.randperm(n_non_anchor, device=device)[:n_sample]
                selected = non_anchor_positions[perm]
                mask[b, l, h, selected] = True
    
    return mask


def create_attention_based_mask(
    anchor_mask: torch.Tensor,
    importance_scores: List[torch.Tensor],
    keep_ratio: float,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    critical_fraction: float = 0.5,
    noise_scale: float = 0.1,
) -> torch.Tensor:
    """
    Create per-layer per-head KV dropout mask using attention importance.

    For each layer/head, the non-anchor budget is split:
    - critical_fraction → top-k by attention importance (+ Gumbel noise)
    - (1 - critical_fraction) → random sampling

    Args:
        anchor_mask: (B, L) bool, True = anchor (always keep)
        importance_scores: list[num_layers] of (B, num_heads, L) float importance
        keep_ratio: fraction of total tokens to keep
        seq_len: sequence length
        num_layers: number of attention layers
        num_heads: number of KV heads
        critical_fraction: fraction of non-anchor budget allocated to top-k
        noise_scale: Gumbel noise scale for smoothing top-k (0 = deterministic)

    Returns:
        mask: (B, num_layers, num_heads, L) bool, True = keep
    """
    B, L = anchor_mask.shape
    device = anchor_mask.device

    total_keep = max(int(L * keep_ratio), 1)

    # Expand anchor mask: (B, L) → (B, num_layers, num_heads, L)
    mask = anchor_mask.unsqueeze(1).unsqueeze(2).expand(B, num_layers, num_heads, L).clone()

    num_anchors = anchor_mask.sum(dim=-1)  # (B,)

    for b in range(B):
        n_anchors = num_anchors[b].item()
        remaining_budget = max(total_keep - n_anchors, 0)

        if remaining_budget <= 0:
            continue

        non_anchor_positions = (~anchor_mask[b]).nonzero(as_tuple=True)[0]
        n_non_anchor = len(non_anchor_positions)

        if n_non_anchor == 0:
            continue

        n_critical = int(remaining_budget * critical_fraction)
        n_random = remaining_budget - n_critical

        for l_idx in range(num_layers):
            # Get importance for this layer: (B, num_heads, L) → (num_heads, L)
            # Move to anchor_mask device (layers may live on different GPUs with device_map="auto")
            layer_imp = importance_scores[l_idx][b].to(device)  # (num_heads, L)

            for h in range(num_heads):
                head_imp = layer_imp[h]  # (L,)
                # Importance only at non-anchor positions
                non_anchor_imp = head_imp[non_anchor_positions]  # (n_non_anchor,)

                # --- Critical part: top-k with optional Gumbel noise ---
                if n_critical > 0:
                    if noise_scale > 0:
                        # Gumbel noise for smooth top-k
                        gumbel = -torch.log(-torch.log(
                            torch.rand_like(non_anchor_imp).clamp(min=1e-8)
                        ))
                        noisy_imp = non_anchor_imp + noise_scale * gumbel
                    else:
                        noisy_imp = non_anchor_imp

                    k_crit = min(n_critical, n_non_anchor)
                    _, top_indices = torch.topk(noisy_imp, k=k_crit)
                    critical_positions = non_anchor_positions[top_indices]
                    mask[b, l_idx, h, critical_positions] = True

                    # --- Random part: sample from remaining non-anchor non-critical ---
                    if n_random > 0:
                        # Build set of positions already selected
                        already_selected = torch.zeros(n_non_anchor, dtype=torch.bool, device=device)
                        already_selected[top_indices] = True
                        remaining_indices = (~already_selected).nonzero(as_tuple=True)[0]
                        n_remaining = len(remaining_indices)

                        if n_remaining > 0:
                            k_rand = min(n_random, n_remaining)
                            perm = torch.randperm(n_remaining, device=device)[:k_rand]
                            random_positions = non_anchor_positions[remaining_indices[perm]]
                            mask[b, l_idx, h, random_positions] = True
                else:
                    # All budget goes to random
                    n_sample = min(remaining_budget, n_non_anchor)
                    perm = torch.randperm(n_non_anchor, device=device)[:n_sample]
                    selected = non_anchor_positions[perm]
                    mask[b, l_idx, h, selected] = True

    return mask


def layer_mask_to_4d_attention_mask(
    layer_mask: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Convert a per-head mask for ONE layer to a 4D SDPA attention mask.
    
    Args:
        layer_mask: (B, num_heads, L) bool, True = keep
        seq_len: sequence length
        dtype: output dtype
        
    Returns:
        attn_mask: (B, num_heads, L, L) float, 0 = attend, -inf = mask
    """
    B, H, L = layer_mask.shape
    device = layer_mask.device
    
    # Causal mask: (1, 1, L, L)
    causal_mask = torch.triu(
        torch.full((L, L), float('-inf'), device=device, dtype=dtype),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)
    
    # KV mask: (B, H, 1, L) → broadcast to (B, H, L, L)
    kv_mask = layer_mask.unsqueeze(2).float()  # (B, H, 1, L)
    kv_mask = torch.where(
        kv_mask.bool(),
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(float('-inf'), device=device, dtype=dtype),
    )
    
    # Combine
    attn_mask = causal_mask + kv_mask  # (B, H, L, L)
    
    return attn_mask


class PerLayerKVDropout:
    """
    Context manager that patches SDPA to apply per-layer per-head KV masks.
    
    Tracks which layer is being computed via a counter. Each SDPA call
    corresponds to one layer; applies the corresponding mask from the
    full (B, num_layers, num_heads, L) mask tensor.
    
    Usage:
        mask = create_kv_dropout_mask(anchor_mask, keep_ratio, L, num_layers, num_heads)
        with PerLayerKVDropout(mask):
            outputs = model(input_ids)
    """
    
    def __init__(self, full_mask: torch.Tensor, dtype: torch.dtype = torch.bfloat16):
        """
        Args:
            full_mask: (B, num_layers, num_heads, L) bool mask
            dtype: attention mask dtype
        """
        self.full_mask = full_mask
        self.dtype = dtype
        self.B, self.num_layers, self.num_heads, self.L = full_mask.shape
        
        # Precompute 4D attention masks for each layer
        self.layer_attn_masks = []
        for l in range(self.num_layers):
            layer_mask = full_mask[:, l, :, :]  # (B, H, L)
            attn_mask = layer_mask_to_4d_attention_mask(layer_mask, self.L, dtype)
            self.layer_attn_masks.append(attn_mask)
        
        self._layer_counter = 0
        self._original_sdpa = None
    
    def __enter__(self):
        self._layer_counter = 0
        self._original_sdpa = F.scaled_dot_product_attention
        
        ctx = self  # capture for closure
        original = self._original_sdpa
        
        def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            layer_idx = ctx._layer_counter
            
            if layer_idx < ctx.num_layers:
                # Get per-head mask for this layer (move to query device for multi-GPU)
                our_mask = ctx.layer_attn_masks[layer_idx].to(query.device)

                # Handle GQA: if query heads > KV heads, expand mask
                q_heads = query.shape[1]
                kv_heads = our_mask.shape[1]
                if q_heads != kv_heads and kv_heads > 0:
                    groups = q_heads // kv_heads
                    our_mask = our_mask.unsqueeze(2).expand(
                        -1, -1, groups, -1, -1
                    ).reshape(our_mask.shape[0], q_heads, our_mask.shape[2], our_mask.shape[3])
                
                # Merge with existing mask
                if attn_mask is not None:
                    combined = attn_mask + our_mask
                else:
                    combined = our_mask
                
                ctx._layer_counter += 1
                return original(query, key, value, attn_mask=combined, dropout_p=dropout_p, is_causal=False, **kwargs)
            else:
                # Beyond expected layers, pass through
                ctx._layer_counter += 1
                return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
        
        F.scaled_dot_product_attention = patched_sdpa
        return self
    
    def __exit__(self, *args):
        F.scaled_dot_product_attention = self._original_sdpa
        self._original_sdpa = None
