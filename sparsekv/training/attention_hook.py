"""
Attention Hook: Captures Q, K, V and attention outputs from each layer.

This is the bridge between the model forward pass and EIT loss computation.
We hook into the attention module to intercept Q, K, V before attention,
and the attention output after attention. This allows us to exactly recompute
attention with evicted KV cache.

Supports: Llama, Qwen, Mistral, Gemma (all use similar attention structure).
"""

import math
import logging
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LayerCapture:
    """Captured data from one attention layer."""
    query: torch.Tensor          # (B, H_q, L, D_head)
    key: torch.Tensor            # (B, H_kv, L, D_head)
    value: torch.Tensor          # (B, H_kv, L, D_head)
    attn_output: torch.Tensor    # (B, L, H_q * D_head) — after o_proj
    attn_weights: Optional[torch.Tensor] = None  # (B, H_q, L, L) if available
    num_key_value_groups: int = 1  # GQA groups


class AttentionHook:
    """
    Hooks into transformer attention layers to capture Q, K, V and outputs.
    
    Strategy:
    - Pre-hook on attention forward to capture Q, K, V after projection
    - Post-hook to capture attention output
    
    We monkey-patch the attention forward to inject our capture logic,
    since standard hooks don't give us access to intermediate Q, K, V.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.captures: dict[int, LayerCapture] = {}
        self._original_forwards = {}
        self._patched = False
    
    def patch(self, model: nn.Module):
        """Monkey-patch attention layers to capture Q, K, V."""
        layers = self._get_layers(model)
        
        for idx, attn_module in enumerate(layers):
            self._original_forwards[idx] = attn_module.forward
            attn_module.forward = self._make_patched_forward(idx, attn_module)
        
        self._patched = True
        logger.info(f"Patched {len(layers)} attention layers for Q/K/V capture")
    
    def unpatch(self, model: nn.Module):
        """Restore original attention forwards."""
        layers = self._get_layers(model)
        for idx, attn_module in enumerate(layers):
            if idx in self._original_forwards:
                attn_module.forward = self._original_forwards[idx]
        self._original_forwards.clear()
        self._patched = False
    
    def _get_layers(self, model: nn.Module) -> list[nn.Module]:
        """Extract attention modules."""
        # Unwrap PEFT if needed
        base = model
        if hasattr(model, 'base_model'):
            base = model.base_model
        if hasattr(base, 'model') and hasattr(base.model, 'model'):
            # PeftModel wrapping
            base = base.model
        
        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
            return [layer.self_attn for layer in base.model.layers]
        elif hasattr(base, 'transformer') and hasattr(base.transformer, 'h'):
            return [layer.attn for layer in base.transformer.h]
        else:
            raise ValueError(f"Unsupported architecture: {type(base)}")
    
    def _make_patched_forward(self, layer_idx: int, attn_module: nn.Module):
        """Create a patched forward that captures Q, K, V."""
        original_forward = self._original_forwards[layer_idx]
        hook = self
        
        def patched_forward(hidden_states, **kwargs):
            if not hook.enabled:
                return original_forward(hidden_states, **kwargs)
            
            # Force output_attentions to get attention weights
            kwargs_with_attn = {**kwargs, "output_attentions": True}
            
            # Capture Q, K, V by hooking into the projections
            # Most models compute: Q = q_proj(hidden), K = k_proj(hidden), V = v_proj(hidden)
            q = attn_module.q_proj(hidden_states)
            k_input = hidden_states
            v_input = hidden_states
            
            # Handle cross-attention or different kv input
            # For self-attention, K and V come from the same hidden_states
            k = attn_module.k_proj(k_input)
            v = attn_module.v_proj(v_input)
            
            # Reshape to (B, H, L, D)
            B, L, _ = hidden_states.shape
            head_dim = attn_module.head_dim
            num_heads = attn_module.num_heads
            num_kv_heads = getattr(attn_module, 'num_key_value_heads', num_heads)
            num_kv_groups = num_heads // num_kv_heads
            
            q_states = q.view(B, L, num_heads, head_dim).transpose(1, 2)
            k_states = k.view(B, L, num_kv_heads, head_dim).transpose(1, 2)
            v_states = v.view(B, L, num_kv_heads, head_dim).transpose(1, 2)
            
            # Run original forward
            output = original_forward(hidden_states, **kwargs_with_attn)
            
            # Extract attention output and weights
            if isinstance(output, tuple):
                attn_output = output[0]
                attn_weights = output[1] if len(output) > 1 else None
            else:
                attn_output = output
                attn_weights = None
            
            # Store capture
            hook.captures[layer_idx] = LayerCapture(
                query=q_states.detach(),
                key=k_states.detach(),
                value=v_states.detach(),
                attn_output=attn_output,  # Keep gradient for loss
                attn_weights=attn_weights.detach() if attn_weights is not None else None,
                num_key_value_groups=num_kv_groups,
            )
            
            # Return original output format (without forcing attn weights in output)
            # Check if caller wanted attention weights
            if kwargs.get("output_attentions", False):
                return output
            else:
                # Return just the attention output (and other non-attn-weight items)
                if isinstance(output, tuple):
                    return (output[0],) + output[2:]  # Skip attn_weights
                return output
        
        return patched_forward
    
    def clear(self):
        """Clear all captures."""
        self.captures.clear()
    
    @property
    def num_layers(self) -> int:
        return len(self.captures)


def compute_evicted_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    eviction_mask: torch.Tensor,
    num_kv_groups: int = 1,
    causal: bool = True,
) -> torch.Tensor:
    """
    Recompute attention output using only kept (non-evicted) KV positions.
    
    This is the core operation for EIT: given the original Q, K, V and an
    eviction mask, compute what the attention output would be if the evicted
    tokens were never in the KV cache.
    
    Args:
        query: (B, H_q, L, D) query states
        key: (B, H_kv, L, D) key states
        value: (B, H_kv, L, D) value states  
        eviction_mask: (B, H_kv, L) binary mask, True = keep, False = evict
        num_kv_groups: number of query heads per KV head (for GQA)
        causal: whether to apply causal mask
        
    Returns:
        evicted_attn_output: (B, H_q, L, D)
    """
    B, H_q, L, D = query.shape
    H_kv = key.shape[1]
    
    # Expand KV for GQA: (B, H_kv, L, D) -> (B, H_q, L, D)
    if num_kv_groups > 1:
        key = key.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1).reshape(B, H_q, L, D)
        value = value.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1).reshape(B, H_q, L, D)
        # Expand mask: (B, H_kv, L) -> (B, H_q, L)
        eviction_mask = eviction_mask.unsqueeze(2).expand(-1, -1, num_kv_groups, -1).reshape(B, H_q, L)
    
    # Compute attention scores
    scale = 1.0 / math.sqrt(D)
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # (B, H_q, L, L)
    
    # Apply causal mask
    if causal:
        causal_mask = torch.triu(
            torch.ones(L, L, device=query.device, dtype=torch.bool), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Apply eviction mask: mask out evicted KV positions
    evict_mask = ~eviction_mask.unsqueeze(2)  # (B, H_q, 1, L)
    attn_scores = attn_scores.masked_fill(evict_mask, float('-inf'))
    
    # Softmax
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_weights = attn_weights.nan_to_num(0.0)  # Handle all-masked rows
    
    # Compute output
    attn_output = torch.matmul(attn_weights, value)  # (B, H_q, L, D)
    
    return attn_output
