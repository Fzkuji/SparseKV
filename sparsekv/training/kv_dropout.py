"""
KV Dropout Module

Implements KV cache eviction during forward pass via attention masking.

Approach:
- Given a KV retention mask (B, L), construct a 4D attention mask
- The 4D mask is (B, 1, L, L) where mask[b, :, i, j] = can query i attend to key j?
- During SDPA, masked-out positions receive -inf logits → zero attention

This simulates KV cache eviction without modifying the model architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class KVDropoutContext:
    """
    Context manager for applying KV dropout during forward pass.
    
    Usage:
        with KVDropoutContext(model, kv_mask):
            outputs = model(input_ids, attention_mask=...)
    """
    
    def __init__(
        self,
        model: nn.Module,
        kv_mask: torch.Tensor,
        original_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            model: the LLM (e.g., LlamaForCausalLM, Qwen2ForCausalLM)
            kv_mask: (B, L) boolean mask, True = keep this KV position
            original_attention_mask: (B, L) padding mask, 1 = valid token
        """
        self.model = model
        self.kv_mask = kv_mask
        self.original_attention_mask = original_attention_mask
        
        self._hooks = []
        self._original_forward = {}
    
    def __enter__(self):
        """Install hooks to inject KV dropout."""
        self._install_kv_dropout()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove hooks."""
        self._remove_kv_dropout()
    
    def _install_kv_dropout(self):
        """
        Modify attention computation to apply KV masking.
        
        Strategy:
        - Find all attention layers in the model
        - Override their attention_mask handling to include KV dropout
        """
        # Build 4D attention mask
        attention_mask_4d = self._build_4d_attention_mask()
        
        # Store in model temporarily (to be accessed by attention layers)
        self.model._kv_dropout_mask = attention_mask_4d
        
        # Hook into attention layers
        for name, module in self.model.named_modules():
            if self._is_attention_layer(module):
                # Store original forward
                self._original_forward[name] = module.forward
                
                # Replace with wrapped version
                module.forward = self._make_wrapped_forward(module, attention_mask_4d)
    
    def _remove_kv_dropout(self):
        """Restore original forward methods."""
        # Remove stored mask
        if hasattr(self.model, '_kv_dropout_mask'):
            delattr(self.model, '_kv_dropout_mask')
        
        # Restore original forwards
        for name, module in self.model.named_modules():
            if name in self._original_forward:
                module.forward = self._original_forward[name]
        
        self._original_forward.clear()
    
    def _is_attention_layer(self, module) -> bool:
        """Check if module is an attention layer."""
        # Common attention layer names across models
        class_name = module.__class__.__name__
        return any(x in class_name.lower() for x in ['attention', 'attn'])
    
    def _build_4d_attention_mask(self) -> torch.Tensor:
        """
        Build 4D attention mask that combines:
        1. Causal mask (can't attend to future)
        2. Padding mask (can't attend to padding)
        3. KV dropout mask (can't attend to evicted KV positions)
        
        Returns:
            mask_4d: (B, 1, L, L) where mask[b, :, i, j] indicates if query i can attend to key j
                     Values: 0.0 = can attend, -inf = cannot attend
        """
        B, L = self.kv_mask.shape
        device = self.kv_mask.device
        dtype = torch.float32  # SDPA will cast as needed
        
        # Start with causal mask: (L, L) upper triangular
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=device, dtype=dtype),
            diagonal=1,
        )
        # Shape: (1, 1, L, L)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine with KV dropout mask
        # kv_mask: (B, L) → broadcast to (B, 1, 1, L)
        # If kv_mask[b, j] = False → cannot attend to position j
        kv_mask_4d = self.kv_mask.unsqueeze(1).unsqueeze(1).float()  # (B, 1, 1, L)
        kv_mask_4d = torch.where(
            kv_mask_4d.bool(),
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(float('-inf'), device=device, dtype=dtype),
        )
        
        # Combine causal + KV dropout
        mask_4d = causal_mask + kv_mask_4d  # Broadcasting: (B, 1, L, L)
        
        # Apply padding mask if provided
        if self.original_attention_mask is not None:
            # original_attention_mask: (B, L), 1 = valid, 0 = padding
            padding_mask = self.original_attention_mask.unsqueeze(1).unsqueeze(1).float()  # (B, 1, 1, L)
            padding_mask = torch.where(
                padding_mask.bool(),
                torch.tensor(0.0, device=device, dtype=dtype),
                torch.tensor(float('-inf'), device=device, dtype=dtype),
            )
            mask_4d = mask_4d + padding_mask
        
        return mask_4d
    
    def _make_wrapped_forward(self, original_module, kv_dropout_mask):
        """
        Create a wrapped forward function that injects KV dropout mask.
        
        This is tricky because different models have different attention APIs.
        We'll use a generic approach: if attention_mask is passed, merge it with our mask.
        """
        original_forward = original_module.forward
        
        def wrapped_forward(*args, **kwargs):
            # Inject or merge attention_mask
            if 'attention_mask' in kwargs:
                # Merge with existing mask
                existing_mask = kwargs['attention_mask']
                # If existing mask is 2D, expand it
                if existing_mask.dim() == 2:
                    B, L = existing_mask.shape
                    # This is a padding mask, already handled in _build_4d_attention_mask
                    pass
                # Use our KV dropout mask
                kwargs['attention_mask'] = kv_dropout_mask
            else:
                kwargs['attention_mask'] = kv_dropout_mask
            
            return original_forward(*args, **kwargs)
        
        return wrapped_forward


def apply_kv_dropout(
    model: nn.Module,
    input_ids: torch.Tensor,
    kv_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    Convenience function: run forward pass with KV dropout.
    
    Args:
        model: LLM
        input_ids: (B, L)
        kv_mask: (B, L) boolean, True = keep
        attention_mask: (B, L) padding mask
        labels: (B, L) optional labels for loss
        **kwargs: other forward arguments
    
    Returns:
        model outputs
    """
    with KVDropoutContext(model, kv_mask, attention_mask):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
    return outputs


# ============================================================
# Alternative: Patch SDPA directly (more robust)
# ============================================================

class SDPAKVDropout:
    """
    More robust approach: patch torch.nn.functional.scaled_dot_product_attention
    to inject KV mask directly.
    
    This avoids model-specific attention layer detection.
    """
    
    def __init__(self, kv_mask: torch.Tensor, original_attention_mask: Optional[torch.Tensor] = None):
        self.kv_mask = kv_mask
        self.original_attention_mask = original_attention_mask
        self.attention_mask_4d = None
        
        self._original_sdpa = None
    
    def __enter__(self):
        """Patch SDPA."""
        self.attention_mask_4d = self._build_4d_mask()
        
        # Save original SDPA
        import torch.nn.functional as F
        self._original_sdpa = F.scaled_dot_product_attention
        
        # Replace with wrapper
        F.scaled_dot_product_attention = self._wrapped_sdpa
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore SDPA."""
        import torch.nn.functional as F
        if self._original_sdpa is not None:
            F.scaled_dot_product_attention = self._original_sdpa
    
    def _build_4d_mask(self) -> torch.Tensor:
        """Same as KVDropoutContext._build_4d_attention_mask."""
        B, L = self.kv_mask.shape
        device = self.kv_mask.device
        dtype = torch.float32
        
        # Causal mask
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=device, dtype=dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        
        # KV dropout mask
        kv_mask_4d = self.kv_mask.unsqueeze(1).unsqueeze(1).float()
        kv_mask_4d = torch.where(
            kv_mask_4d.bool(),
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(float('-inf'), device=device, dtype=dtype),
        )
        
        mask_4d = causal_mask + kv_mask_4d
        
        # Padding
        if self.original_attention_mask is not None:
            padding_mask = self.original_attention_mask.unsqueeze(1).unsqueeze(1).float()
            padding_mask = torch.where(
                padding_mask.bool(),
                torch.tensor(0.0, device=device, dtype=dtype),
                torch.tensor(float('-inf'), device=device, dtype=dtype),
            )
            mask_4d = mask_4d + padding_mask
        
        return mask_4d
    
    def _wrapped_sdpa(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        """
        Wrapped SDPA that injects our KV dropout mask.
        
        Args are same as torch.nn.functional.scaled_dot_product_attention.
        """
        # Merge our mask with any existing mask
        if attn_mask is not None:
            # Merge (element-wise minimum of logits)
            combined_mask = torch.minimum(attn_mask, self.attention_mask_4d)
        else:
            combined_mask = self.attention_mask_4d
        
        # Call original SDPA with our mask
        return self._original_sdpa(
            query, key, value,
            attn_mask=combined_mask,
            dropout_p=dropout_p,
            is_causal=False,  # Already handled in our mask
            **kwargs,
        )


def apply_kv_dropout_sdpa(
    model: nn.Module,
    input_ids: torch.Tensor,
    kv_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    Apply KV dropout via SDPA patching (recommended).
    
    More robust than hooking into attention layers.
    """
    with SDPAKVDropout(kv_mask, attention_mask):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
    return outputs
