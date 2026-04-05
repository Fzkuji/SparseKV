"""
Attention Capture: Extract windowed attention importance during teacher forward.

Patches SDPA to compute Q[-W:] @ K^T importance scores (matching SnapKV inference)
without materializing the full L×L attention matrix.

Usage:
    with AttentionCapture(window_size=64, num_kv_heads=8) as cap:
        outputs = model(input_ids=input_ids)
    importance = cap.importance  # list[num_layers] of (B, kv_heads, L)
"""

import contextlib

import torch
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionCapture:
    """
    Context manager that patches SDPA to capture windowed attention importance.

    For each layer, computes:
        scores = Q[:, :, -W:, :] @ K^T / sqrt(d)   # (B, H_q, W, L)
        importance = softmax(scores, dim=-1).mean(dim=2)  # (B, H_q, L)

    Then aggregates from query heads to KV heads (GQA-aware).

    This matches SnapKV's inference-time importance estimation:
    use the last W query positions to score all K positions.
    """

    def __init__(self, window_size: int = 64, num_kv_heads: Optional[int] = None,
                 differentiable: bool = False):
        """
        Args:
            window_size: Number of trailing query positions to use for scoring.
            num_kv_heads: Number of KV heads (for GQA aggregation).
                          If None, no aggregation is performed.
            differentiable: If True, keep gradient graph for importance scores
                            (needed for entropy loss backprop).
        """
        self.window_size = window_size
        self.num_kv_heads = num_kv_heads
        self.differentiable = differentiable
        self.importance: List[torch.Tensor] = []  # list[num_layers] of (B, kv_heads, L)
        self._layer_counter = 0
        self._original_sdpa = None

    def __enter__(self):
        self._layer_counter = 0
        self.importance = []
        self._original_sdpa = F.scaled_dot_product_attention

        ctx = self
        original = self._original_sdpa

        def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            # Capture importance from this layer's Q and K
            # query: (B, H_q, L_q, d), key: (B, H_kv, L_kv, d)
            ctx._capture_importance(query, key, attn_mask, is_causal)
            ctx._layer_counter += 1

            # Call original SDPA unchanged
            return original(query, key, value, attn_mask=attn_mask,
                            dropout_p=dropout_p, is_causal=is_causal, **kwargs)

        F.scaled_dot_product_attention = patched_sdpa
        return self

    def __exit__(self, *args):
        F.scaled_dot_product_attention = self._original_sdpa
        self._original_sdpa = None

    def _capture_importance(self, query, key, attn_mask, is_causal):
        """
        Compute windowed attention importance for one layer.

        Only uses the last W query positions to avoid L² memory.
        When differentiable=True, keeps the gradient graph for backprop.
        """
        grad_ctx = contextlib.nullcontext() if self.differentiable else torch.no_grad()
        with grad_ctx:
            B, H_q, L_q, d = query.shape
            _, H_kv, L_kv, _ = key.shape

            W = min(self.window_size, L_q)

            # Use last W queries: (B, H_q, W, d)
            q_window = query[:, :, -W:, :]

            # If GQA, expand key to match query heads for score computation
            if H_q != H_kv and H_kv > 0:
                groups = H_q // H_kv
                # (B, H_kv, L_kv, d) → (B, H_q, L_kv, d)
                key_expanded = key.unsqueeze(2).expand(B, H_kv, groups, L_kv, d)
                key_expanded = key_expanded.reshape(B, H_q, L_kv, d)
            else:
                key_expanded = key

            # scores: (B, H_q, W, L_kv)
            scale = d ** -0.5
            scores = torch.matmul(q_window, key_expanded.transpose(-2, -1)) * scale

            # Apply causal mask for the window region
            # Query positions are [L_q-W, ..., L_q-1], Key positions are [0, ..., L_kv-1]
            # Query i can attend to Key j only if (L_q - W + i) >= j
            if is_causal:
                q_positions = torch.arange(L_q - W, L_q, device=query.device)  # (W,)
                k_positions = torch.arange(L_kv, device=query.device)  # (L_kv,)
                causal = q_positions.unsqueeze(1) < k_positions.unsqueeze(0)  # (W, L_kv)
                scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

            # Apply existing attention mask if present
            if attn_mask is not None:
                if attn_mask.dim() == 4:
                    # (B, H, L_q, L_kv) → take last W rows
                    mask_window = attn_mask[:, :H_q, -W:, :L_kv]
                elif attn_mask.dim() == 2:
                    # (L_q, L_kv) → take last W rows
                    mask_window = attn_mask[-W:, :L_kv].unsqueeze(0).unsqueeze(0)
                else:
                    mask_window = None

                if mask_window is not None:
                    scores = scores + mask_window

            # Softmax → mean over window → (B, H_q, L_kv)
            attn_weights = torch.softmax(scores, dim=-1)
            importance = attn_weights.mean(dim=2)  # (B, H_q, L_kv)

            # GQA aggregation: (B, H_q, L_kv) → (B, H_kv, L_kv)
            if self.num_kv_heads is not None and H_q != self.num_kv_heads:
                groups = H_q // self.num_kv_heads
                importance = importance.reshape(B, self.num_kv_heads, groups, L_kv)
                importance = importance.mean(dim=2)  # Average across query heads in each group

            self.importance.append(importance if self.differentiable else importance.detach())
