# SPDX-License-Identifier: Apache-2.0

"""
Eviction stability metrics inspired by DefensiveKV.

Measures how stable the importance of retained tokens is across different
generation steps. Unstable importance means eviction decisions may be unreliable.
"""

import torch
from torch import nn
from transformers import DynamicCache, PreTrainedModel

from kvpress.presses.scorer_press import ScorerPress


def compute_retained_importance(
    attention: torch.Tensor,
    compression_ratio: float = 0.5,
) -> dict[str, float]:
    """
    Compute retained importance after eviction.

    Simulates eviction by selecting top tokens based on attention scores,
    then measures what fraction of total attention mass is retained.

    Parameters
    ----------
    attention : torch.Tensor
        Attention weights (batch, num_heads, q_len, kv_len).
    compression_ratio : float
        Fraction of KV pairs to remove.

    Returns
    -------
    dict
        min_retained_importance: worst-case retention across heads
        mean_retained_importance: average retention
        std_retained_importance: variance of retention
    """
    # Average attention over queries to get per-KV importance
    # (batch, num_heads, kv_len)
    importance = attention.mean(dim=2)

    batch_size, num_heads, kv_len = importance.shape
    n_kept = int(kv_len * (1 - compression_ratio))

    # Select top-k for each head
    top_k_values = importance.topk(n_kept, dim=-1).values  # (B, H, n_kept)
    retained_mass = top_k_values.sum(dim=-1)  # (B, H)
    total_mass = importance.sum(dim=-1)  # (B, H)

    # Retention ratio per head
    retention = retained_mass / (total_mass + 1e-8)  # (B, H)

    return {
        "min_retained_importance": retention.min().item(),
        "mean_retained_importance": retention.mean().item(),
        "std_retained_importance": retention.std().item(),
    }


@torch.no_grad()
def compute_stability_across_steps(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    press: ScorerPress,
    n_steps: int = 10,
) -> dict[str, float]:
    """
    Measure how stable eviction decisions are across generation steps.

    Generates tokens step by step and tracks whether the same KV pairs
    would be selected for retention at each step.

    Parameters
    ----------
    model : PreTrainedModel
        The language model.
    input_ids : torch.Tensor
        Input token ids (batch, seq_len).
    press : ScorerPress
        A kvpress ScorerPress to compute importance scores.
    n_steps : int
        Number of generation steps to track.

    Returns
    -------
    dict
        jaccard_stability: average Jaccard similarity of retained sets between steps
        flip_rate: fraction of tokens that change retain/evict status between steps
    """
    model.eval()
    device = input_ids.device

    cache = DynamicCache()
    prev_retained_set: set | None = None
    jaccard_scores: list[float] = []
    flip_counts: list[float] = []

    # Initial prefill
    outputs = model(input_ids, use_cache=True, past_key_values=cache, output_attentions=True)
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

    for step in range(n_steps):
        outputs = model(
            next_token,
            use_cache=True,
            past_key_values=cache,
            output_attentions=True,
        )

        # Get attention from last layer
        last_attn = outputs.attentions[-1]  # (B, H, 1, kv_len)
        importance = last_attn.mean(dim=(0, 1)).squeeze()  # (kv_len,)

        kv_len = importance.shape[0]
        n_kept = int(kv_len * 0.5)  # 50% retention
        retained_indices = set(importance.topk(n_kept).indices.cpu().tolist())

        if prev_retained_set is not None:
            # Jaccard similarity
            intersection = len(retained_indices & prev_retained_set)
            union = len(retained_indices | prev_retained_set)
            jaccard = intersection / max(union, 1)
            jaccard_scores.append(jaccard)

            # Flip rate
            flips = len(retained_indices.symmetric_difference(prev_retained_set))
            flip_counts.append(flips / max(kv_len, 1))

        prev_retained_set = retained_indices
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

    return {
        "jaccard_stability": sum(jaccard_scores) / max(len(jaccard_scores), 1),
        "flip_rate": sum(flip_counts) / max(len(flip_counts), 1),
    }


def compute_stability_metrics(
    attention: torch.Tensor,
    compression_ratio: float = 0.5,
) -> dict[str, float]:
    """Compute all stability metrics from attention weights."""
    return compute_retained_importance(attention, compression_ratio)
