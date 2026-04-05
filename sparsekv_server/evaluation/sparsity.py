# SPDX-License-Identifier: Apache-2.0

"""
Attention sparsity metrics for evaluating how concentrated attention patterns are.

These metrics help quantify whether training with SparseKV actually makes
the model's attention patterns sparser (which is the whole point).
"""

import torch


def effective_support(attention: torch.Tensor) -> torch.Tensor:
    """
    Compute effective support (exponential of entropy) of attention distributions.

    Lower values = sparser attention (more concentrated on fewer tokens).
    A value of 1.0 means all mass on one token. A value of N means uniform.

    Parameters
    ----------
    attention : torch.Tensor
        Attention weights (batch, num_heads, q_len, kv_len).

    Returns
    -------
    torch.Tensor
        Effective support averaged over batch, heads, and queries.
    """
    eps = 1e-8
    entropy = -(attention * (attention + eps).log()).sum(dim=-1)  # (B, H, Q)
    return entropy.exp().mean()


def top_k_coverage(attention: torch.Tensor, k_ratio: float = 0.3) -> torch.Tensor:
    """
    Compute fraction of attention mass captured by top-k% tokens.

    Higher values = sparser attention (most mass in few tokens).

    Parameters
    ----------
    attention : torch.Tensor
        Attention weights (batch, num_heads, q_len, kv_len).
    k_ratio : float
        Fraction of tokens to consider (0.3 = top 30%).

    Returns
    -------
    torch.Tensor
        Average coverage across batch, heads, and queries.
    """
    kv_len = attention.shape[-1]
    k = max(1, int(kv_len * k_ratio))

    top_k_values = attention.topk(k, dim=-1).values  # (B, H, Q, k)
    coverage = top_k_values.sum(dim=-1)  # (B, H, Q) — fraction of mass in top-k

    return coverage.mean()


def block_sparsity(
    attention: torch.Tensor,
    block_size: int = 64,
    threshold: float = 0.01,
) -> torch.Tensor:
    """
    Compute fraction of blocks with very low total attention.

    Higher values = more blocks can be safely evicted.

    Parameters
    ----------
    attention : torch.Tensor
        Attention weights (batch, num_heads, q_len, kv_len).
    block_size : int
        Block size for grouping tokens.
    threshold : float
        Blocks with average attention below this are considered "sparse".

    Returns
    -------
    torch.Tensor
        Fraction of sparse blocks.
    """
    # Average over batch and heads
    avg_attn = attention.mean(dim=(0, 1))  # (q_len, kv_len)

    # Average attention received per kv position across all queries
    received_attn = avg_attn.mean(dim=0)  # (kv_len,)

    # Group into blocks
    kv_len = received_attn.shape[0]
    n_blocks = kv_len // block_size

    if n_blocks == 0:
        return torch.tensor(0.0)

    # Reshape and compute per-block average
    truncated = received_attn[: n_blocks * block_size]
    block_attn = truncated.view(n_blocks, block_size).mean(dim=1)  # (n_blocks,)

    # Count sparse blocks
    sparse_count = (block_attn < threshold).float().sum()
    return sparse_count / n_blocks


def compute_sparsity_metrics(
    attention: torch.Tensor,
    block_size: int = 64,
) -> dict[str, float]:
    """
    Compute all sparsity metrics.

    Parameters
    ----------
    attention : torch.Tensor
        Attention weights (batch, num_heads, q_len, kv_len).
    block_size : int
        Block size for block_sparsity metric.

    Returns
    -------
    dict
        Dictionary of sparsity metrics.
    """
    return {
        "effective_support": effective_support(attention).item(),
        "top_30_coverage": top_k_coverage(attention, k_ratio=0.3).item(),
        "top_10_coverage": top_k_coverage(attention, k_ratio=0.1).item(),
        "block_sparsity": block_sparsity(attention, block_size=block_size).item(),
    }
