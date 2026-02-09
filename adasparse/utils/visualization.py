# SPDX-License-Identifier: Apache-2.0

"""Attention pattern visualization utilities."""

from pathlib import Path

import torch


def plot_attention_heatmap(
    attention: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
    save_path: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 10),
):
    """
    Plot attention heatmap for a specific layer and head.

    Parameters
    ----------
    attention : torch.Tensor
        Attention weights (batch, num_heads, q_len, kv_len) or (q_len, kv_len).
    layer_idx : int
        Layer index (if attention is from a list of layers).
    head_idx : int
        Head index to visualize.
    save_path : str, optional
        Path to save the figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if attention.dim() == 4:
        attn = attention[0, head_idx].cpu().float().numpy()
    elif attention.dim() == 2:
        attn = attention.cpu().float().numpy()
    else:
        raise ValueError(f"Unexpected attention shape: {attention.shape}")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(attn, ax=ax, cmap="Blues", vmin=0)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(title or f"Attention Map (Layer {layer_idx}, Head {head_idx})")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_sparsity_comparison(
    metrics_original: dict[str, float],
    metrics_trained: dict[str, float],
    save_path: str | None = None,
):
    """
    Bar chart comparing sparsity metrics between original and trained model.
    """
    import matplotlib.pyplot as plt

    keys = list(metrics_original.keys())
    x = range(len(keys))

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    ax.bar([i - width / 2 for i in x], [metrics_original[k] for k in keys], width, label="Original", alpha=0.8)
    ax.bar([i + width / 2 for i in x], [metrics_trained[k] for k in keys], width, label="Trained", alpha=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Sparsity Metrics: Original vs AdaSparseKV-Trained")
    ax.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_block_importance(
    attention: torch.Tensor,
    block_size: int = 64,
    save_path: str | None = None,
):
    """
    Plot per-block importance (average attention received).
    Useful for visualizing which blocks are safe to evict.
    """
    import matplotlib.pyplot as plt

    # Average over batch, heads, queries
    if attention.dim() == 4:
        received = attention.mean(dim=(0, 1, 2)).cpu().float()  # (kv_len,)
    else:
        received = attention.mean(dim=0).cpu().float()

    kv_len = received.shape[0]
    n_blocks = kv_len // block_size

    if n_blocks == 0:
        return None

    block_importance = received[: n_blocks * block_size].view(n_blocks, block_size).mean(dim=1).numpy()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(n_blocks), block_importance, color="steelblue", alpha=0.8)
    ax.set_xlabel("Block Index")
    ax.set_ylabel("Average Attention Received")
    ax.set_title(f"Per-Block Importance (block_size={block_size})")
    ax.axhline(y=block_importance.mean(), color="red", linestyle="--", label="Mean", alpha=0.7)
    ax.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
