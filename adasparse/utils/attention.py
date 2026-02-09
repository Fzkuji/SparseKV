# SPDX-License-Identifier: Apache-2.0

"""Attention analysis utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


@torch.no_grad()
def extract_attention_maps(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Extract attention maps from all layers.

    Returns list of tensors, each (batch, num_heads, q_len, kv_len).
    """
    model.eval()
    outputs = model(input_ids, output_attentions=True)
    return list(outputs.attentions)


@torch.no_grad()
def compare_attention_maps(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    input_ids: torch.Tensor,
) -> dict[str, float]:
    """
    Compare attention patterns between two models (e.g., original vs trained).

    Returns per-layer cosine similarity and KL divergence.
    """
    attns_a = extract_attention_maps(model_a, input_ids)
    attns_b = extract_attention_maps(model_b, input_ids)

    n_layers = min(len(attns_a), len(attns_b))
    cosine_sims = []
    kl_divs = []

    for i in range(n_layers):
        a = attns_a[i].flatten(0, 2)  # (B*H*Q, kv_len)
        b = attns_b[i].flatten(0, 2)

        # Cosine similarity
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
        cosine_sims.append(cos.item())

        # KL divergence
        eps = 1e-8
        kl = (a * ((a + eps).log() - (b + eps).log())).sum(dim=-1).mean()
        kl_divs.append(kl.item())

    return {
        "mean_cosine_similarity": sum(cosine_sims) / len(cosine_sims),
        "mean_kl_divergence": sum(kl_divs) / len(kl_divs),
        "per_layer_cosine": cosine_sims,
        "per_layer_kl": kl_divs,
    }
