#!/usr/bin/env python3
"""
Analyze attention patterns of a pretrained model.

Discovers:
1. Which TOKEN TYPES naturally receive more attention (punctuation, stopwords, etc.)
2. Which POSITIONS naturally receive more attention (sink, recent, periodic, etc.)
3. Per-head patterns: which heads attend to which token types

Usage:
    python scripts/analyze_attention.py --model Qwen/Qwen3-8B --num_samples 100
    python scripts/analyze_attention.py --model meta-llama/Llama-3.1-8B-Instruct --num_samples 100
"""

import argparse
import json
import logging
import os
import string
import sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Token Type Classification
# ============================================================

def build_token_type_map(tokenizer):
    """
    Classify every token in the vocabulary into types.
    Returns a dict: token_id -> token_type
    """
    type_map = {}
    vocab = tokenizer.get_vocab()
    
    # Define punctuation set (broad)
    punct_chars = set(string.punctuation + '。，！？；：、""''【】（）《》——…·')
    
    for token_str, token_id in vocab.items():
        # Clean token string (remove special prefixes like Ġ, ▁, etc.)
        clean = token_str.replace('Ġ', '').replace('▁', '').replace('Ċ', '\n').strip()
        
        if token_id in tokenizer.all_special_ids:
            type_map[token_id] = 'special'
        elif clean == '' or clean == '\n' or clean == '\t':
            type_map[token_id] = 'whitespace'
        elif all(c in punct_chars for c in clean):
            type_map[token_id] = 'punctuation'
        elif clean.isdigit():
            type_map[token_id] = 'number'
        elif clean.lower() in {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by', 'about',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
            'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
            'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
            'de', 'la', 'le', 'les', 'un', 'une', 'des', 'du', 'et',  # French
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',  # Chinese
            '他', '这', '中', '大', '为', '上', '个', '国', '说', '到',
        }:
            type_map[token_id] = 'stopword'
        else:
            type_map[token_id] = 'content'
    
    return type_map


# ============================================================
# Attention Analysis
# ============================================================

@torch.no_grad()
def analyze_single_sample(
    model,
    tokenizer,
    text: str,
    token_type_map: dict,
    max_len: int = 2048,
) -> dict:
    """
    Analyze attention patterns for a single text sample.
    
    Returns per-layer, per-head statistics about attention distribution
    across token types and positions.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    L = input_ids.shape[1]
    
    if L < 64:
        return None
    
    # Forward with attention weights
    outputs = model(input_ids, output_attentions=True)
    attentions = outputs.attentions  # tuple of (1, H, L, L) per layer
    
    # Get token types for this sequence
    token_ids = input_ids[0].cpu().tolist()
    token_types = [token_type_map.get(tid, 'content') for tid in token_ids]
    
    result = {
        "seq_len": L,
        "token_type_counts": {},
        "layers": [],
    }
    
    # Count token types
    for tt in set(token_types):
        result["token_type_counts"][tt] = token_types.count(tt)
    
    for layer_idx, attn in enumerate(attentions):
        attn = attn[0]  # (H, L, L)
        H = attn.shape[0]
        
        # Importance score: how much attention each KV position receives
        # Sum over all query positions
        importance = attn.sum(dim=1)  # (H, L) — total attention received per KV position
        
        layer_stats = {
            "layer": layer_idx,
            "heads": [],
        }
        
        for h in range(H):
            head_importance = importance[h].cpu().float().numpy()  # (L,)
            
            # --- Token Type Analysis ---
            type_attention = defaultdict(float)
            type_count = defaultdict(int)
            for pos, tt in enumerate(token_types):
                type_attention[tt] += head_importance[pos]
                type_count[tt] += 1
            
            # Normalize: average attention per token of each type
            type_avg_attention = {}
            for tt in type_attention:
                if type_count[tt] > 0:
                    type_avg_attention[tt] = type_attention[tt] / type_count[tt]
            
            # Fraction of total attention going to each type
            total_attn = head_importance.sum()
            type_fraction = {}
            for tt in type_attention:
                type_fraction[tt] = type_attention[tt] / (total_attn + 1e-10)
            
            # --- Position Analysis ---
            # Bin positions: first 4 (sink), 5-10%, 10-50%, 50-90%, last 10%
            position_bins = {
                "sink_0_4": head_importance[:4].mean() if L >= 4 else 0,
                "early_5pct": head_importance[4:max(int(L*0.05), 5)].mean() if L > 5 else 0,
                "first_10pct": head_importance[:int(L*0.1)].mean(),
                "middle": head_importance[int(L*0.1):int(L*0.9)].mean(),
                "last_10pct": head_importance[int(L*0.9):].mean(),
                "last_64": head_importance[-min(64, L):].mean(),
            }
            
            # Entropy of attention distribution (per query, averaged)
            attn_h = attn[h]  # (L, L)
            entropy = -(attn_h * (attn_h + 1e-10).log()).sum(dim=-1).mean().item()
            
            # Concentration: what fraction of tokens capture 80% of attention
            sorted_importance = np.sort(head_importance)[::-1]
            cumsum = np.cumsum(sorted_importance)
            threshold_80 = np.searchsorted(cumsum, total_attn * 0.8) + 1
            concentration_80 = threshold_80 / L  # fraction of tokens for 80% attention
            
            head_stats = {
                "head": h,
                "type_avg_attention": {k: float(v) for k, v in type_avg_attention.items()},
                "type_fraction": {k: float(v) for k, v in type_fraction.items()},
                "position_bins": {k: float(v) for k, v in position_bins.items()},
                "entropy": entropy,
                "concentration_80": float(concentration_80),
            }
            layer_stats["heads"].append(head_stats)
        
        result["layers"].append(layer_stats)
    
    return result


def aggregate_results(all_results: list) -> dict:
    """Aggregate analysis across multiple samples."""
    num_layers = len(all_results[0]["layers"])
    num_heads = len(all_results[0]["layers"][0]["heads"])
    
    agg = {
        "num_samples": len(all_results),
        "avg_seq_len": np.mean([r["seq_len"] for r in all_results]),
        "token_type_analysis": {},
        "position_analysis": {},
        "concentration_analysis": {},
        "per_layer_per_head": [],
    }
    
    # Aggregate per token type
    all_types = set()
    for r in all_results:
        for layer in r["layers"]:
            for head in layer["heads"]:
                all_types.update(head["type_avg_attention"].keys())
    
    for tt in all_types:
        values = []
        fractions = []
        for r in all_results:
            for layer in r["layers"]:
                for head in layer["heads"]:
                    if tt in head["type_avg_attention"]:
                        values.append(head["type_avg_attention"][tt])
                    if tt in head["type_fraction"]:
                        fractions.append(head["type_fraction"][tt])
        agg["token_type_analysis"][tt] = {
            "avg_attention_per_token": float(np.mean(values)) if values else 0,
            "std": float(np.std(values)) if values else 0,
            "avg_fraction_of_total": float(np.mean(fractions)) if fractions else 0,
        }
    
    # Aggregate per position bin
    position_keys = ["sink_0_4", "early_5pct", "first_10pct", "middle", "last_10pct", "last_64"]
    for pk in position_keys:
        values = []
        for r in all_results:
            for layer in r["layers"]:
                for head in layer["heads"]:
                    values.append(head["position_bins"][pk])
        agg["position_analysis"][pk] = {
            "avg_importance": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    
    # Aggregate concentration (per layer)
    for l in range(num_layers):
        layer_data = {"layer": l, "heads": []}
        for h in range(num_heads):
            concentrations = []
            entropies = []
            type_fractions = defaultdict(list)
            
            for r in all_results:
                head = r["layers"][l]["heads"][h]
                concentrations.append(head["concentration_80"])
                entropies.append(head["entropy"])
                for tt, frac in head["type_fraction"].items():
                    type_fractions[tt].append(frac)
            
            head_data = {
                "head": h,
                "avg_concentration_80": float(np.mean(concentrations)),
                "avg_entropy": float(np.mean(entropies)),
                "type_fractions": {
                    tt: float(np.mean(v)) for tt, v in type_fractions.items()
                },
            }
            layer_data["heads"].append(head_data)
        agg["per_layer_per_head"].append(layer_data)
    
    return agg


def print_summary(agg: dict):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print(f"ATTENTION PATTERN ANALYSIS ({agg['num_samples']} samples, avg len {agg['avg_seq_len']:.0f})")
    print("=" * 70)
    
    # Token type ranking
    print("\n📊 TOKEN TYPE ANALYSIS (avg attention per token, higher = more attended)")
    print("-" * 50)
    types_sorted = sorted(
        agg["token_type_analysis"].items(),
        key=lambda x: x[1]["avg_attention_per_token"],
        reverse=True,
    )
    for tt, stats in types_sorted:
        bar = "█" * int(stats["avg_attention_per_token"] * 20)
        print(f"  {tt:15s} | {stats['avg_attention_per_token']:8.4f} ± {stats['std']:.4f} | frac: {stats['avg_fraction_of_total']:.3f} | {bar}")
    
    # Position analysis
    print("\n📍 POSITION ANALYSIS (avg importance)")
    print("-" * 50)
    for pk, stats in agg["position_analysis"].items():
        bar = "█" * int(stats["avg_importance"] * 20)
        print(f"  {pk:15s} | {stats['avg_importance']:8.4f} ± {stats['std']:.4f} | {bar}")
    
    # Concentration
    print("\n🎯 ATTENTION CONCENTRATION (fraction of tokens for 80% attention)")
    print("-" * 50)
    all_conc = []
    for layer_data in agg["per_layer_per_head"]:
        for head in layer_data["heads"]:
            all_conc.append(head["avg_concentration_80"])
    print(f"  Overall: {np.mean(all_conc):.3f} (= {np.mean(all_conc)*100:.1f}% of tokens carry 80% of attention)")
    
    # Per-layer concentration
    print("\n  Per-layer avg concentration:")
    for layer_data in agg["per_layer_per_head"]:
        l = layer_data["layer"]
        concs = [h["avg_concentration_80"] for h in layer_data["heads"]]
        avg_c = np.mean(concs)
        bar = "█" * int(avg_c * 40)
        print(f"    Layer {l:2d}: {avg_c:.3f} | {bar}")
    
    # Most concentrated heads (potential anchor heads)
    print("\n🔑 MOST CONCENTRATED HEADS (lowest concentration = most focused)")
    print("-" * 50)
    all_heads = []
    for layer_data in agg["per_layer_per_head"]:
        for head in layer_data["heads"]:
            all_heads.append({
                "layer": layer_data["layer"],
                "head": head["head"],
                "concentration": head["avg_concentration_80"],
                "entropy": head["avg_entropy"],
                "type_fractions": head["type_fractions"],
            })
    
    all_heads.sort(key=lambda x: x["concentration"])
    for h in all_heads[:20]:
        top_type = max(h["type_fractions"].items(), key=lambda x: x[1])
        print(f"  Layer {h['layer']:2d} Head {h['head']:2d}: "
              f"conc={h['concentration']:.3f}, entropy={h['entropy']:.2f}, "
              f"top_type={top_type[0]}({top_type[1]:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./analysis")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_subset", type=str, default="sample-10BT")
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",  # Need attention weights
        device_map="auto",
    )
    model.eval()
    
    # Build token type map
    logger.info("Building token type map...")
    token_type_map = build_token_type_map(tokenizer)
    type_counts = defaultdict(int)
    for tt in token_type_map.values():
        type_counts[tt] += 1
    logger.info(f"Token types in vocab: {dict(type_counts)}")
    
    # Load data
    logger.info(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, args.dataset_subset, split="train", streaming=True)
    
    # Analyze samples
    all_results = []
    for i, example in enumerate(ds):
        if i >= args.num_samples * 2:
            break
        
        text = example.get("text", "")
        if len(text) < 200:
            continue
        
        logger.info(f"Analyzing sample {len(all_results)+1}/{args.num_samples} (len={len(text[:args.max_len*4])})")
        
        try:
            result = analyze_single_sample(model, tokenizer, text, token_type_map, args.max_len)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.warning(f"Error on sample {i}: {e}")
            continue
        
        if len(all_results) >= args.num_samples:
            break
    
    if not all_results:
        logger.error("No valid results!")
        return
    
    # Aggregate
    logger.info("Aggregating results...")
    agg = aggregate_results(all_results)
    
    # Save
    model_name = args.model.replace("/", "--")
    output_file = output_dir / f"attention_analysis_{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(agg, f, indent=2)
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    print_summary(agg)


if __name__ == "__main__":
    main()
