"""
Anchor Token Selection

Defines which tokens should always be retained in the KV cache (never evicted).
These "anchor tokens" provide critical positional and semantic anchors that
stabilize the model's attention patterns even with sparse KV cache.

Anchor types:
- Special tokens: BOS, EOS, PAD, etc.
- Punctuation: periods, commas, etc. (high attention receivers)
- Sink tokens: first K tokens (StreamingLLM observation)
- Recent tokens: last N tokens (recency bias)
"""

import torch
import string
from typing import List, Set, Optional
from dataclasses import dataclass


@dataclass
class AnchorConfig:
    """Configuration for anchor token selection."""
    use_special: bool = True         # Always keep special tokens
    use_punctuation: bool = True     # Keep punctuation marks
    use_sink: bool = True            # Keep first K tokens
    use_recent: bool = True          # Keep last N tokens
    
    sink_size: int = 4               # Number of sink tokens
    recent_size: int = 64            # Number of recent tokens
    
    # Custom punctuation set (can extend beyond ASCII)
    extra_punctuation: str = "。，！？；：、""''【】（）《》——…·"


class AnchorSelector:
    """
    Identifies anchor tokens in a sequence.
    
    Returns a boolean mask where True = anchor token (always keep).
    """
    
    def __init__(self, config: AnchorConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # Build punctuation token set
        self.punctuation_tokens = self._build_punctuation_set()
        
        # Special token IDs
        self.special_token_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, 'all_special_ids') else set()
    
    def _build_punctuation_set(self) -> Set[int]:
        """
        Identify all token IDs that represent punctuation.
        
        Strategy:
        - Iterate through vocabulary
        - Check if decoded token consists only of punctuation characters
        """
        punct_chars = set(string.punctuation + self.config.extra_punctuation)
        punct_tokens = set()
        
        vocab = self.tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            # Clean token (remove special prefixes like Ġ, ▁)
            clean = token_str.replace('Ġ', '').replace('▁', '').replace('Ċ', '\n').strip()
            
            # Check if all characters are punctuation
            if clean and all(c in punct_chars for c in clean):
                punct_tokens.add(token_id)
        
        return punct_tokens
    
    def get_anchor_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute anchor mask for a batch of sequences.
        
        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) optional attention mask (1 = valid, 0 = padding)
        
        Returns:
            anchor_mask: (B, L) boolean tensor, True = anchor token
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        anchor_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        
        # 1. Special tokens
        if self.config.use_special:
            for special_id in self.special_token_ids:
                anchor_mask |= (input_ids == special_id)
        
        # 2. Punctuation
        if self.config.use_punctuation:
            for punct_id in self.punctuation_tokens:
                anchor_mask |= (input_ids == punct_id)
        
        # 3. Sink tokens (first K)
        if self.config.use_sink:
            sink_size = min(self.config.sink_size, L)
            anchor_mask[:, :sink_size] = True
        
        # 4. Recent tokens (last N)
        if self.config.use_recent:
            if attention_mask is not None:
                # Find actual sequence lengths (handle padding)
                seq_lens = attention_mask.sum(dim=1)  # (B,)
                for b in range(B):
                    actual_len = seq_lens[b].item()
                    recent_start = max(0, actual_len - self.config.recent_size)
                    anchor_mask[b, recent_start:actual_len] = True
            else:
                # No padding, use last N positions
                recent_start = max(0, L - self.config.recent_size)
                anchor_mask[:, recent_start:] = True
        
        return anchor_mask
    
    def count_anchors(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> dict:
        """
        Count anchor tokens by type (for analysis/logging).
        
        Returns dict with counts per type.
        """
        anchor_mask = self.get_anchor_mask(input_ids, attention_mask)
        
        counts = {
            "total": anchor_mask.sum().item(),
            "special": 0,
            "punctuation": 0,
            "sink": 0,
            "recent": 0,
        }
        
        B, L = input_ids.shape
        
        # Count each type (note: overlap is possible)
        if self.config.use_special:
            for special_id in self.special_token_ids:
                counts["special"] += (input_ids == special_id).sum().item()
        
        if self.config.use_punctuation:
            for punct_id in self.punctuation_tokens:
                counts["punctuation"] += (input_ids == punct_id).sum().item()
        
        if self.config.use_sink:
            sink_size = min(self.config.sink_size, L)
            counts["sink"] = B * sink_size
        
        if self.config.use_recent:
            if attention_mask is not None:
                seq_lens = attention_mask.sum(dim=1)
                for b in range(B):
                    actual_len = seq_lens[b].item()
                    counts["recent"] += min(self.config.recent_size, actual_len)
            else:
                counts["recent"] = B * min(self.config.recent_size, L)
        
        return counts
    
    def visualize_anchors(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> str:
        """
        Create a visual representation of anchor positions (for debugging).
        
        Returns a string showing tokens with anchor markers.
        """
        anchor_mask = self.get_anchor_mask(input_ids, attention_mask)
        
        # Take first sequence in batch
        tokens = input_ids[0].tolist()
        anchors = anchor_mask[0].tolist()
        
        lines = []
        for i, (tok, is_anchor) in enumerate(zip(tokens, anchors)):
            token_str = self.tokenizer.decode([tok])
            marker = "🔒" if is_anchor else "  "
            lines.append(f"{i:4d} {marker} {token_str}")
        
        return "\n".join(lines)


def build_kv_mask_from_anchors(
    anchor_mask: torch.Tensor,
    keep_ratio: float,
    training: bool = True,
) -> torch.Tensor:
    """
    Build KV retention mask from anchor mask + random sampling.
    
    Strategy:
    - Anchors: always keep (mask = True)
    - Non-anchors: randomly keep a fraction (keep_ratio)
    
    Args:
        anchor_mask: (B, L) boolean mask, True = anchor
        keep_ratio: fraction of non-anchor tokens to keep (0.0 to 1.0)
        training: if True, use random sampling; if False, use deterministic top-k
    
    Returns:
        kv_mask: (B, L) boolean mask for KV cache retention
    """
    B, L = anchor_mask.shape
    device = anchor_mask.device
    
    # Start with anchors
    kv_mask = anchor_mask.clone()
    
    # Identify non-anchor positions
    non_anchor_mask = ~anchor_mask
    
    if training:
        # Random sampling of non-anchors
        random_scores = torch.rand(B, L, device=device)
        random_scores = random_scores * non_anchor_mask.float()  # Zero out anchors
        
        # Keep top keep_ratio fraction of non-anchors
        num_non_anchors = non_anchor_mask.sum(dim=1)  # (B,)
        k_values = (num_non_anchors.float() * keep_ratio).long()  # (B,)
        
        for b in range(B):
            k = k_values[b].item()
            if k > 0:
                # Get top-k non-anchor positions for this sequence
                _, top_indices = torch.topk(random_scores[b], k=min(k, L))
                kv_mask[b, top_indices] = True
    else:
        # Deterministic: keep first keep_ratio fraction of non-anchors
        # (for evaluation consistency)
        num_non_anchors = non_anchor_mask.sum(dim=1)
        k_values = (num_non_anchors.float() * keep_ratio).long()
        
        for b in range(B):
            non_anchor_indices = non_anchor_mask[b].nonzero(as_tuple=True)[0]
            k = min(k_values[b].item(), len(non_anchor_indices))
            if k > 0:
                selected = non_anchor_indices[:k]
                kv_mask[b, selected] = True
    
    return kv_mask
