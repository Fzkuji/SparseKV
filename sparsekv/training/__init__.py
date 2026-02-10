"""
SparseKV Training Module

Provides tools for training LLMs with anchor-based KV cache dropout (EIT).
"""

from sparsekv.training.anchor import (
    AnchorConfig,
    AnchorSelector,
    build_kv_mask_from_anchors,
)
from sparsekv.training.kv_dropout import (
    KVDropoutContext,
    SDPAKVDropout,
    apply_kv_dropout,
    apply_kv_dropout_sdpa,
)
from sparsekv.training.eit_trainer import (
    EITConfig,
    EITTrainer,
)
from sparsekv.training.scheduler import (
    SchedulerConfig,
    CompressionScheduler,
)

__all__ = [
    # Anchor tokens
    "AnchorConfig",
    "AnchorSelector",
    "build_kv_mask_from_anchors",
    
    # KV dropout
    "KVDropoutContext",
    "SDPAKVDropout",
    "apply_kv_dropout",
    "apply_kv_dropout_sdpa",
    
    # Training
    "EITConfig",
    "EITTrainer",
    
    # Scheduling
    "SchedulerConfig",
    "CompressionScheduler",
]
