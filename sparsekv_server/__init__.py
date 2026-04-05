# SPDX-License-Identifier: Apache-2.0

"""
SparseKV: Adaptive Block Dropout Training for Robust KV Cache Eviction

Train LLMs with block-wise KV cache dropout so that inference-time eviction
methods work better. The model learns to be robust to missing KV cache entries.
"""

__version__ = "0.1.0"

from sparsekv.presses.adaptive_dropout_press import AdaptiveBlockDropoutPress
from sparsekv.presses.block_dropout_press import BlockDropoutPress
from sparsekv.presses.eviction_aug_press import EvictionAugPress
from sparsekv.presses.sentence_dropout_press import SentenceDropoutPress
from sparsekv.presses.soft_threshold_press import SoftThresholdPress
from sparsekv.presses.sparse_reg_press import SparseRegPress
from sparsekv.presses.variable_block_press import VariableBlockDropoutPress

__all__ = [
    "BlockDropoutPress",
    "AdaptiveBlockDropoutPress",
    "VariableBlockDropoutPress",
    "SentenceDropoutPress",
    "SoftThresholdPress",
    "EvictionAugPress",
    "SparseRegPress",
]
