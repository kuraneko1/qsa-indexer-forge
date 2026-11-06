"""Reference implementation of Qwen Sparse Attention (QSA) for adaptation studies.

The design follows "On the Design of Qwen3.8-Next Architecture" (arXiv 2608.30320),
Sec. 2.1.2: a compressed MQA indexer scores mean-pooled key blocks, the top-k
blocks plus the current (incomplete) block form the token set for core attention.

This module is deliberately small and dependency-light (torch only) so it can be
unit-tested on CPU and reused as the training-side reference in later stages.
"""

from .indexer import QSAIndexer, apply_partial_rope, rms_norm
from .sparse_attention import (
    attention_mass_capture,
    build_sparse_mask,
    qsa_attention,
    select_blocks,
)
from .losses import (indexer_kl_from_target, indexer_kl_loss, indexer_kl_per_query,
                     pool_teacher_distribution)
from .teacher import block_teacher_from_qk

__all__ = [
    "QSAIndexer",
    "apply_partial_rope",
    "rms_norm",
    "attention_mass_capture",
    "build_sparse_mask",
    "qsa_attention",
    "select_blocks",
    "indexer_kl_loss",
    "indexer_kl_per_query",
    "indexer_kl_from_target",
    "block_teacher_from_qk",
    "pool_teacher_distribution",
]
