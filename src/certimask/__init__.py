"""CertiMask: Runtime-certified low-bit sparse-prefill indexer.

Minimal research prototype. Historical threshold/mean-pooled paths have been
removed. Only current AGLR-C v1 validation and optimized Triton certificate
components remain.
"""

from contextlib import suppress as _suppress

# ---------------------------------------------------------------------------
# AGLR-C reference-first validation pipeline
# ---------------------------------------------------------------------------
from certimask.aglr_certimask import (
    AGLRCertiMaskMetrics,
    AGLRCertiMaskResult,
    aglr_certimask_topk,
    compute_aglr_certimask_metrics,
)

# ---------------------------------------------------------------------------
# AGLR-C v1 reference indexer
# ---------------------------------------------------------------------------
from certimask.aglr_indexer import (
    AGLRMaskResult,
    combine_aglr_scores,
    compute_antidiagonal_block_scores,
)

# ---------------------------------------------------------------------------
# Block summaries and GQA
# ---------------------------------------------------------------------------
from certimask.block_summary import BlockSummaries, expand_kv_heads, mean_pool_qk_blocks

# ---------------------------------------------------------------------------
# Score bounds and validation
# ---------------------------------------------------------------------------
from certimask.bounds import (
    ScoreBounds,
    compute_coordinate_score_bounds,
    compute_group_quantized_coordinate_bounds,
    compute_groupwise_score_bounds,
    compute_k_only_per_group_bounds,
    compute_k_only_per_vector_bounds,
    compute_score_bounds,
    validate_score_bounds,
)

# ---------------------------------------------------------------------------
# Candidate pruning (AGLR-C v2)
# ---------------------------------------------------------------------------
from certimask.candidate_pruning import (
    CandidateMaskResult,
    compute_candidate_antidiagonal_scores,
    compute_teacher_mask_overlap,
    compute_teacher_selected_coverage,
    generate_candidate_mask,
)

# ---------------------------------------------------------------------------
# Block masking utilities
# ---------------------------------------------------------------------------
from certimask.masking import make_block_causal_valid_mask

# ---------------------------------------------------------------------------
# Core quantization
# ---------------------------------------------------------------------------
from certimask.quantization import (
    GroupQuantizedTensor,
    QuantizedTensor,
    quantize_int8_per_group,
    quantize_int8_per_vector,
)

# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------
from certimask.scoring import (
    GroupQuantizedScoreResult,
    KOnlyGroupScoreResult,
    KOnlyScoreResult,
    QuantizedScoreResult,
    group_quantized_int8_scores,
    k_only_per_group_scores,
    k_only_per_vector_scores,
    quantized_int8_scores,
    reference_scores,
)

# ---------------------------------------------------------------------------
# Top-k certificate (reference — slow PyTorch loop)
# ---------------------------------------------------------------------------
from certimask.topk_certificate import (
    AMBIGUOUS,
    DROP,
    INVALID,
    KEEP,
    TopKCertificateResult,
    certified_topk_mask,
    logsumexp_interval,
)

# ---------------------------------------------------------------------------
# Vectorized top-k (optimized)
# ---------------------------------------------------------------------------
from certimask.vectorized_topk import VectorizedTopKMaskResult, vectorized_topk_mask

# ---------------------------------------------------------------------------
# HF extraction (optional, requires transformers)
# ---------------------------------------------------------------------------
with _suppress(ImportError):
    from certimask.hf_extraction import (
        ExtractedQK,
        ExtractedQKV,
        extract_qk_from_qwen2,
        extract_qkv_from_qwen2,
    )

# ---------------------------------------------------------------------------
# Triton accelerated paths (optional, requires CUDA + triton)
# ---------------------------------------------------------------------------
with _suppress(ImportError):
    from certimask.triton_aglr_kernels import triton_aglr_logsumexp_scoring
    from certimask.triton_aglr_ops import (
        TritonAGLRCertiMaskResult,
        compute_fallback_metrics,
        triton_aglr_certimask_logsumexp_g4,
        triton_aglr_certimask_logsumexp_g4_optimized,
    )
    from certimask.triton_topk_certificate import (
        triton_certified_topk_mask_partition,
    )

__all__ = [
    # Core quantization
    "GroupQuantizedTensor",
    "QuantizedTensor",
    "quantize_int8_per_group",
    "quantize_int8_per_vector",
    # Core scoring
    "GroupQuantizedScoreResult",
    "KOnlyGroupScoreResult",
    "KOnlyScoreResult",
    "QuantizedScoreResult",
    "group_quantized_int8_scores",
    "k_only_per_group_scores",
    "k_only_per_vector_scores",
    "quantized_int8_scores",
    "reference_scores",
    # Score bounds
    "ScoreBounds",
    "compute_coordinate_score_bounds",
    "compute_group_quantized_coordinate_bounds",
    "compute_groupwise_score_bounds",
    "compute_k_only_per_group_bounds",
    "compute_k_only_per_vector_bounds",
    "compute_score_bounds",
    "validate_score_bounds",
    # AGLR-C v1 reference indexer
    "AGLRMaskResult",
    "combine_aglr_scores",
    "compute_antidiagonal_block_scores",
    # AGLR-C reference-first validation
    "AGLRCertiMaskMetrics",
    "AGLRCertiMaskResult",
    "aglr_certimask_topk",
    "compute_aglr_certimask_metrics",
    # Top-k certificate (reference — slow PyTorch loop)
    "AMBIGUOUS",
    "DROP",
    "INVALID",
    "KEEP",
    "TopKCertificateResult",
    "certified_topk_mask",
    "logsumexp_interval",
    # Block masking
    "make_block_causal_valid_mask",
    # Candidate pruning (AGLR-C v2)
    "CandidateMaskResult",
    "compute_candidate_antidiagonal_scores",
    "compute_teacher_mask_overlap",
    "compute_teacher_selected_coverage",
    "generate_candidate_mask",
    # Block summaries
    "BlockSummaries",
    "expand_kv_heads",
    "mean_pool_qk_blocks",
    # Vectorized top-k
    "VectorizedTopKMaskResult",
    "vectorized_topk_mask",
    # HF extraction (optional)
    "ExtractedQK",
    "ExtractedQKV",
    "extract_qk_from_qwen2",
    "extract_qkv_from_qwen2",
    # Triton accelerated paths (optional)
    "TritonAGLRCertiMaskResult",
    "compute_fallback_metrics",
    "triton_aglr_certimask_logsumexp_g4",
    "triton_aglr_certimask_logsumexp_g4_optimized",
    "triton_aglr_logsumexp_scoring",
    "triton_certified_topk_mask_partition",
]
