"""CertiMask: Certified masking for attention with INT8 quantization."""

from contextlib import suppress as _suppress

# ---------------------------------------------------------------------------
# AGLR-C CertiMask pipeline (reference-first validation path)
# ---------------------------------------------------------------------------
from certimask.aglr_certimask import (
    AGLRCertiMaskMetrics,
    AGLRCertiMaskResult,
    aglr_certimask_topk,  # reference-first validation; not deployable online
    compute_aglr_certimask_metrics,
)

# ---------------------------------------------------------------------------
# AGLR-C reference indexer
# ---------------------------------------------------------------------------
from certimask.aglr_indexer import (
    AGLRMaskResult,
    BlockLandmarks,
    aglr_adaptive_mass_budget_mask,  # historical: experimental, not used in active benchmarks
    aglr_local_plus_landmark_mask,  # historical: slow loop-based; use vectorized_topk_mask
    combine_aglr_scores,
    compute_antidiagonal_block_scores,
    compute_landmark_block_scores,
    select_block_landmarks,
)

# ---------------------------------------------------------------------------
# Attention quality and diagnostics
# ---------------------------------------------------------------------------
from certimask.attention_quality import (
    AttentionQualityMetrics,
    BenefitProxyMetrics,
    block_sparse_attention_output,
    compute_attention_quality,
    compute_benefit_proxy,
    compute_oracle_block_mass_scores,
    dense_attention_output,
    expand_block_mask_to_token_mask,
    local_window_block_mask,
    oracle_block_mass_mask,
    random_valid_block_mask,
)
from certimask.block_summary import BlockSummaries, expand_kv_heads, mean_pool_qk_blocks
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
from certimask.diagnostics import (
    AttentionReconstructionDiagnostics,
    DiagnosticQuantiles,
    PerTileDiagnostics,
    RefinementDecomposition,
    compute_diagnostic_quantiles,  # diagnostic helper
    compute_per_tile_diagnostics,  # diagnostic helper
    compute_refinement_decomposition,  # diagnostic helper
    compute_row_subset_stats,  # diagnostic helper
)

# ---------------------------------------------------------------------------
# Masking and threshold certificate (historical baseline)
# ---------------------------------------------------------------------------
from certimask.masking import (
    CertiMaskDecision,
    CertiMaskResult,
    certified_threshold_mask,  # historical: threshold-based baseline, superseded by AGLR-C
    make_block_causal_valid_mask,
    naive_quantized_mask,  # historical: naive quantized mask
    reference_mask,  # historical: reference mask for comparison
    thresholds_for_target_sparsity,
)
from certimask.metrics import (
    BoundMetrics,
    MaskMetrics,
    compute_bound_metrics,
    compute_mask_metrics,
)

# ---------------------------------------------------------------------------
# Core quantization and bounds
# ---------------------------------------------------------------------------
from certimask.quantization import (
    GroupQuantizedTensor,
    QuantizedTensor,
    quantize_int8_per_group,
    quantize_int8_per_vector,
)
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
from certimask.synthetic import generate_synthetic_summaries

# ---------------------------------------------------------------------------
# Top-k certificate
# ---------------------------------------------------------------------------
from certimask.topk_certificate import (
    AMBIGUOUS,
    DROP,
    INVALID,
    KEEP,
    TopKCertificateResult,
    certified_topk_mask,  # historical: slow PyTorch loop; use triton_certified_topk_mask_partition
    logsumexp_interval,
)

# ---------------------------------------------------------------------------
# Vectorized top-k (replaced loop-based implementation)
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
        triton_aglr_certimask_logsumexp_g4,  # main Triton pipeline (uses PyTorch cert internally)
    )
    from certimask.triton_topk_certificate import (
        triton_certified_topk_mask_partition,  # optimized cert
    )

__all__ = [
    # Core quantization and bounds
    "GroupQuantizedTensor",
    "QuantizedTensor",
    "quantize_int8_per_group",
    "quantize_int8_per_vector",
    "ScoreBounds",
    "compute_coordinate_score_bounds",
    "compute_group_quantized_coordinate_bounds",
    "compute_groupwise_score_bounds",
    "compute_k_only_per_group_bounds",
    "compute_k_only_per_vector_bounds",
    "compute_score_bounds",
    "validate_score_bounds",
    "GroupQuantizedScoreResult",
    "KOnlyGroupScoreResult",
    "KOnlyScoreResult",
    "QuantizedScoreResult",
    "group_quantized_int8_scores",
    "k_only_per_group_scores",
    "k_only_per_vector_scores",
    "quantized_int8_scores",
    "reference_scores",
    # Masking and threshold certificate (historical baseline)
    "CertiMaskDecision",
    "CertiMaskResult",
    "certified_threshold_mask",  # historical
    "make_block_causal_valid_mask",
    "naive_quantized_mask",  # historical
    "reference_mask",  # historical
    "thresholds_for_target_sparsity",
    # AGLR-C reference indexer
    "AGLRMaskResult",
    "BlockLandmarks",
    "aglr_adaptive_mass_budget_mask",  # historical
    "aglr_local_plus_landmark_mask",  # historical: slow loop
    "combine_aglr_scores",
    "compute_antidiagonal_block_scores",
    "compute_landmark_block_scores",
    "select_block_landmarks",
    "BlockSummaries",
    "expand_kv_heads",
    "mean_pool_qk_blocks",
    # Top-k certificate
    "AMBIGUOUS",
    "DROP",
    "INVALID",
    "KEEP",
    "TopKCertificateResult",
    "certified_topk_mask",  # historical: slow PyTorch loop
    "logsumexp_interval",
    # AGLR-C CertiMask pipeline (reference-first validation)
    "AGLRCertiMaskMetrics",
    "AGLRCertiMaskResult",
    "aglr_certimask_topk",  # reference-first validation, not deployable
    "compute_aglr_certimask_metrics",
    # Attention quality and diagnostics
    "AttentionQualityMetrics",
    "BenefitProxyMetrics",
    "block_sparse_attention_output",
    "compute_attention_quality",
    "compute_benefit_proxy",
    "compute_oracle_block_mass_scores",
    "dense_attention_output",
    "expand_block_mask_to_token_mask",
    "local_window_block_mask",
    "oracle_block_mass_mask",
    "random_valid_block_mask",
    "AttentionReconstructionDiagnostics",
    "DiagnosticQuantiles",
    "PerTileDiagnostics",
    "RefinementDecomposition",
    "compute_diagnostic_quantiles",  # diagnostic helper
    "compute_per_tile_diagnostics",  # diagnostic helper
    "compute_refinement_decomposition",  # diagnostic helper
    "compute_row_subset_stats",  # diagnostic helper
    "BoundMetrics",
    "MaskMetrics",
    "compute_bound_metrics",
    "compute_mask_metrics",
    "generate_synthetic_summaries",
    # Vectorized top-k (optimized)
    "VectorizedTopKMaskResult",
    "vectorized_topk_mask",
    # HF extraction (optional)
    "ExtractedQK",
    "ExtractedQKV",
    "extract_qk_from_qwen2",
    "extract_qkv_from_qwen2",
    # Triton accelerated paths (optional, requires CUDA + triton)
    "TritonAGLRCertiMaskResult",
    "compute_fallback_metrics",
    "triton_aglr_certimask_logsumexp_g4",
    "triton_aglr_logsumexp_scoring",
    "triton_certified_topk_mask_partition",
]
