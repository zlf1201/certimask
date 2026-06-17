"""CertiMask: Certified masking for attention with INT8 quantization."""

from contextlib import suppress as _suppress

from certimask.aglr_certimask import (
    AGLRCertiMaskMetrics,
    AGLRCertiMaskResult,
    aglr_certimask_topk,
    compute_aglr_certimask_metrics,
)
from certimask.aglr_indexer import (
    AGLRMaskResult,
    BlockLandmarks,
    aglr_adaptive_mass_budget_mask,
    aglr_local_plus_landmark_mask,
    combine_aglr_scores,
    compute_antidiagonal_block_scores,
    compute_landmark_block_scores,
    select_block_landmarks,
)
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
    compute_diagnostic_quantiles,
    compute_per_tile_diagnostics,
    compute_refinement_decomposition,
    compute_row_subset_stats,
)
from certimask.masking import (
    CertiMaskDecision,
    CertiMaskResult,
    certified_threshold_mask,
    make_block_causal_valid_mask,
    naive_quantized_mask,
    reference_mask,
    thresholds_for_target_sparsity,
)
from certimask.metrics import (
    BoundMetrics,
    MaskMetrics,
    compute_bound_metrics,
    compute_mask_metrics,
)
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
from certimask.topk_certificate import (
    AMBIGUOUS,
    DROP,
    INVALID,
    KEEP,
    TopKCertificateResult,
    certified_topk_mask,
    logsumexp_interval,
)

# HF extraction is optional
with _suppress(ImportError):
    from certimask.hf_extraction import (
        ExtractedQK,
        ExtractedQKV,
        extract_qk_from_qwen2,
        extract_qkv_from_qwen2,
    )

# Vectorized top-k (always available)
from certimask.vectorized_topk import VectorizedTopKMaskResult, vectorized_topk_mask

# Triton kernels are optional (requires CUDA + triton)
with _suppress(ImportError):
    from certimask.triton_aglr_kernels import triton_aglr_logsumexp_scoring
    from certimask.triton_aglr_ops import (
        TritonAGLRCertiMaskResult,
        compute_fallback_metrics,
        triton_aglr_certimask_logsumexp_g4,
    )
    from certimask.triton_topk_certificate import triton_certified_topk_mask_partition

__all__ = [
    "AGLRCertiMaskMetrics",
    "AGLRCertiMaskResult",
    "AGLRMaskResult",
    "AMBIGUOUS",
    "VectorizedTopKMaskResult",
    "AttentionQualityMetrics",
    "AttentionReconstructionDiagnostics",
    "BenefitProxyMetrics",
    "BlockLandmarks",
    "BlockSummaries",
    "BoundMetrics",
    "CertiMaskDecision",
    "CertiMaskResult",
    "DROP",
    "DiagnosticQuantiles",
    "ExtractedQK",
    "ExtractedQKV",
    "GroupQuantizedScoreResult",
    "GroupQuantizedTensor",
    "INVALID",
    "KEEP",
    "KOnlyGroupScoreResult",
    "KOnlyScoreResult",
    "MaskMetrics",
    "PerTileDiagnostics",
    "QuantizedScoreResult",
    "QuantizedTensor",
    "RefinementDecomposition",
    "ScoreBounds",
    "TopKCertificateResult",
    "TritonAGLRCertiMaskResult",
    "aglr_adaptive_mass_budget_mask",
    "aglr_certimask_topk",
    "aglr_local_plus_landmark_mask",
    "block_sparse_attention_output",
    "certified_threshold_mask",
    "certified_topk_mask",
    "combine_aglr_scores",
    "compute_aglr_certimask_metrics",
    "compute_fallback_metrics",
    "compute_antidiagonal_block_scores",
    "compute_landmark_block_scores",
    "compute_attention_quality",
    "compute_benefit_proxy",
    "compute_oracle_block_mass_scores",
    "compute_bound_metrics",
    "compute_coordinate_score_bounds",
    "compute_diagnostic_quantiles",
    "compute_group_quantized_coordinate_bounds",
    "compute_groupwise_score_bounds",
    "compute_k_only_per_group_bounds",
    "compute_k_only_per_vector_bounds",
    "compute_mask_metrics",
    "compute_per_tile_diagnostics",
    "compute_refinement_decomposition",
    "compute_row_subset_stats",
    "compute_score_bounds",
    "dense_attention_output",
    "expand_block_mask_to_token_mask",
    "expand_kv_heads",
    "local_plus_extra_mask",
    "extract_qk_from_qwen2",
    "extract_qkv_from_qwen2",
    "generate_synthetic_summaries",
    "group_quantized_int8_scores",
    "local_window_block_mask",
    "logsumexp_interval",
    "k_only_per_group_scores",
    "k_only_per_vector_scores",
    "make_block_causal_valid_mask",
    "mean_pool_qk_blocks",
    "naive_quantized_mask",
    "oracle_block_mass_mask",
    "quantize_int8_per_group",
    "quantize_int8_per_vector",
    "quantized_int8_scores",
    "random_valid_block_mask",
    "reference_mask",
    "select_block_landmarks",
    "reference_scores",
    "thresholds_for_target_sparsity",
    "triton_aglr_certimask_logsumexp_g4",
    "triton_aglr_logsumexp_scoring",
    "triton_certified_topk_mask_partition",
    "validate_score_bounds",
    "vectorized_topk_mask",
]
