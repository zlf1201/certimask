"""High-level Triton AGLR-C CertiMask operations.

Provides ``triton_aglr_certimask_logsumexp_g4`` — the main entry point that
combines Triton scoring with the existing PyTorch partition certificate.
Also provides ``compute_fallback_metrics`` for certificate fallback analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from certimask.quantization import quantize_int8_per_group
from certimask.topk_certificate import AMBIGUOUS, DROP, KEEP, certified_topk_mask


@dataclass
class TritonAGLRCertiMaskResult:
    """Result of Triton AGLR CertiMask scoring + certification.

    Attributes:
        quantized_scores: Quantized logsumexp block scores [B, H, Q_blk, K_blk].
        lower_scores: Lower bound logsumexp scores [B, H, Q_blk, K_blk].
        upper_scores: Upper bound logsumexp scores [B, H, Q_blk, K_blk].
        decisions: Per-tile decision codes [B, H, Q_blk, K_blk].
        ambiguous: Boolean ambiguous-tile mask [B, H, Q_blk, K_blk].
        mask: Final boolean block mask [B, H, Q_blk, K_blk].
        reference_mask: FP32 AGLR reference mask [B, H, Q_blk, K_blk].
        exact_match: Whether mask equals reference_mask.
        mismatch_count: Number of mismatched tiles.
    """

    quantized_scores: torch.Tensor
    lower_scores: torch.Tensor
    upper_scores: torch.Tensor
    decisions: torch.Tensor
    ambiguous: torch.Tensor
    mask: torch.Tensor
    reference_mask: torch.Tensor
    exact_match: bool
    mismatch_count: int


def compute_fallback_metrics(
    result: TritonAGLRCertiMaskResult,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute certificate fallback metrics from a Triton result.

    These metrics come from the partition certificate decisions,
    not from score equality between Triton and PyTorch.

    Args:
        result: TritonAGLRCertiMaskResult from triton_aglr_certimask_logsumexp_g4.
        valid_mask: Boolean causal valid mask [B, H, Q_blk, K_blk].

    Returns:
        Dict with fallback_rate, ambiguous_rate, row_certification_rate,
        certified_keep_rate, certified_drop_rate.
    """
    decisions = result.decisions
    valid = valid_mask

    valid_decisions = decisions[valid]
    total_valid = int(valid_decisions.numel())

    if total_valid == 0:
        return {
            "fallback_rate": 0.0,
            "ambiguous_rate": 0.0,
            "row_certification_rate": 0.0,
            "certified_keep_rate": 0.0,
            "certified_drop_rate": 0.0,
        }

    # Fallback = tiles that needed FP reference (AMBIGUOUS)
    ambiguous_count = int((valid_decisions == AMBIGUOUS).sum().item())
    fallback_rate = ambiguous_count / total_valid
    ambiguous_rate = fallback_rate  # same concept here

    # Row certification: rows with zero ambiguous tiles
    ambig_per_row = (result.ambiguous & valid).any(dim=-1)  # [B, H, Q_blk]
    total_rows = ambig_per_row.numel()
    certified_rows = int((~ambig_per_row).sum().item())
    row_certification_rate = certified_rows / total_rows if total_rows > 0 else 0.0

    # Certified KEEP/DROP rates (among valid tiles)
    keep_count = int((valid_decisions == KEEP).sum().item())
    drop_count = int((valid_decisions == DROP).sum().item())
    certifiable = keep_count + drop_count
    certified_keep_rate = keep_count / certifiable if certifiable > 0 else 0.0
    certified_drop_rate = drop_count / certifiable if certifiable > 0 else 0.0

    return {
        "fallback_rate": fallback_rate,
        "ambiguous_rate": ambiguous_rate,
        "row_certification_rate": row_certification_rate,
        "certified_keep_rate": certified_keep_rate,
        "certified_drop_rate": certified_drop_rate,
    }


def _expand_group_tensor(
    grouped: torch.Tensor,
    group_size: int,
    target_dim: int,
) -> torch.Tensor:
    """Expand a per-group tensor to per-coordinate by repeating.

    Args:
        grouped: [..., num_groups] tensor.
        group_size: Number of coordinates per group.
        target_dim: Target size of the last dimension (= D).

    Returns:
        Expanded [..., D] tensor.
    """
    # grouped: [..., G]
    # Repeat each group value group_size times along last dim
    expanded = grouped.repeat_interleave(group_size, dim=-1)
    # Trim to target_dim (handles last group being shorter)
    return expanded[..., :target_dim]


def _compute_reference_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    block_size: int,
    sample_pattern: str,
    aggregation: str,
    scale_by_sqrt_dim: bool,
) -> torch.Tensor:
    """Compute FP32 reference AGLR block scores."""
    from certimask.aglr_indexer import compute_antidiagonal_block_scores

    return compute_antidiagonal_block_scores(
        query,
        key,
        block_size=block_size,
        sample_pattern=sample_pattern,
        aggregation=aggregation,
        valid_mask=valid_mask,
        scale_by_sqrt_dim=scale_by_sqrt_dim,
    )


def _compute_reference_mask(
    fp_scores: torch.Tensor,
    *,
    target_sparsity: float,
    local_blocks: int,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the FP32 reference AGLR mask using loop-based approach."""
    from certimask.aglr_indexer import aglr_local_plus_landmark_mask

    result = aglr_local_plus_landmark_mask(
        fp_scores,
        target_sparsity=target_sparsity,
        local_blocks=local_blocks,
        valid_mask=valid_mask,
    )
    return result.mask


def _compute_reference_mask_vectorized(
    fp_scores: torch.Tensor,
    *,
    target_sparsity: float,
    local_blocks: int,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the FP32 reference AGLR mask using vectorized top-k.

    This avoids Python row loops and .item() CPU-GPU sync points.
    Currently supports local_blocks=0 only; falls back to loop for local_blocks>0.
    """
    from certimask.vectorized_topk import vectorized_topk_mask

    batch, heads, q_blk, k_blk = fp_scores.shape
    device = fp_scores.device

    if local_blocks > 0:
        # For local_blocks > 0, build mandatory keep mask
        # Local blocks: for each query row q, keep blocks [max(0, q-local_blocks+1), q]
        q_idx = torch.arange(q_blk, device=device).unsqueeze(1)  # [Q, 1]
        k_idx = torch.arange(k_blk, device=device).unsqueeze(0)  # [1, K]
        local_start = torch.clamp(q_idx - local_blocks + 1, min=0)
        is_local = (k_idx >= local_start) & (k_idx <= q_idx) & valid_mask
        mandatory_keep_mask = is_local.unsqueeze(0).unsqueeze(0).expand(
            batch, heads, -1, -1,
        )

        # Compute budget: total keep = valid_per_row * (1 - sparsity)
        valid_per_row = valid_mask.sum(dim=-1)  # [B, H, Q]
        keep_per_row = (valid_per_row.float() * (1.0 - target_sparsity)).ceil().long()
        keep_per_row = torch.clamp(keep_per_row, min=1)
    else:
        mandatory_keep_mask = None
        valid_per_row = valid_mask.sum(dim=-1)  # [B, H, Q]
        keep_per_row = (valid_per_row.float() * (1.0 - target_sparsity)).ceil().long()
        keep_per_row = torch.clamp(keep_per_row, min=1)

    result = vectorized_topk_mask(
        fp_scores,
        k_per_row=keep_per_row,
        valid_mask=valid_mask,
        mandatory_keep_mask=mandatory_keep_mask,
    )
    return result.mask


def _compute_pytorch_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    block_size: int,
    group_size: int,
    scale_by_sqrt_dim: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference for quantized/lower/upper scores.

    Uses the same math as ``aglr_certimask_topk`` but only scores, no mask.
    """
    from certimask.aglr_certimask import _compute_sampled_dot_intervals
    from certimask.bounds import _get_group_per_coord_error
    from certimask.topk_certificate import logsumexp_interval

    batch, heads, seq_len, dim = query.shape  # noqa: F841

    # Quantize K
    k_quantized = quantize_int8_per_group(key, group_size=group_size)
    k_tilde = k_quantized.dequantized.to(torch.float32)
    k_err_per_coord = _get_group_per_coord_error(k_quantized, "analytic")

    # Compute per-sample intervals
    _, lower_samples, upper_samples = _compute_sampled_dot_intervals(
        query,
        k_tilde,
        k_err_per_coord,
        block_size=block_size,
        sample_pattern="both_diagonals",
        scale_by_sqrt_dim=scale_by_sqrt_dim,
    )

    # Also compute quantized dots (midpoint)
    quant_samples = (lower_samples + upper_samples) / 2.0

    # logsumexp aggregation
    lower_scores, upper_scores = logsumexp_interval(
        lower_samples, upper_samples, dim=-1,
    )
    # For quantized: use logsumexp of midpoints
    quantized_scores = torch.logsumexp(quant_samples, dim=-1)

    # Mask invalid
    lower_scores = lower_scores.masked_fill(
        ~valid_mask, torch.finfo(torch.float32).min,
    )
    upper_scores = upper_scores.masked_fill(
        ~valid_mask, torch.finfo(torch.float32).min,
    )
    quantized_scores = quantized_scores.masked_fill(
        ~valid_mask, torch.finfo(torch.float32).min,
    )

    return quantized_scores, lower_scores, upper_scores


def triton_aglr_certimask_logsumexp_g4(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    target_sparsity: float,
    local_blocks: int = 0,
    valid_mask: torch.Tensor | None = None,
    reference_scores: torch.Tensor | None = None,
    scale_by_sqrt_dim: bool = True,
    ambiguity_mode: str = "partition",
    topk_mask_mode: str = "vectorized",
) -> TritonAGLRCertiMaskResult:
    """Triton-accelerated AGLR-C CertiMask with logsumexp and group_size=4.

    Fixed parameters:
        block_size=8, group_size=4, sample_pattern=both_diagonals,
        aggregation=logsumexp.

    Args:
        query: [B, H, L, D] FP16 or FP32 query on CUDA.
        key: [B, H, L, D] FP16 or FP32 key on CUDA.
        target_sparsity: Fraction of valid tiles to drop.
        local_blocks: Mandatory local blocks per query row.
        valid_mask: Optional [B, H, Q_blk, K_blk] bool causal mask.
        reference_scores: Optional pre-computed FP32 reference scores.
        scale_by_sqrt_dim: Whether to scale dots by 1/sqrt(d).
        ambiguity_mode: "threshold" or "partition".
        topk_mask_mode: "loop", "vectorized", or "triton".

    Returns:
        TritonAGLRCertiMaskResult with scores, certificate, and mask.

    Raises:
        RuntimeError: If CUDA or Triton is not available.
    """
    from certimask.masking import make_block_causal_valid_mask
    from certimask.triton_aglr_kernels import triton_aglr_logsumexp_scoring

    batch, heads, seq_len, dim = query.shape
    block_size = 8
    group_size = 4
    num_blocks = seq_len // block_size

    assert query.device.type == "cuda", f"Expected CUDA tensor, got {query.device}"
    assert dim == 64, f"Expected D=64, got {dim}"

    # Valid mask
    if valid_mask is None:
        valid_block_mask = make_block_causal_valid_mask(
            num_blocks, num_blocks, device=query.device,
        ).expand(batch, heads, num_blocks, num_blocks)
    else:
        valid_block_mask = valid_mask

    valid_scores = valid_block_mask[:, :, :num_blocks, :num_blocks]

    # --- FP32 reference ---
    if reference_scores is None:
        fp_scores = _compute_reference_scores(
            query,
            key,
            valid_mask=valid_scores,
            block_size=block_size,
            sample_pattern="both_diagonals",
            aggregation="logsumexp",
            scale_by_sqrt_dim=scale_by_sqrt_dim,
        )
    else:
        fp_scores = reference_scores

    # --- Reference mask construction ---
    if topk_mask_mode == "vectorized":
        reference_mask = _compute_reference_mask_vectorized(
            fp_scores,
            target_sparsity=target_sparsity,
            local_blocks=local_blocks,
            valid_mask=valid_scores,
        )
    elif topk_mask_mode == "loop":
        reference_mask = _compute_reference_mask(
            fp_scores,
            target_sparsity=target_sparsity,
            local_blocks=local_blocks,
            valid_mask=valid_scores,
        )
    else:
        raise ValueError(
            f"Unknown topk_mask_mode '{topk_mask_mode}'. "
            "Supported: 'loop', 'vectorized'"
        )

    # Pad reference mask if needed
    if reference_mask.shape != valid_block_mask.shape:
        padded = torch.zeros_like(valid_block_mask)
        n2, k2 = reference_mask.shape[2], reference_mask.shape[3]
        padded[:, :, :n2, :k2] = reference_mask
        reference_mask = padded

    # Per-row k
    k_per_row = (reference_mask & valid_block_mask).sum(dim=-1).long()

    # --- INT8 K quantization + expand ---
    k_quantized = quantize_int8_per_group(key, group_size=group_size)
    key_int8 = k_quantized.values  # [B, H, L, D] int8
    key_scales = k_quantized.scale  # [B, H, L, G] fp32
    key_is_zero = k_quantized.is_zero_group  # [B, H, L, G] bool

    key_scales_expanded = _expand_group_tensor(key_scales, group_size, dim).contiguous()
    key_is_zero_expanded = _expand_group_tensor(
        key_is_zero.to(torch.int8), group_size, dim,
    ).to(torch.bool).contiguous()

    # --- Triton kernel ---
    triton_quant, triton_lower, triton_upper = triton_aglr_logsumexp_scoring(
        query,
        key_int8,
        key_scales_expanded,
        key_is_zero_expanded,
        valid_scores,
    )

    # --- Partition certificate (PyTorch) ---
    topk_result = certified_topk_mask(
        fp_scores,
        triton_lower,
        triton_upper,
        k_per_row=k_per_row,
        valid_mask=valid_scores,
        ambiguity_mode=ambiguity_mode,
    )

    mask = topk_result.certified_mask
    mismatch = (mask != reference_mask) & valid_block_mask
    mismatch_count = int(mismatch.sum().item())
    exact_match = mismatch_count == 0

    return TritonAGLRCertiMaskResult(
        quantized_scores=triton_quant,
        lower_scores=triton_lower,
        upper_scores=triton_upper,
        decisions=topk_result.decisions,
        ambiguous=topk_result.ambiguous,
        mask=mask,
        reference_mask=reference_mask,
        exact_match=exact_match,
        mismatch_count=mismatch_count,
    )
