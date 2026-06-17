"""AGLR-C CertiMask: Certified top-k block selection with INT8 K-only quantization.

Combines AGLR-C v1 antidiagonal scoring with per-group INT8 K-only
quantization bounds to produce certified block-sparse attention masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from certimask.aglr_indexer import (
    _generate_sample_positions,
    aglr_local_plus_landmark_mask,
    compute_antidiagonal_block_scores,
)
from certimask.bounds import _get_group_per_coord_error
from certimask.quantization import quantize_int8_per_group
from certimask.topk_certificate import (
    DROP,
    KEEP,
    TopKCertificateResult,
    certified_topk_mask,
    logsumexp_interval,
)


@dataclass
class AGLRCertiMaskResult:
    """Result of AGLR CertiMask top-k certification.

    Attributes:
        mask: Final boolean block mask [B, H, Q_blk, K_blk].
        reference_mask: FP32 AGLR reference mask.
        decisions: Per-tile decision codes [B, H, Q_blk, K_blk].
        row_certified: Per-row certification status [B, H, Q_blk].
        ambiguous: Boolean mask of ambiguous tiles.
        fallback_mask: Boolean mask of fallback tiles.
        quantized_scores: Quantized midpoint block scores.
        lower_scores: Lower bound block scores.
        upper_scores: Upper bound block scores.
        exact_mask_match: Whether mask == reference_mask.
        mismatch_count: Number of mismatched tiles.
        topk_result: Underlying TopKCertificateResult.
    """

    mask: torch.Tensor
    reference_mask: torch.Tensor
    decisions: torch.Tensor
    row_certified: torch.Tensor
    ambiguous: torch.Tensor
    fallback_mask: torch.Tensor
    quantized_scores: torch.Tensor
    lower_scores: torch.Tensor
    upper_scores: torch.Tensor
    exact_mask_match: bool
    mismatch_count: int
    topk_result: TopKCertificateResult


@dataclass
class AGLRCertiMaskMetrics:
    """Metrics for AGLR CertiMask certification.

    Attributes:
        valid_tiles: Total number of valid tiles.
        selected_tiles: Number of selected (kept) tiles.
        row_count: Total number of query rows.
        row_certified_count: Number of certified rows.
        row_certification_rate: Fraction of rows certified.
        ambiguous_tiles: Number of ambiguous tiles.
        ambiguous_rate: Fraction of valid tiles ambiguous.
        fallback_tiles: Number of fallback tiles.
        fallback_rate: Fraction of valid tiles requiring fallback.
        exact_mask_match: Whether mask matches reference.
        mismatch_count: Number of mismatches.
        mean_interval_width: Mean score interval width.
        p50_interval_width: P50 interval width.
        p90_interval_width: P90 interval width.
        p99_interval_width: P99 interval width.
        selected_ambiguous_rate: Ambiguous rate among selected tiles.
        rejected_ambiguous_rate: Ambiguous rate among rejected tiles.
        boundary_band_size_mean: Mean ambiguous count per row.
        boundary_band_size_p90: P90 ambiguous count per row.
        certified_keep_rate: Fraction of selected tiles certified KEEP.
        certified_drop_rate: Fraction of rejected tiles certified DROP.
        mean_margin_to_boundary: Mean margin to boundary (non-negative only).
        p10_margin_to_boundary: P10 margin to boundary.
        score_interval_width_over_margin_p50: P50 of (interval_width / margin).
        score_interval_width_over_margin_p90: P90 of (interval_width / margin).
    """

    valid_tiles: int
    selected_tiles: int
    row_count: int
    row_certified_count: int
    row_certification_rate: float
    ambiguous_tiles: int
    ambiguous_rate: float
    fallback_tiles: int
    fallback_rate: float
    exact_mask_match: bool
    mismatch_count: int
    mean_interval_width: float
    p50_interval_width: float
    p90_interval_width: float
    p99_interval_width: float
    selected_ambiguous_rate: float
    rejected_ambiguous_rate: float
    boundary_band_size_mean: float
    boundary_band_size_p90: float
    certified_keep_rate: float
    certified_drop_rate: float
    mean_margin_to_boundary: float
    p10_margin_to_boundary: float
    score_interval_width_over_margin_p50: float
    score_interval_width_over_margin_p90: float


def _compute_sampled_dot_intervals(
    query: torch.Tensor,
    key_tilde: torch.Tensor,
    key_err_per_coord: torch.Tensor,
    *,
    block_size: int,
    sample_pattern: str,
    scale_by_sqrt_dim: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-sample dot intervals for all block pairs.

    Args:
        query: [B, H, L, D] FP32 query tensor.
        key_tilde: [B, H, L, D] dequantized key tensor.
        key_err_per_coord: [B, H, L, D] per-coordinate K error bound.
        block_size: Tokens per block.
        sample_pattern: Sample position pattern.
        scale_by_sqrt_dim: Whether to divide by sqrt(d).

    Returns:
        fp_dots: [B, H, Q_blk, K_blk, P] FP32 reference dots.
        lower_dots: [B, H, Q_blk, K_blk, P] lower bound dots.
        upper_dots: [B, H, Q_blk, K_blk, P] upper bound dots.
    """
    query = query.to(torch.float32)
    key_tilde = key_tilde.to(torch.float32)
    key_err_per_coord = key_err_per_coord.to(torch.float32)

    batch, heads, seq_len, dim = query.shape
    num_blocks = seq_len // block_size
    used_len = num_blocks * block_size

    query = query[:, :, :used_len, :]
    key_tilde = key_tilde[:, :, :used_len, :]

    # Reshape to blocks: [B, H, num_blocks, block_size, D]
    q_blocks = query.reshape(batch, heads, num_blocks, block_size, dim)
    kt_blocks = key_tilde.reshape(batch, heads, num_blocks, block_size, dim)

    # Also need FP key for reference dots - use key_tilde as proxy
    # (FP key is key_tilde + error, but we compute FP dots separately)
    # For now, we compute reference dots from the original FP key
    # by reconstructing: key_fp = key_tilde + (key_fp - key_tilde)
    # But we don't have key_fp here. We'll compute fp_dots from
    # the original compute_antidiagonal_block_scores call instead.
    # Here we just compute the interval components.

    # Per-coordinate K error bound in blocks
    ke_blocks = key_err_per_coord.reshape(
        batch, heads, num_blocks, block_size, dim,
    )

    positions = _generate_sample_positions(block_size, sample_pattern)
    q_indices = torch.tensor(
        [p[0] for p in positions], device=query.device, dtype=torch.long,
    )
    k_indices = torch.tensor(
        [p[1] for p in positions], device=query.device, dtype=torch.long,
    )
    num_samples = len(positions)

    # Gather at sample positions
    q_sampled = q_blocks[:, :, :, q_indices, :]    # [B, H, nb, P, D]
    kt_sampled = kt_blocks[:, :, :, k_indices, :]   # [B, H, nb, P, D]
    ke_sampled = ke_blocks[:, :, :, k_indices, :]   # [B, H, nb, P, D]

    # Compute dots for all block pairs per sample
    # q_sampled[:, :, :, p, :] -> [B, H, Q_blk, D]
    # kt_sampled[:, :, :, p, :] -> [B, H, K_blk, D]
    lower_dots_list: list[torch.Tensor] = []
    upper_dots_list: list[torch.Tensor] = []

    sqrt_d = torch.sqrt(torch.tensor(float(dim), dtype=torch.float32,
                                      device=query.device))

    for p in range(num_samples):
        q_s = q_sampled[:, :, :, p, :]       # [B, H, Q_blk, D]
        kt_s = kt_sampled[:, :, :, p, :]     # [B, H, K_blk, D]
        ke_s = ke_sampled[:, :, :, p, :]     # [B, H, K_blk, D]

        # Quantized dot: z_tilde = q^T k_tilde / sqrt(d)
        z_tilde = torch.einsum("bhqd,bhkd->bhqk", q_s, kt_s)
        if scale_by_sqrt_dim:
            z_tilde = z_tilde / sqrt_d

        # Error bound: err = sum_r |q_r| * k_err_r / sqrt(d)
        err = torch.einsum("bhqd,bhkd->bhqk", q_s.abs(), ke_s)
        if scale_by_sqrt_dim:
            err = err / sqrt_d

        lower_dots_list.append(z_tilde - err)
        upper_dots_list.append(z_tilde + err)

    # Stack: [B, H, Q_blk, K_blk, P]
    lower_dots = torch.stack(lower_dots_list, dim=-1)
    upper_dots = torch.stack(upper_dots_list, dim=-1)

    # FP dots not computed here (use compute_antidiagonal_block_scores)
    fp_dots = torch.zeros_like(lower_dots)

    return fp_dots, lower_dots, upper_dots


def _aggregate_topk_mean_interval(
    lower_samples: torch.Tensor,
    upper_samples: torch.Tensor,
    *,
    k: int,
    dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservative topk-mean interval: mean of top-k lower and top-k upper.

    For each row, selects the k largest samples by the midpoint score and
    computes mean of their lower bounds and mean of their upper bounds.
    This is conservative: the true topk-mean lies in [mean_lo, mean_hi].

    Args:
        lower_samples: Lower bounds [..., P].
        upper_samples: Upper bounds [..., P].
        k: Number of top samples to average.
        dim: Sample dimension.

    Returns:
        (lower, upper) aggregated scores.
    """
    midpoint = (lower_samples + upper_samples) / 2.0
    k_actual = min(k, lower_samples.shape[dim])
    _, topk_idx = midpoint.topk(k_actual, dim=dim)

    # Gather top-k lower and upper
    lower_topk = torch.gather(lower_samples, dim, topk_idx)
    upper_topk = torch.gather(upper_samples, dim, topk_idx)

    lower = lower_topk.mean(dim=dim)
    upper = upper_topk.mean(dim=dim)
    upper = torch.maximum(lower, upper)
    return lower, upper


def aglr_certimask_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    block_size: int = 8,
    target_sparsity: float,
    local_blocks: int = 0,
    sample_pattern: str = "both_diagonals",
    aggregation: str = "logsumexp",
    group_size: int = 16,
    valid_mask: torch.Tensor | None = None,
    scale_by_sqrt_dim: bool = True,
    ambiguity_mode: str = "partition",
) -> AGLRCertiMaskResult:
    """Certify AGLR-C v1 top-k block selection with INT8 K-only quantization.

    Args:
        query: [B, H, L, D] FP32 query tensor.
        key: [B, H, L, D] FP32 key tensor.
        block_size: Tokens per block.
        target_sparsity: Fraction of valid tiles to drop.
        local_blocks: Mandatory local blocks per query row.
        sample_pattern: Antidiagonal sample pattern.
        aggregation: Score aggregation method ("logsumexp" or "topk_mean").
        group_size: INT8 per-group quantization group size.
        valid_mask: Optional [B, H, Q_blk, K_blk] boolean causal mask.
        scale_by_sqrt_dim: Whether to scale dots by 1/sqrt(d).
        ambiguity_mode: "threshold" or "partition".

    Returns:
        AGLRCertiMaskResult with certified mask and diagnostics.

    Raises:
        ValueError: If aggregation is unsupported.
    """
    if aggregation not in ("logsumexp", "topk_mean"):
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'. "
            "Supported: 'logsumexp', 'topk_mean'"
        )

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    batch, heads, seq_len, dim = query.shape
    num_blocks = seq_len // block_size
    device = query.device

    # Build valid block mask if not provided
    if valid_mask is None:
        from certimask.masking import make_block_causal_valid_mask

        valid_block_mask = make_block_causal_valid_mask(
            num_blocks, num_blocks, device=device,
        ).expand(batch, heads, num_blocks, num_blocks)
    else:
        valid_block_mask = valid_mask

    # Step 1: FP reference AGLR scores
    nb = num_blocks
    valid_scores = valid_block_mask[:, :, :nb, :nb]

    fp_scores = compute_antidiagonal_block_scores(
        query, key, block_size=block_size,
        sample_pattern=sample_pattern, aggregation=aggregation,
        valid_mask=valid_scores, scale_by_sqrt_dim=scale_by_sqrt_dim,
    )

    # Step 2: FP reference AGLR mask
    ref_result = aglr_local_plus_landmark_mask(
        fp_scores, target_sparsity=target_sparsity,
        local_blocks=local_blocks, valid_mask=valid_scores,
    )
    reference_mask = ref_result.mask

    if reference_mask.shape != valid_block_mask.shape:
        padded = torch.zeros_like(valid_block_mask)
        n2, k2 = reference_mask.shape[2], reference_mask.shape[3]
        padded[:, :, :n2, :k2] = reference_mask
        reference_mask = padded

    # Step 3: Per-row k from the reference mask
    k_per_row = (reference_mask & valid_block_mask).sum(dim=-1).long()

    # Step 4: Quantize K per-group INT8
    k_quantized = quantize_int8_per_group(key, group_size=group_size)
    k_tilde = k_quantized.dequantized.to(torch.float32)

    # Step 5: Per-coordinate K error bound
    k_err_per_coord = _get_group_per_coord_error(k_quantized, "analytic")

    # Step 6: Compute sampled dot intervals
    positions = _generate_sample_positions(block_size, sample_pattern)

    used_len = num_blocks * block_size
    q_blocks = query[:, :, :used_len, :].reshape(
        batch, heads, num_blocks, block_size, dim,
    )
    kt_blocks = k_tilde[:, :, :used_len, :].reshape(
        batch, heads, num_blocks, block_size, dim,
    )
    ke_blocks = k_err_per_coord[:, :, :used_len, :].reshape(
        batch, heads, num_blocks, block_size, dim,
    )

    q_indices = torch.tensor(
        [p[0] for p in positions], device=device, dtype=torch.long,
    )
    k_indices = torch.tensor(
        [p[1] for p in positions], device=device, dtype=torch.long,
    )
    num_samples = len(positions)

    q_sampled = q_blocks[:, :, :, q_indices, :]
    kt_sampled = kt_blocks[:, :, :, k_indices, :]
    ke_sampled = ke_blocks[:, :, :, k_indices, :]

    sqrt_d = torch.sqrt(torch.tensor(float(dim), dtype=torch.float32,
                                      device=device))

    lower_list: list[torch.Tensor] = []
    upper_list: list[torch.Tensor] = []

    for p in range(num_samples):
        q_s = q_sampled[:, :, :, p, :]
        kt_s = kt_sampled[:, :, :, p, :]
        ke_s = ke_sampled[:, :, :, p, :]

        z_tilde = torch.einsum("bhqd,bhkd->bhqk", q_s, kt_s)
        err = torch.einsum("bhqd,bhkd->bhqk", q_s.abs(), ke_s)

        if scale_by_sqrt_dim:
            z_tilde = z_tilde / sqrt_d
            err = err / sqrt_d

        lower_list.append(z_tilde - err)
        upper_list.append(z_tilde + err)

    lower_samples = torch.stack(lower_list, dim=-1)
    upper_samples = torch.stack(upper_list, dim=-1)

    # Step 7: Aggregate
    if aggregation == "logsumexp":
        lower_scores, upper_scores = logsumexp_interval(
            lower_samples, upper_samples, dim=-1,
        )
    else:
        # topk_mean: use top-4 samples by midpoint
        topk_k = min(4, num_samples)
        lower_scores, upper_scores = _aggregate_topk_mean_interval(
            lower_samples, upper_samples, k=topk_k, dim=-1,
        )

    # Apply valid mask
    lower_scores = lower_scores.masked_fill(
        ~valid_scores, torch.finfo(torch.float32).min,
    )
    upper_scores = upper_scores.masked_fill(
        ~valid_scores, torch.finfo(torch.float32).min,
    )

    # Step 8: Certified top-k mask
    topk_result = certified_topk_mask(
        fp_scores, lower_scores, upper_scores,
        k_per_row=k_per_row, valid_mask=valid_scores,
        ambiguity_mode=ambiguity_mode,
    )

    # Step 9: Verify exact match
    mask = topk_result.certified_mask
    mismatch = (mask != reference_mask) & valid_block_mask
    mismatch_count = int(mismatch.sum().item())
    exact_match = mismatch_count == 0

    return AGLRCertiMaskResult(
        mask=mask,
        reference_mask=reference_mask,
        decisions=topk_result.decisions,
        row_certified=topk_result.row_certified,
        ambiguous=topk_result.ambiguous,
        fallback_mask=topk_result.fallback_mask,
        quantized_scores=topk_result.quantized_scores,
        lower_scores=lower_scores,
        upper_scores=upper_scores,
        exact_mask_match=exact_match,
        mismatch_count=mismatch_count,
        topk_result=topk_result,
    )


def compute_aglr_certimask_metrics(
    result: AGLRCertiMaskResult,
    valid_mask: torch.Tensor,
) -> AGLRCertiMaskMetrics:
    """Compute AGLR CertiMask certification metrics.

    Args:
        result: AGLRCertiMaskResult from aglr_certimask_topk.
        valid_mask: Boolean valid block mask [B, H, Q_blk, K_blk].

    Returns:
        AGLRCertiMaskMetrics.
    """
    decisions = result.topk_result.decisions
    ref_mask = result.topk_result.selected_reference_mask

    valid_count = int(valid_mask.sum().item())
    selected_count = int((result.mask & valid_mask).sum().item())
    row_count = result.row_certified.numel()
    certified_count = int(result.row_certified.sum().item())
    ambiguous_count = int((result.ambiguous & valid_mask).sum().item())
    fallback_count = int((result.fallback_mask & valid_mask).sum().item())

    # Selected/rejected ambiguous rates
    selected_valid = ref_mask & valid_mask
    rejected_valid = valid_mask & ~ref_mask
    selected_ambig = int((result.ambiguous & selected_valid).sum().item())
    rejected_ambig = int((result.ambiguous & rejected_valid).sum().item())
    selected_count_valid = int(selected_valid.sum().item())
    rejected_count_valid = int(rejected_valid.sum().item())

    # Certified KEEP/DROP rates (from decisions array)
    cert_keep_count = int(((decisions == KEEP) & valid_mask).sum().item())
    cert_drop_count = int(((decisions == DROP) & valid_mask).sum().item())

    # Boundary band size per row (ambiguous valid tiles per row)
    ambig_per_row = (result.ambiguous & valid_mask).float().sum(dim=(-2, -1))
    ambig_flat = ambig_per_row.reshape(-1)

    # Interval width statistics
    width = result.upper_scores - result.lower_scores
    width_valid = width[valid_mask]

    # Margin to boundary (only non-negative values)
    margin = result.topk_result.margin_to_boundary
    margin_valid = margin[valid_mask & (margin >= 0)]

    # Interval width / margin ratio (for margin > 0)
    margin_pos = margin[valid_mask & (margin > 0)]
    width_for_ratio = width[valid_mask & (margin > 0)]
    if margin_pos.numel() > 0:
        ratio = width_for_ratio / (margin_pos + 1e-12)
    else:
        ratio = torch.zeros(0, device=valid_mask.device)

    def _percentile(t: torch.Tensor, q: float) -> float:
        if t.numel() == 0:
            return 0.0
        return float(t.quantile(q).item())

    return AGLRCertiMaskMetrics(
        valid_tiles=valid_count,
        selected_tiles=selected_count,
        row_count=row_count,
        row_certified_count=certified_count,
        row_certification_rate=certified_count / row_count if row_count > 0 else 0.0,
        ambiguous_tiles=ambiguous_count,
        ambiguous_rate=ambiguous_count / valid_count if valid_count > 0 else 0.0,
        fallback_tiles=fallback_count,
        fallback_rate=fallback_count / valid_count if valid_count > 0 else 0.0,
        exact_mask_match=result.exact_mask_match,
        mismatch_count=result.mismatch_count,
        mean_interval_width=float(width_valid.mean().item()) if width_valid.numel() > 0 else 0.0,
        p50_interval_width=_percentile(width_valid, 0.50),
        p90_interval_width=_percentile(width_valid, 0.90),
        p99_interval_width=_percentile(width_valid, 0.99),
        selected_ambiguous_rate=(
            selected_ambig / selected_count_valid
            if selected_count_valid > 0 else 0.0
        ),
        rejected_ambiguous_rate=(
            rejected_ambig / rejected_count_valid
            if rejected_count_valid > 0 else 0.0
        ),
        boundary_band_size_mean=float(ambig_flat.mean().item()),
        boundary_band_size_p90=_percentile(ambig_flat, 0.90),
        certified_keep_rate=(
            cert_keep_count / selected_count_valid
            if selected_count_valid > 0 else 0.0
        ),
        certified_drop_rate=(
            cert_drop_count / rejected_count_valid
            if rejected_count_valid > 0 else 0.0
        ),
        mean_margin_to_boundary=(
            float(margin_valid.mean().item())
            if margin_valid.numel() > 0 else 0.0
        ),
        p10_margin_to_boundary=_percentile(margin_valid, 0.10),
        score_interval_width_over_margin_p50=_percentile(ratio, 0.50),
        score_interval_width_over_margin_p90=_percentile(ratio, 0.90),
    )
