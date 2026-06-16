"""Mask metrics and score certificate tightness metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from certimask.bounds import ScoreBounds
from certimask.masking import CertiMaskResult


@dataclass
class MaskMetrics:
    """Metrics for mask quality evaluation.

    Attributes:
        valid_tiles: Number of valid tiles.
        reference_kept: Number of tiles kept by reference mask.
        reference_dropped: Number of tiles dropped by reference mask.
        actual_sparsity: Fraction of valid tiles that are dropped.
        naive_mismatch_count: Number of tiles where naive INT8 mask differs
            from reference mask.
        naive_mismatch_rate: naive_mismatch_count / valid_tiles.
        false_drop_count: Tiles kept by reference but dropped by naive INT8.
        false_drop_rate: false_drop_count / valid_tiles.
        false_keep_count: Tiles dropped by reference but kept by naive INT8.
        false_keep_rate: false_keep_count / valid_tiles.
        certimask_mismatch_count: Tiles where CertiMask differs from reference.
            Should always be 0.
        certimask_match_rate: Fraction of valid tiles where CertiMask matches
            reference. Should always be 1.0.
        certain_keep_count: Number of CERTAIN KEEP tiles.
        certain_drop_count: Number of CERTAIN DROP tiles.
        ambiguous_count: Number of AMBIGUOUS tiles.
        refinement_rate: Fraction of valid tiles that are AMBIGUOUS.
    """

    valid_tiles: int
    reference_kept: int
    reference_dropped: int
    actual_sparsity: float
    naive_mismatch_count: int
    naive_mismatch_rate: float
    false_drop_count: int
    false_drop_rate: float
    false_keep_count: int
    false_keep_rate: float
    certimask_mismatch_count: int
    certimask_match_rate: float
    certain_keep_count: int
    certain_drop_count: int
    ambiguous_count: int
    refinement_rate: float


@dataclass
class BoundMetrics:
    """Metrics for score certificate tightness.

    Attributes:
        valid_tiles: Number of valid tiles.
        violations: Number of score bound violations.
        violation_rate: violations / valid_tiles.
        mean_error: Mean absolute score error |s - s_tilde|.
        max_error: Maximum absolute score error.
        mean_bound: Mean error bound E.
        max_bound: Maximum error bound.
        rho_mean: Mean of |s - s_tilde| / (E + eps).
        rho_p50: P50 of rho.
        rho_p90: P90 of rho.
        rho_p99: P99 of rho.
        rho_max: Maximum rho.
        margin_over_bound_mean: Mean of |s - tau| / (E + eps).
        margin_over_bound_p50: P50 of margin-over-bound.
        margin_over_bound_p90: P90 of margin-over-bound.
    """

    valid_tiles: int
    violations: int
    violation_rate: float
    mean_error: float
    max_error: float
    mean_bound: float
    max_bound: float
    rho_mean: float
    rho_p50: float
    rho_p90: float
    rho_p99: float
    rho_max: float
    margin_over_bound_mean: float
    margin_over_bound_p50: float
    margin_over_bound_p90: float


def compute_mask_metrics(
    reference: torch.Tensor,
    naive: torch.Tensor,
    certified: CertiMaskResult,
    *,
    valid_mask: torch.Tensor | None = None,
) -> MaskMetrics:
    """Compute mask quality metrics.

    Args:
        reference: Boolean reference mask, shape [B, H, Q, K].
        naive: Boolean naive INT8 mask, shape [B, H, Q, K].
        certified: CertiMaskResult from certified_threshold_mask.
        valid_mask: Optional boolean mask. Only valid tiles are counted.

    Returns:
        MaskMetrics with all metric values.

    Raises:
        ValueError: If shapes don't match.
    """
    if reference.shape != naive.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs naive {naive.shape}"
        )
    if reference.shape != certified.mask.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs "
            f"certified {certified.mask.shape}"
        )

    if valid_mask is not None:
        if valid_mask.shape != reference.shape:
            raise ValueError(
                f"Shape mismatch: valid_mask {valid_mask.shape} vs "
                f"reference {reference.shape}"
            )
        vm = valid_mask
    else:
        vm = torch.ones_like(reference, dtype=torch.bool)

    valid_count = int(vm.sum().item())
    if valid_count == 0:
        raise ValueError("No valid tiles found")

    ref_valid = reference & vm
    naive_valid = naive & vm
    cert_valid = certified.mask & vm

    reference_kept = int(ref_valid.sum().item())
    reference_dropped = valid_count - reference_kept
    actual_sparsity = reference_dropped / valid_count

    # Naive mismatch
    naive_mismatch = (naive_valid != ref_valid) & vm
    naive_mismatch_count = int(naive_mismatch.sum().item())

    # False drop: reference KEEP but naive DROP
    false_drop = ref_valid & ~naive_valid
    false_drop_count = int(false_drop.sum().item())

    # False keep: reference DROP but naive KEEP
    false_keep = ~ref_valid & naive_valid
    false_keep_count = int(false_keep.sum().item())

    # CertiMask mismatch
    cert_mismatch = (cert_valid != ref_valid) & vm
    certimask_mismatch_count = int(cert_mismatch.sum().item())

    # Counts from decisions
    certain_keep_count = int(
        (certified.certain_keep & vm).sum().item()
    )
    certain_drop_count = int(
        (certified.certain_drop & vm).sum().item()
    )
    ambiguous_count = int(
        (certified.ambiguous & vm).sum().item()
    )

    return MaskMetrics(
        valid_tiles=valid_count,
        reference_kept=reference_kept,
        reference_dropped=reference_dropped,
        actual_sparsity=actual_sparsity,
        naive_mismatch_count=naive_mismatch_count,
        naive_mismatch_rate=naive_mismatch_count / valid_count,
        false_drop_count=false_drop_count,
        false_drop_rate=false_drop_count / valid_count,
        false_keep_count=false_keep_count,
        false_keep_rate=false_keep_count / valid_count,
        certimask_mismatch_count=certimask_mismatch_count,
        certimask_match_rate=1.0 - certimask_mismatch_count / valid_count,
        certain_keep_count=certain_keep_count,
        certain_drop_count=certain_drop_count,
        ambiguous_count=ambiguous_count,
        refinement_rate=ambiguous_count / valid_count,
    )


def compute_bound_metrics(
    reference_scores: torch.Tensor,
    quantized_scores: torch.Tensor,
    bounds: ScoreBounds,
    threshold: torch.Tensor | float,
    *,
    valid_mask: torch.Tensor | None = None,
) -> BoundMetrics:
    """Compute score certificate tightness metrics.

    Args:
        reference_scores: FP32 reference scores, shape [B, H, Q, K].
        quantized_scores: INT8 dequantized scores, shape [B, H, Q, K].
        bounds: ScoreBounds from compute_score_bounds.
        threshold: The threshold used for masking.
        valid_mask: Optional boolean mask. Only valid tiles are counted.

    Returns:
        BoundMetrics with tightness metrics.

    Raises:
        ValueError: If shapes don't match or no valid tiles.
    """
    if reference_scores.shape != quantized_scores.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference_scores.shape} vs "
            f"quantized {quantized_scores.shape}"
        )
    if reference_scores.shape != bounds.lower.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference_scores.shape} vs "
            f"bounds {bounds.lower.shape}"
        )

    ref = reference_scores.to(torch.float32)
    quant = quantized_scores.to(torch.float32)

    vm = valid_mask if valid_mask is not None else torch.ones_like(ref, dtype=torch.bool)

    valid_count = int(vm.sum().item())
    if valid_count == 0:
        raise ValueError("No valid tiles found")

    # Score error
    error = (ref - quant).abs()
    error_valid = error[vm]

    # Bound
    bound = bounds.error_bound.to(torch.float32)
    bound_valid = bound[vm]

    # Violations: reference outside [lower - atol, upper + atol]
    violations = ((ref < bounds.lower - 1e-5) | (ref > bounds.upper + 1e-5)) & vm
    violation_count = int(violations.sum().item())

    # rho = |s - s_tilde| / (E + eps)
    eps = torch.finfo(torch.float32).eps
    rho = error / (bound + eps)
    rho_valid = rho[vm]

    # margin_over_bound = |s - tau| / (E + eps)
    tau = torch.as_tensor(threshold, dtype=torch.float32, device=ref.device)
    margin = (ref - tau).abs() / (bound + eps)
    margin_valid = margin[vm]

    return BoundMetrics(
        valid_tiles=valid_count,
        violations=violation_count,
        violation_rate=violation_count / valid_count,
        mean_error=float(error_valid.mean().item()),
        max_error=float(error_valid.max().item()),
        mean_bound=float(bound_valid.mean().item()),
        max_bound=float(bound_valid.max().item()),
        rho_mean=float(rho_valid.mean().item()),
        rho_p50=float(rho_valid.quantile(0.50).item()),
        rho_p90=float(rho_valid.quantile(0.90).item()),
        rho_p99=float(rho_valid.quantile(0.99).item()),
        rho_max=float(rho_valid.max().item()),
        margin_over_bound_mean=float(margin_valid.mean().item()),
        margin_over_bound_p50=float(margin_valid.quantile(0.50).item()),
        margin_over_bound_p90=float(margin_valid.quantile(0.90).item()),
    )
