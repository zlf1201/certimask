"""Diagnostic tools for analyzing CertiMask refinement behavior."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from certimask.bounds import ScoreBounds


@dataclass
class AttentionReconstructionDiagnostics:
    """Results of attention reconstruction verification.

    Attributes:
        query_max_abs_diff: Max |Q_extracted - Q_internal|.
        key_max_abs_diff: Max |K_extracted - K_internal|.
        logits_max_abs_diff: Max |QK^T/sqrt(d) - internal_logits|, or None
            if internal logits were not captured.
        masked_logits_max_abs_diff: Max |masked_logits - internal_masked|,
            or None.
        probabilities_max_abs_diff: Max |reconstructed_probs - model_probs|.
        probabilities_mean_abs_diff: Mean |reconstructed_probs - model_probs|.
    """

    query_max_abs_diff: float
    key_max_abs_diff: float
    logits_max_abs_diff: float | None
    masked_logits_max_abs_diff: float | None
    probabilities_max_abs_diff: float
    probabilities_mean_abs_diff: float


@dataclass
class PerTileDiagnostics:
    """Per-tile diagnostic quantities for one configuration.

    All tensors have shape [B, H, Q, K] with valid_mask applied.

    Attributes:
        score_error: |s - s_tilde|.
        margin: |s - threshold|.
        actual_bound: E_actual.
        analytic_bound: E_analytic.
        flip_mask: True where naive decision differs from reference.
        oracle_crossing: True where the segment [s, s_tilde] contains threshold.
        ratio_error_to_margin: score_error / (margin + eps).
        ratio_actual_to_error: actual_bound / (score_error + eps).
        ratio_analytic_to_error: analytic_bound / (score_error + eps).
        ratio_margin_to_actual: margin / (actual_bound + eps).
        ratio_margin_to_analytic: margin / (analytic_bound + eps).
        valid_mask: Boolean mask of valid tiles.
        valid_key_blocks_per_query: Number of valid key blocks per query row.
    """

    score_error: torch.Tensor
    margin: torch.Tensor
    actual_bound: torch.Tensor
    analytic_bound: torch.Tensor
    flip_mask: torch.Tensor
    oracle_crossing: torch.Tensor
    ratio_error_to_margin: torch.Tensor
    ratio_actual_to_error: torch.Tensor
    ratio_analytic_to_error: torch.Tensor
    ratio_margin_to_actual: torch.Tensor
    ratio_margin_to_analytic: torch.Tensor
    valid_mask: torch.Tensor
    valid_key_blocks_per_query: torch.Tensor


@dataclass
class DiagnosticQuantiles:
    """Quantile statistics for diagnostic quantities.

    Attributes:
        score_std: Standard deviation of reference scores.
        score_mean: Mean of reference scores.
        score_p01, score_p10, score_p50, score_p90, score_p99: Score quantiles.
        abs_score_p50, abs_score_p90, abs_score_p99: |score| quantiles.
        threshold_mean, threshold_std: Threshold statistics.
        threshold_p10, threshold_p50, threshold_p90: Threshold quantiles.
        abs_error_p50, abs_error_p90, abs_error_p99: |s - s_tilde| quantiles.
        actual_bound_p50, actual_bound_p90, actual_bound_p99: E_actual quantiles.
        analytic_bound_p50, analytic_bound_p90, analytic_bound_p99: E_analytic quantiles.
        margin_p10, margin_p50, margin_p90: |s - tau| quantiles.
        ratio_error_to_margin_p50, _p90, _p99: error/margin quantiles.
        ratio_actual_inflation_p50, _p90, _p99: actual_bound/error quantiles.
        ratio_analytic_inflation_p50, _p90, _p99: analytic_bound/error quantiles.
        ratio_margin_to_actual_p50, _p90, _p99: margin/actual_bound quantiles.
        ratio_margin_to_analytic_p50, _p90, _p99: margin/analytic_bound quantiles.
    """

    score_std: float
    score_mean: float
    score_p01: float
    score_p10: float
    score_p50: float
    score_p90: float
    score_p99: float
    abs_score_p50: float
    abs_score_p90: float
    abs_score_p99: float
    threshold_mean: float
    threshold_std: float
    threshold_p10: float
    threshold_p50: float
    threshold_p90: float
    abs_error_p50: float
    abs_error_p90: float
    abs_error_p99: float
    actual_bound_p50: float
    actual_bound_p90: float
    actual_bound_p99: float
    analytic_bound_p50: float
    analytic_bound_p90: float
    analytic_bound_p99: float
    margin_p10: float
    margin_p50: float
    margin_p90: float
    ratio_error_to_margin_p50: float
    ratio_error_to_margin_p90: float
    ratio_error_to_margin_p99: float
    ratio_actual_inflation_p50: float
    ratio_actual_inflation_p90: float
    ratio_actual_inflation_p99: float
    ratio_analytic_inflation_p50: float
    ratio_analytic_inflation_p90: float
    ratio_analytic_inflation_p99: float
    ratio_margin_to_actual_p50: float
    ratio_margin_to_actual_p90: float
    ratio_margin_to_actual_p99: float
    ratio_margin_to_analytic_p50: float
    ratio_margin_to_analytic_p90: float
    ratio_margin_to_analytic_p99: float


@dataclass
class RefinementDecomposition:
    """Decomposition of refinement into naive mismatch, oracle crossing,
    actual-L2 refinement, and analytic refinement.

    Attributes:
        naive_mismatch_rate: Fraction of valid tiles where naive INT8 mask
            differs from reference mask.
        oracle_crossing_rate: Fraction where the segment [s, s_tilde]
            contains the threshold.
        actual_refinement_rate: Fraction classified AMBIGUOUS by actual cert.
        analytic_refinement_rate: Fraction classified AMBIGUOUS by analytic cert.
    """

    naive_mismatch_rate: float
    oracle_crossing_rate: float
    actual_refinement_rate: float
    analytic_refinement_rate: float


@dataclass
class RowSubsetStats:
    """Statistics for a subset of query rows filtered by valid key count.

    Attributes:
        label: Description of the subset.
        num_rows: Number of query rows in this subset.
        naive_mismatch_rate: Mismatch rate for this subset.
        oracle_crossing_rate: Oracle crossing rate for this subset.
        actual_refinement_rate: Actual-L2 refinement rate for this subset.
        analytic_refinement_rate: Analytic refinement rate for this subset.
        actual_sparsity: Sparsity for this subset.
    """

    label: str
    num_rows: int
    naive_mismatch_rate: float
    oracle_crossing_rate: float
    actual_refinement_rate: float
    analytic_refinement_rate: float
    actual_sparsity: float


def compute_per_tile_diagnostics(
    reference_scores: torch.Tensor,
    quantized_scores: torch.Tensor,
    bounds_actual: ScoreBounds,
    bounds_analytic: ScoreBounds,
    threshold: torch.Tensor | float,
    *,
    valid_mask: torch.Tensor | None = None,
) -> PerTileDiagnostics:
    """Compute per-tile diagnostic quantities.

    Args:
        reference_scores: FP32 reference scores, shape [B, H, Q, K].
        quantized_scores: INT8 dequantized scores, shape [B, H, Q, K].
        bounds_actual: ScoreBounds using actual-L2 certificate.
        bounds_analytic: ScoreBounds using analytic certificate.
        threshold: Threshold used for masking.
        valid_mask: Optional boolean mask.

    Returns:
        PerTileDiagnostics with all diagnostic tensors.
    """
    ref = reference_scores.to(torch.float32)
    quant = quantized_scores.to(torch.float32)
    tau = torch.as_tensor(threshold, dtype=torch.float32, device=ref.device)

    if valid_mask is not None:
        vm = valid_mask.expand_as(ref)
    else:
        vm = torch.ones_like(ref, dtype=torch.bool)

    eps = torch.finfo(torch.float32).eps

    score_error = (ref - quant).abs()
    margin = (ref - tau).abs()
    actual_bound = bounds_actual.error_bound.to(torch.float32)
    analytic_bound = bounds_analytic.error_bound.to(torch.float32)

    # Naive mismatch: reference decision differs from quantized decision
    ref_keep = ref > tau
    naive_keep = quant > tau
    naive_mismatch = ref_keep != naive_keep

    # Oracle crossing: threshold lies in the open-closed interval
    # [min(s, s_tilde), max(s, s_tilde))
    # This must be equivalent to naive_mismatch for strict comparisons.
    lo = torch.min(ref, quant)
    hi = torch.max(ref, quant)
    oracle_crossing = (lo <= tau) & (tau < hi)

    # Ratios
    ratio_error_to_margin = score_error / (margin + eps)
    ratio_actual_to_error = actual_bound / (score_error + eps)
    ratio_analytic_to_error = analytic_bound / (score_error + eps)
    ratio_margin_to_actual = margin / (actual_bound + eps)
    ratio_margin_to_analytic = margin / (analytic_bound + eps)

    # Valid key blocks per query
    valid_key_blocks_per_query = vm.sum(dim=-1).to(torch.int64)  # [B, H, Q]

    return PerTileDiagnostics(
        score_error=score_error,
        margin=margin,
        actual_bound=actual_bound,
        analytic_bound=analytic_bound,
        flip_mask=naive_mismatch,
        oracle_crossing=oracle_crossing,
        ratio_error_to_margin=ratio_error_to_margin,
        ratio_actual_to_error=ratio_actual_to_error,
        ratio_analytic_to_error=ratio_analytic_to_error,
        ratio_margin_to_actual=ratio_margin_to_actual,
        ratio_margin_to_analytic=ratio_margin_to_analytic,
        valid_mask=vm,
        valid_key_blocks_per_query=valid_key_blocks_per_query,
    )


def compute_diagnostic_quantiles(
    diag: PerTileDiagnostics,
    reference_scores: torch.Tensor,
    threshold: torch.Tensor | float,
) -> DiagnosticQuantiles:
    """Compute quantile statistics from per-tile diagnostics.

    Args:
        diag: PerTileDiagnostics from compute_per_tile_diagnostics.
        reference_scores: FP32 reference scores.
        threshold: Threshold used for masking.

    Returns:
        DiagnosticQuantiles with all quantile values.
    """
    ref = reference_scores.to(torch.float32)
    tau = torch.as_tensor(threshold, dtype=torch.float32, device=ref.device)
    vm = diag.valid_mask

    def _q(tensor: torch.Tensor, q: float) -> float:
        return float(tensor[vm].quantile(q).item())

    margin_vals = diag.margin[vm]
    error_vals = diag.score_error[vm]
    actual_vals = diag.actual_bound[vm]
    analytic_vals = diag.analytic_bound[vm]
    ref_vals = ref[vm]
    abs_ref_vals = ref_vals.abs()

    # Threshold per-tile: expand tau to match
    tau_vals = (
        torch.full_like(ref_vals, tau.item()) if tau.dim() == 0
        else tau.expand_as(ref)[vm]
    )

    r_err_margin = diag.ratio_error_to_margin[vm]
    r_actual = diag.ratio_actual_to_error[vm]
    r_analytic = diag.ratio_analytic_to_error[vm]
    r_m_actual = diag.ratio_margin_to_actual[vm]
    r_m_analytic = diag.ratio_margin_to_analytic[vm]

    return DiagnosticQuantiles(
        score_std=float(ref_vals.std().item()),
        score_mean=float(ref_vals.mean().item()),
        score_p01=_q(ref, 0.01),
        score_p10=_q(ref, 0.10),
        score_p50=_q(ref, 0.50),
        score_p90=_q(ref, 0.90),
        score_p99=_q(ref, 0.99),
        abs_score_p50=float(abs_ref_vals.quantile(0.50).item()),
        abs_score_p90=float(abs_ref_vals.quantile(0.90).item()),
        abs_score_p99=float(abs_ref_vals.quantile(0.99).item()),
        threshold_mean=float(tau_vals.mean().item()),
        threshold_std=float(tau_vals.std().item()),
        threshold_p10=float(tau_vals.quantile(0.10).item()),
        threshold_p50=float(tau_vals.quantile(0.50).item()),
        threshold_p90=float(tau_vals.quantile(0.90).item()),
        abs_error_p50=float(error_vals.quantile(0.50).item()),
        abs_error_p90=float(error_vals.quantile(0.90).item()),
        abs_error_p99=float(error_vals.quantile(0.99).item()),
        actual_bound_p50=float(actual_vals.quantile(0.50).item()),
        actual_bound_p90=float(actual_vals.quantile(0.90).item()),
        actual_bound_p99=float(actual_vals.quantile(0.99).item()),
        analytic_bound_p50=float(analytic_vals.quantile(0.50).item()),
        analytic_bound_p90=float(analytic_vals.quantile(0.90).item()),
        analytic_bound_p99=float(analytic_vals.quantile(0.99).item()),
        margin_p10=float(margin_vals.quantile(0.10).item()),
        margin_p50=float(margin_vals.quantile(0.50).item()),
        margin_p90=float(margin_vals.quantile(0.90).item()),
        ratio_error_to_margin_p50=float(r_err_margin.quantile(0.50).item()),
        ratio_error_to_margin_p90=float(r_err_margin.quantile(0.90).item()),
        ratio_error_to_margin_p99=float(r_err_margin.quantile(0.99).item()),
        ratio_actual_inflation_p50=float(r_actual.quantile(0.50).item()),
        ratio_actual_inflation_p90=float(r_actual.quantile(0.90).item()),
        ratio_actual_inflation_p99=float(r_actual.quantile(0.99).item()),
        ratio_analytic_inflation_p50=float(r_analytic.quantile(0.50).item()),
        ratio_analytic_inflation_p90=float(r_analytic.quantile(0.90).item()),
        ratio_analytic_inflation_p99=float(r_analytic.quantile(0.99).item()),
        ratio_margin_to_actual_p50=float(r_m_actual.quantile(0.50).item()),
        ratio_margin_to_actual_p90=float(r_m_actual.quantile(0.90).item()),
        ratio_margin_to_actual_p99=float(r_m_actual.quantile(0.99).item()),
        ratio_margin_to_analytic_p50=float(r_m_analytic.quantile(0.50).item()),
        ratio_margin_to_analytic_p90=float(r_m_analytic.quantile(0.90).item()),
        ratio_margin_to_analytic_p99=float(r_m_analytic.quantile(0.99).item()),
    )


def compute_refinement_decomposition(
    diag: PerTileDiagnostics,
) -> RefinementDecomposition:
    """Compute the four-part refinement decomposition.

    Args:
        diag: PerTileDiagnostics from compute_per_tile_diagnostics.

    Returns:
        RefinementDecomposition with the four rates.
    """
    vm = diag.valid_mask
    n = int(vm.sum().item())
    if n == 0:
        return RefinementDecomposition(0.0, 0.0, 0.0, 0.0)

    # Naive mismatch = flip_mask & valid
    naive_mm = int((diag.flip_mask & vm).sum().item())

    # Oracle crossing & valid
    oracle_cross = int((diag.oracle_crossing & vm).sum().item())

    # Actual refinement: ambiguous under actual cert = margin <= actual_bound
    # (i.e., the interval [s-E, s+E] contains the threshold)
    actual_ambig = (diag.margin <= diag.actual_bound) & vm
    actual_ref = int(actual_ambig.sum().item())

    # Analytic refinement
    analytic_ambig = (diag.margin <= diag.analytic_bound) & vm
    analytic_ref = int(analytic_ambig.sum().item())

    return RefinementDecomposition(
        naive_mismatch_rate=naive_mm / n,
        oracle_crossing_rate=oracle_cross / n,
        actual_refinement_rate=actual_ref / n,
        analytic_refinement_rate=analytic_ref / n,
    )


def compute_row_subset_stats(
    diag: PerTileDiagnostics,
    ref_keep: torch.Tensor,
    *,
    min_valid_keys: int,
    label: str,
) -> RowSubsetStats:
    """Compute statistics for query rows with at least min_valid_keys valid keys.

    Args:
        diag: PerTileDiagnostics.
        ref_keep: Boolean reference mask, shape [B, H, Q, K].
        min_valid_keys: Minimum number of valid key blocks.
        label: Description of the subset.

    Returns:
        RowSubsetStats for this subset.
    """
    vm = diag.valid_mask
    # valid_key_blocks_per_query: [B, H, Q]
    row_mask = diag.valid_key_blocks_per_query >= min_valid_keys  # [B, H, Q]

    # Expand to [B, H, Q, K]
    subset_vm = vm & row_mask.unsqueeze(-1)

    n = int(subset_vm.sum().item())
    if n == 0:
        return RowSubsetStats(
            label=label, num_rows=int(row_mask.sum().item()),
            naive_mismatch_rate=0.0, oracle_crossing_rate=0.0,
            actual_refinement_rate=0.0, analytic_refinement_rate=0.0,
            actual_sparsity=0.0,
        )

    naive_mm = int((diag.flip_mask & subset_vm).sum().item())
    oracle_cross = int((diag.oracle_crossing & subset_vm).sum().item())
    actual_ambig = (diag.margin <= diag.actual_bound) & subset_vm
    analytic_ambig = (diag.margin <= diag.analytic_bound) & subset_vm
    ref_keep_subset = ref_keep & subset_vm
    actual_sp = 1.0 - int(ref_keep_subset.sum().item()) / n

    return RowSubsetStats(
        label=label,
        num_rows=int(row_mask.sum().item()),
        naive_mismatch_rate=naive_mm / n,
        oracle_crossing_rate=oracle_cross / n,
        actual_refinement_rate=int(actual_ambig.sum().item()) / n,
        analytic_refinement_rate=int(analytic_ambig.sum().item()) / n,
        actual_sparsity=actual_sp,
    )
