"""Tests for diagnostic tools."""

from __future__ import annotations

import torch

from certimask.bounds import compute_score_bounds
from certimask.diagnostics import (
    compute_diagnostic_quantiles,
    compute_per_tile_diagnostics,
    compute_refinement_decomposition,
    compute_row_subset_stats,
)
from certimask.masking import (
    reference_mask,
)
from certimask.scoring import quantized_int8_scores, reference_scores


class TestOracleCrossing:
    """Test oracle crossing detection."""

    def test_crossing_matches_flip(self) -> None:
        """Oracle crossing should be True wherever naive decision flips."""
        q = torch.randn(1, 1, 4, 32)
        k = torch.randn(1, 1, 4, 32)

        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)
        bounds_a = compute_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="actual",
        )
        bounds_b = compute_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="analytic",
        )

        threshold = ref.mean()
        diag = compute_per_tile_diagnostics(
            ref, result.scores, bounds_a, bounds_b, threshold,
        )

        # Every flip implies crossing
        assert torch.all(diag.flip_mask <= diag.oracle_crossing)

    def test_no_flip_means_no_crossing(self) -> None:
        """If s and s_tilde are on the same side of threshold, no crossing."""
        # Construct: s=10, s_tilde=9, threshold=5 => no crossing
        ref = torch.tensor([[[[10.0]]]])
        quant = torch.tensor([[[[9.0]]]])
        from certimask.bounds import ScoreBounds
        bounds = ScoreBounds(
            lower=torch.tensor([[[[8.0]]]]),
            upper=torch.tensor([[[[11.0]]]]),
            error_bound=torch.tensor([[[[1.0]]]]),
        )
        diag = compute_per_tile_diagnostics(ref, quant, bounds, bounds, 5.0)
        assert not diag.oracle_crossing[0, 0, 0, 0].item()
        assert not diag.flip_mask[0, 0, 0, 0].item()

    def test_crossing_detected(self) -> None:
        """If s > threshold but s_tilde < threshold, crossing detected."""
        ref = torch.tensor([[[[10.0]]]])
        quant = torch.tensor([[[[3.0]]]])
        from certimask.bounds import ScoreBounds
        bounds = ScoreBounds(
            lower=torch.tensor([[[[2.0]]]]),
            upper=torch.tensor([[[[11.0]]]]),
            error_bound=torch.tensor([[[[1.0]]]]),
        )
        diag = compute_per_tile_diagnostics(ref, quant, bounds, bounds, 5.0)
        assert diag.oracle_crossing[0, 0, 0, 0].item()
        assert diag.flip_mask[0, 0, 0, 0].item()


class TestRatios:
    """Test ratio computation correctness."""

    def test_zero_error_no_nan(self) -> None:
        """Zero error should not produce NaN in ratios."""
        ref = torch.tensor([[[[1.0, 2.0]]]])
        quant = ref.clone()  # zero error
        from certimask.bounds import ScoreBounds
        bounds = ScoreBounds(
            lower=torch.tensor([[[[0.5, 1.5]]]]),
            upper=torch.tensor([[[[1.5, 2.5]]]]),
            error_bound=torch.tensor([[[[0.5, 0.5]]]]),
        )
        diag = compute_per_tile_diagnostics(ref, quant, bounds, bounds, 0.0)
        assert torch.isfinite(diag.ratio_error_to_margin).all()
        assert torch.isfinite(diag.ratio_actual_to_error).all()
        assert torch.isfinite(diag.ratio_margin_to_actual).all()


class TestRefinementDecomposition:
    """Test the four-part decomposition."""

    def test_basic(self) -> None:
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 8, 32)

        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)
        bounds_a = compute_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="actual",
        )
        bounds_b = compute_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="analytic",
        )

        threshold = ref.mean()
        diag = compute_per_tile_diagnostics(
            ref, result.scores, bounds_a, bounds_b, threshold,
        )
        decomp = compute_refinement_decomposition(diag)

        # Oracle crossing >= naive mismatch (every flip is a crossing)
        assert decomp.oracle_crossing_rate >= decomp.naive_mismatch_rate - 1e-6
        # Analytic refinement >= actual refinement
        assert decomp.analytic_refinement_rate >= decomp.actual_refinement_rate - 1e-6


class TestRowSubsets:
    """Test row subset filtering."""

    def test_all_subset(self) -> None:
        """min_valid_keys=1 should include all valid rows."""
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 8, 32)
        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)
        bounds = compute_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="actual",
        )
        threshold = ref.mean()
        diag = compute_per_tile_diagnostics(
            ref, result.scores, bounds, bounds, threshold,
        )
        ref_mask = reference_mask(ref, threshold)
        stats = compute_row_subset_stats(diag, ref_mask, min_valid_keys=1, label="all")
        assert stats.num_rows > 0


class TestDiagnosticQuantiles:
    """Test quantile computation."""

    def test_no_nan(self) -> None:
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 8, 32)
        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)
        bounds = compute_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="actual",
        )
        threshold = ref.mean()
        diag = compute_per_tile_diagnostics(
            ref, result.scores, bounds, bounds, threshold,
        )
        qntls = compute_diagnostic_quantiles(diag, ref, threshold)
        # All values should be finite
        for field_name, val in vars(qntls).items():
            assert val == val, f"{field_name} is NaN"  # NaN != NaN
            assert abs(val) < float("inf"), f"{field_name} is Inf"
