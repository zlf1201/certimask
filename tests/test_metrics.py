"""Tests for mask metrics and bound metrics."""

from __future__ import annotations

import pytest
import torch

from certimask.bounds import compute_score_bounds
from certimask.masking import (
    CertiMaskResult,
    certified_threshold_mask,
    naive_quantized_mask,
    reference_mask,
)
from certimask.metrics import (
    compute_bound_metrics,
    compute_mask_metrics,
)
from certimask.scoring import quantized_int8_scores, reference_scores


class TestMaskMetricsBasic:
    """Test 9: Metrics definition with small hand-crafted masks."""

    def test_basic_metrics(self) -> None:
        # Reference:  [T, T, F, F]
        # Naive:      [T, F, T, F]
        # CertiMask:  [T, T, F, F]  (exact match)
        reference = torch.tensor([[[[True, True, False, False]]]])
        naive = torch.tensor([[[[True, False, True, False]]]])

        # Create a minimal CertiMaskResult
        decisions = torch.tensor(
            [[[[
                1,  # KEEP
                1,  # KEEP (ambiguous, refined to KEEP)
                0,  # DROP
                0,  # DROP
            ]]]],
            dtype=torch.int64,
        )
        certified = CertiMaskResult(
            mask=torch.tensor([[[[True, True, False, False]]]]),
            decisions=decisions,
            certain_keep=torch.tensor([[[[True, False, False, False]]]]),
            certain_drop=torch.tensor([[[[False, False, True, True]]]]),
            ambiguous=torch.tensor([[[[False, True, False, False]]]]),
            refined_scores=None,
        )

        metrics = compute_mask_metrics(reference, naive, certified)

        assert metrics.valid_tiles == 4
        assert metrics.reference_kept == 2
        assert metrics.reference_dropped == 2
        assert metrics.actual_sparsity == 0.5

        # Naive mismatch: positions 1 and 2 differ
        assert metrics.naive_mismatch_count == 2
        assert metrics.naive_mismatch_rate == 0.5

        # False drop: ref=T, naive=F => position 1
        assert metrics.false_drop_count == 1
        assert metrics.false_drop_rate == 0.25

        # False keep: ref=F, naive=T => position 2
        assert metrics.false_keep_count == 1
        assert metrics.false_keep_rate == 0.25

        # CertiMask exact match
        assert metrics.certimask_mismatch_count == 0
        assert metrics.certimask_match_rate == 1.0

        # Classification counts
        assert metrics.certain_keep_count == 1
        assert metrics.certain_drop_count == 2
        assert metrics.ambiguous_count == 1
        assert metrics.refinement_rate == 0.25

    def test_actual_sparsity(self) -> None:
        reference = torch.tensor([[[[True, False, False, False, True]]]])
        naive = torch.tensor([[[[True, False, False, False, True]]]])
        certified = CertiMaskResult(
            mask=reference.clone(),
            decisions=torch.zeros(1, 1, 1, 5, dtype=torch.int64),
            certain_keep=torch.zeros(1, 1, 1, 5, dtype=torch.bool),
            certain_drop=torch.zeros(1, 1, 1, 5, dtype=torch.bool),
            ambiguous=torch.zeros(1, 1, 1, 5, dtype=torch.bool),
            refined_scores=None,
        )
        metrics = compute_mask_metrics(reference, naive, certified)
        assert metrics.actual_sparsity == pytest.approx(0.6)


class TestBoundMetrics:
    """Test bound tightness metrics."""

    def test_basic(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)

        ref = reference_scores(q, k, scale_by_sqrt_dim=True)
        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=True)

        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )

        metrics = compute_bound_metrics(
            ref, result.scores, bounds, 0.0
        )

        assert metrics.valid_tiles == 2 * 4 * 8 * 8
        assert metrics.violations == 0
        assert metrics.violation_rate == 0.0
        assert metrics.mean_error >= 0
        assert metrics.max_error >= 0
        assert metrics.mean_bound >= 0
        assert metrics.max_bound >= 0
        assert metrics.rho_mean >= 0
        assert metrics.rho_p50 >= 0
        assert metrics.rho_p90 >= 0
        assert metrics.rho_p99 >= 0
        assert metrics.rho_max >= 0

    def test_with_valid_mask(self) -> None:
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 4, 16)

        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)
        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
        )

        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        valid[0, 0, 0, 1:] = False  # Only first key for first query

        metrics = compute_bound_metrics(
            ref, result.scores, bounds, 0.0, valid_mask=valid
        )
        # valid mask: row 0 has 1 valid, rows 1-3 have 4 each = 13
        assert metrics.valid_tiles == 13


class TestMetricsInvalidInputs:
    """Test 10: Invalid inputs raise exceptions."""

    def test_shape_mismatch(self) -> None:
        ref = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        naive = torch.ones(1, 1, 4, 5, dtype=torch.bool)
        certified = CertiMaskResult(
            mask=torch.ones(1, 1, 4, 4, dtype=torch.bool),
            decisions=torch.zeros(1, 1, 4, 4, dtype=torch.int64),
            certain_keep=torch.zeros(1, 1, 4, 4, dtype=torch.bool),
            certain_drop=torch.zeros(1, 1, 4, 4, dtype=torch.bool),
            ambiguous=torch.zeros(1, 1, 4, 4, dtype=torch.bool),
            refined_scores=None,
        )
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_mask_metrics(ref, naive, certified)

    def test_no_valid_tiles(self) -> None:
        ref = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        naive = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        certified = CertiMaskResult(
            mask=torch.ones(1, 1, 4, 4, dtype=torch.bool),
            decisions=torch.zeros(1, 1, 4, 4, dtype=torch.int64),
            certain_keep=torch.zeros(1, 1, 4, 4, dtype=torch.bool),
            certain_drop=torch.zeros(1, 1, 4, 4, dtype=torch.bool),
            ambiguous=torch.zeros(1, 1, 4, 4, dtype=torch.bool),
            refined_scores=None,
        )
        valid = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match="No valid"):
            compute_mask_metrics(ref, naive, certified, valid_mask=valid)


class TestStressTestMetrics:
    """Stress test: verify metrics on random data."""

    @pytest.mark.parametrize("seed", range(20))
    def test_certimask_match_rate(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(1, 2, 8, 32, generator=gen)
        k = torch.randn(1, 2, 8, 32, generator=gen)

        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)
        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
        )

        threshold = ref.mean()
        ref_mask = reference_mask(ref, threshold)
        naive_mask = naive_quantized_mask(result.scores, threshold)
        cert_result = certified_threshold_mask(bounds, ref, threshold)

        metrics = compute_mask_metrics(ref_mask, naive_mask, cert_result)
        assert metrics.certimask_match_rate == 1.0
        assert metrics.certimask_mismatch_count == 0
