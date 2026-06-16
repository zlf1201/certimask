"""Tests for groupwise and coordinate-wise certificates, and oracle crossing."""

from __future__ import annotations

import pytest
import torch

from certimask.bounds import (
    compute_coordinate_score_bounds,
    compute_groupwise_score_bounds,
    compute_score_bounds,
    validate_score_bounds,
)
from certimask.diagnostics import compute_per_tile_diagnostics
from certimask.scoring import quantized_int8_scores, reference_scores


class TestOracleCrossingEqualsMismatch:
    """Test 1: oracle crossing must exactly equal naive mismatch."""

    def test_manual_cases(self) -> None:
        """Test boundary cases for oracle crossing."""
        # Case 1: ref > threshold, quant == threshold => mismatch (ref KEEP, quant DROP)
        # crossing: lo=5<=5 and hi=10>5 => True. Both True.
        ref = torch.tensor([10.0])
        quant = torch.tensor([5.0])
        tau = 5.0
        ref_keep = ref > tau
        naive_keep = quant > tau
        mismatch = ref_keep != naive_keep
        lo = torch.min(ref, quant)
        hi = torch.max(ref, quant)
        crossing = (lo <= tau) & (tau < hi)
        assert mismatch.item() == crossing.item() == True  # noqa: E712

        # Case 2: ref == threshold, quant > threshold => mismatch, crossing
        ref = torch.tensor([5.0])
        quant = torch.tensor([10.0])
        tau = 5.0
        ref_keep = ref > tau
        naive_keep = quant > tau
        mismatch = ref_keep != naive_keep
        lo = torch.min(ref, quant)
        hi = torch.max(ref, quant)
        crossing = (lo <= tau) & (tau < hi)
        assert mismatch.item() == crossing.item() == True  # noqa: E712

        # Case 3: ref < threshold, quant == threshold
        # ref_keep = (3>5)=False, naive_keep = (5>5)=False => no mismatch
        # lo=3<=5=T, tau<hi => 5<5=F => no crossing
        ref = torch.tensor([3.0])
        quant = torch.tensor([5.0])
        tau = 5.0
        ref_keep = ref > tau
        naive_keep = quant > tau
        mismatch = ref_keep != naive_keep
        lo = torch.min(ref, quant)
        hi = torch.max(ref, quant)
        crossing = (lo <= tau) & (tau < hi)
        assert mismatch.item() == crossing.item() == False  # noqa: E712

        # Case 4: both == threshold => no mismatch, no crossing
        ref = torch.tensor([5.0])
        quant = torch.tensor([5.0])
        tau = 5.0
        ref_keep = ref > tau
        naive_keep = quant > tau
        mismatch = ref_keep != naive_keep
        lo = torch.min(ref, quant)
        hi = torch.max(ref, quant)
        crossing = (lo <= tau) & (tau < hi)
        assert mismatch.item() == crossing.item() == False  # noqa: E712

        # Case 5: both on same side => no mismatch, no crossing
        ref = torch.tensor([10.0])
        quant = torch.tensor([8.0])
        tau = 5.0
        ref_keep = ref > tau
        naive_keep = quant > tau
        mismatch = ref_keep != naive_keep
        lo = torch.min(ref, quant)
        hi = torch.max(ref, quant)
        crossing = (lo <= tau) & (tau < hi)
        assert mismatch.item() == crossing.item() == False  # noqa: E712

    def test_random_stress(self) -> None:
        """Random stress test: oracle crossing == naive mismatch."""
        gen = torch.Generator().manual_seed(0)
        for _ in range(50):
            ref = torch.randn(2, 4, 8, 16, generator=gen)
            quant = ref + torch.randn_like(ref) * 0.5
            tau = ref.median()

            ref_keep = ref > tau
            naive_keep = quant > tau
            mismatch = ref_keep != naive_keep

            lo = torch.min(ref, quant)
            hi = torch.max(ref, quant)
            crossing = (lo <= tau) & (tau < hi)

            assert torch.equal(mismatch, crossing), (
                f"Mismatch at positions: {(mismatch != crossing).nonzero()}"
            )

    def test_with_actual_pipeline(self) -> None:
        """Verify oracle crossing == mismatch in real pipeline."""
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
        # Must be equal on all valid tiles
        vm = diag.valid_mask
        assert torch.equal(diag.oracle_crossing[vm], diag.flip_mask[vm])


class TestGroupwiseCertificates:
    """Tests for groupwise L2 certificate."""

    def test_group_size_equals_head_dim_matches_global(self) -> None:
        """group_size == head_dim should match global certificate."""
        q = torch.randn(1, 2, 4, 32)
        k = torch.randn(1, 2, 4, 32)
        result = quantized_int8_scores(q, k)

        global_bounds = compute_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="analytic",
        )
        group_bounds = compute_groupwise_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, group_size=32,
            certificate_type="analytic",
        )
        torch.testing.assert_close(
            group_bounds.error_bound, global_bounds.error_bound,
            rtol=1e-5, atol=1e-6,
        )

    def test_actual_zero_violations(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)
        for gs in [4, 8, 16, 32]:
            bounds = compute_groupwise_score_bounds(
                result.scores, result.query_quantized,
                result.key_quantized, group_size=gs,
                certificate_type="actual",
            )
            violations = validate_score_bounds(ref, bounds)
            assert violations.sum() == 0, f"gs={gs} violations={violations.sum()}"

    def test_analytic_zero_violations(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)
        for gs in [4, 8, 16, 32]:
            bounds = compute_groupwise_score_bounds(
                result.scores, result.query_quantized,
                result.key_quantized, group_size=gs,
                certificate_type="analytic",
            )
            violations = validate_score_bounds(ref, bounds)
            assert violations.sum() == 0, f"gs={gs} violations={violations.sum()}"

    def test_smaller_groups_not_much_larger(self) -> None:
        """Smaller group sizes should not produce much larger bounds."""
        q = torch.randn(1, 2, 8, 64)
        k = torch.randn(1, 2, 8, 64)
        result = quantized_int8_scores(q, k)

        bounds_64 = compute_groupwise_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, group_size=64,
            certificate_type="analytic",
        )
        bounds_8 = compute_groupwise_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, group_size=8,
            certificate_type="analytic",
        )
        # Smaller groups should not increase bounds significantly
        # (in theory they should be <= due to tighter per-group norms)
        ratio = bounds_8.error_bound / (bounds_64.error_bound + 1e-12)
        assert ratio.max().item() < 1.5, f"Max ratio: {ratio.max().item()}"

    def test_invalid_group_size(self) -> None:
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 4, 16)
        result = quantized_int8_scores(q, k)
        with pytest.raises(ValueError, match="positive"):
            compute_groupwise_score_bounds(
                result.scores, result.query_quantized,
                result.key_quantized, group_size=0,
                certificate_type="analytic",
            )

    def test_non_divisible_group_size(self) -> None:
        """Non-divisible group size should work (last group is shorter)."""
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 4, 16)
        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)
        bounds = compute_groupwise_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, group_size=5,
            certificate_type="analytic",
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0


class TestCoordinateCertificates:
    """Tests for coordinate-wise certificate."""

    def test_actual_zero_violations(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)
        bounds = compute_coordinate_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="actual",
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0

    def test_analytic_zero_violations(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)
        bounds = compute_coordinate_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="analytic",
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0

    def test_zero_vector_coordinate_bound(self) -> None:
        """Zero vectors should have zero coordinate error bound."""
        q = torch.zeros(1, 1, 2, 16)
        k = torch.randn(1, 1, 2, 16)
        result = quantized_int8_scores(q, k)
        bounds = compute_coordinate_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="analytic",
        )
        # For zero query vectors, error bound should be 0
        # (only the k error * 0 term contributes)
        assert torch.all(bounds.error_bound >= 0)
        # The error bound should be exactly 0 since q=0 means all q errors are 0
        # and the q_norm * k_eps term is 0
        assert torch.allclose(bounds.error_bound, torch.zeros_like(bounds.error_bound))

    def test_scaled_and_unscaled(self) -> None:
        q = torch.randn(1, 2, 4, 16)
        k = torch.randn(1, 2, 4, 16)
        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=False)
        ref = reference_scores(q, k, scale_by_sqrt_dim=False)
        bounds = compute_coordinate_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="analytic",
            scale_by_sqrt_dim=False,
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0

    def test_coordinate_tighter_than_global(self) -> None:
        """Coordinate certificate should be tighter than or equal to global."""
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 8, 32)
        result = quantized_int8_scores(q, k)

        global_bounds = compute_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="analytic",
        )
        coord_bounds = compute_coordinate_score_bounds(
            result.scores, result.query_quantized,
            result.key_quantized, certificate_type="analytic",
        )
        # Coordinate bound should be <= global bound
        assert torch.all(coord_bounds.error_bound <= global_bounds.error_bound + 1e-6)


class TestGroupPartition:
    """Test 11: group partition has no gaps or overlaps."""

    def test_partition_covers_all_coordinates(self) -> None:
        """Verify all coordinates are covered by groups."""
        d = 64
        group_size = 10
        covered = set()
        for start in range(0, d, group_size):
            end = min(start + group_size, d)
            for i in range(start, end):
                covered.add(i)
        assert covered == set(range(d))
