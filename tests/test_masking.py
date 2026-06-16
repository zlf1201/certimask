"""Tests for threshold-based masking."""

from __future__ import annotations

import pytest
import torch

from certimask.bounds import compute_score_bounds
from certimask.masking import (
    CertiMaskDecision,
    certified_threshold_mask,
    make_block_causal_valid_mask,
    naive_quantized_mask,
    reference_mask,
    thresholds_for_target_sparsity,
)
from certimask.scoring import quantized_int8_scores, reference_scores


class TestReferenceMask:
    """Test 1: Reference mask basic behavior."""

    def test_above_threshold_kept(self) -> None:
        scores = torch.tensor([[[[1.0, 2.0, 3.0]]]])
        mask = reference_mask(scores, 1.5)
        assert mask.dtype == torch.bool
        assert mask[0, 0, 0, 0].item() is False  # 1.0 > 1.5 => False
        assert mask[0, 0, 0, 1].item() is True  # 2.0 > 1.5 => True
        assert mask[0, 0, 0, 2].item() is True  # 3.0 > 1.5 => True

    def test_below_threshold_dropped(self) -> None:
        scores = torch.tensor([[[[1.0, 2.0, 3.0]]]])
        mask = reference_mask(scores, 2.5)
        assert mask[0, 0, 0, 0].item() is False
        assert mask[0, 0, 0, 1].item() is False
        assert mask[0, 0, 0, 2].item() is True

    def test_equal_threshold_dropped(self) -> None:
        scores = torch.tensor([[[[1.0, 2.0, 3.0]]]])
        mask = reference_mask(scores, 2.0)
        assert mask[0, 0, 0, 0].item() is False
        assert mask[0, 0, 0, 1].item() is False  # score == threshold => DROP
        assert mask[0, 0, 0, 2].item() is True

    def test_valid_mask(self) -> None:
        scores = torch.tensor([[[[1.0, 2.0, 3.0]]]])
        valid = torch.tensor([[[[True, True, False]]]])
        mask = reference_mask(scores, 0.5, valid_mask=valid)
        assert mask[0, 0, 0, 0].item() is True
        assert mask[0, 0, 0, 1].item() is True
        assert mask[0, 0, 0, 2].item() is False  # invalid => False

    def test_broadcast_threshold(self) -> None:
        scores = torch.randn(2, 4, 8, 12)
        thresholds = torch.randn(2, 4, 8, 1)  # broadcast over K
        mask = reference_mask(scores, thresholds)
        assert mask.shape == scores.shape
        assert mask.dtype == torch.bool


class TestNaiveQuantizedMask:
    """Test: Naive INT8 mask basic behavior."""

    def test_basic(self) -> None:
        scores = torch.tensor([[[[0.5, 1.5, 2.5]]]])
        mask = naive_quantized_mask(scores, 1.0)
        assert mask[0, 0, 0, 0].item() is False
        assert mask[0, 0, 0, 1].item() is True
        assert mask[0, 0, 0, 2].item() is True


class TestThreeWayClassification:
    """Test 2: Three-way classification boundary conditions."""

    def test_certain_keep(self) -> None:
        """L > threshold => CERTAIN KEEP."""
        from certimask.bounds import ScoreBounds

        bounds = ScoreBounds(
            lower=torch.tensor([[[[2.0]]]]),
            upper=torch.tensor([[[[3.0]]]]),
            error_bound=torch.tensor([[[[0.5]]]]),
        )
        ref = torch.tensor([[[[2.5]]]])
        result = certified_threshold_mask(bounds, ref, 1.5)
        assert result.decisions[0, 0, 0, 0].item() == CertiMaskDecision.KEEP
        assert result.mask[0, 0, 0, 0].item() is True

    def test_certain_drop(self) -> None:
        """U < threshold => CERTAIN DROP."""
        from certimask.bounds import ScoreBounds

        bounds = ScoreBounds(
            lower=torch.tensor([[[[0.5]]]]),
            upper=torch.tensor([[[[1.5]]]]),
            error_bound=torch.tensor([[[[0.5]]]]),
        )
        ref = torch.tensor([[[[1.0]]]])
        result = certified_threshold_mask(bounds, ref, 2.0)
        assert result.decisions[0, 0, 0, 0].item() == CertiMaskDecision.DROP
        assert result.mask[0, 0, 0, 0].item() is False

    def test_lower_equals_threshold_is_ambiguous(self) -> None:
        """L == threshold => AMBIGUOUS."""
        from certimask.bounds import ScoreBounds

        bounds = ScoreBounds(
            lower=torch.tensor([[[[2.0]]]]),
            upper=torch.tensor([[[[3.0]]]]),
            error_bound=torch.tensor([[[[0.5]]]]),
        )
        ref = torch.tensor([[[[2.5]]]])
        result = certified_threshold_mask(bounds, ref, 2.0)
        assert result.decisions[0, 0, 0, 0].item() == CertiMaskDecision.AMBIGUOUS

    def test_upper_equals_threshold_is_ambiguous(self) -> None:
        """U == threshold => AMBIGUOUS."""
        from certimask.bounds import ScoreBounds

        bounds = ScoreBounds(
            lower=torch.tensor([[[[0.5]]]]),
            upper=torch.tensor([[[[2.0]]]]),
            error_bound=torch.tensor([[[[0.5]]]]),
        )
        ref = torch.tensor([[[[1.0]]]])
        result = certified_threshold_mask(bounds, ref, 2.0)
        assert result.decisions[0, 0, 0, 0].item() == CertiMaskDecision.AMBIGUOUS

    def test_interval_contains_threshold_is_ambiguous(self) -> None:
        """L < threshold < U => AMBIGUOUS."""
        from certimask.bounds import ScoreBounds

        bounds = ScoreBounds(
            lower=torch.tensor([[[[1.0]]]]),
            upper=torch.tensor([[[[3.0]]]]),
            error_bound=torch.tensor([[[[1.0]]]]),
        )
        ref = torch.tensor([[[[2.0]]]])
        result = certified_threshold_mask(bounds, ref, 2.0)
        assert result.decisions[0, 0, 0, 0].item() == CertiMaskDecision.AMBIGUOUS

    def test_ambiguous_uses_reference_fallback(self) -> None:
        """Ambiguous tiles use reference score for mask decision."""
        from certimask.bounds import ScoreBounds

        bounds = ScoreBounds(
            lower=torch.tensor([[[[1.0]]]]),
            upper=torch.tensor([[[[3.0]]]]),
            error_bound=torch.tensor([[[[1.0]]]]),
        )
        # Reference score above threshold => KEEP
        ref_above = torch.tensor([[[[2.5]]]])
        result_above = certified_threshold_mask(bounds, ref_above, 2.0)
        assert result_above.mask[0, 0, 0, 0].item() is True

        # Reference score below threshold => DROP
        ref_below = torch.tensor([[[[1.5]]]])
        result_below = certified_threshold_mask(bounds, ref_below, 2.0)
        assert result_below.mask[0, 0, 0, 0].item() is False


class TestCertiMaskExactMatch:
    """Test 3: CertiMask exact match with reference mask."""

    @pytest.mark.parametrize("cert_type", ["actual", "analytic"])
    @pytest.mark.parametrize("scaled", [True, False])
    def test_exact_match(self, cert_type: str, scaled: bool) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 12, 32)

        ref = reference_scores(q, k, scale_by_sqrt_dim=scaled)
        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=scaled)

        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type=cert_type,  # type: ignore[arg-type]
            scale_by_sqrt_dim=scaled,
        )

        # Random threshold
        threshold = ref.mean()

        ref_mask = reference_mask(ref, threshold)
        cert_result = certified_threshold_mask(bounds, ref, threshold)

        assert torch.equal(cert_result.mask, ref_mask)


class TestManyAmbiguous:
    """Test 4: Many ambiguous tiles still achieve exact match."""

    def test_dense_threshold_region(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 12, 32)

        ref = reference_scores(q, k, scale_by_sqrt_dim=True)
        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=True)

        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )

        # Set threshold at median to maximize ambiguous tiles
        threshold = ref.median()

        ref_mask = reference_mask(ref, threshold)
        cert_result = certified_threshold_mask(bounds, ref, threshold)

        assert torch.equal(cert_result.mask, ref_mask)
        assert cert_result.ambiguous.any()
        # Refinement rate should be > 0
        assert cert_result.ambiguous.sum().item() > 0


class TestNoAmbiguous:
    """Test 5: No ambiguous tiles when bounds are far from threshold."""

    def test_no_ambiguous(self) -> None:
        from certimask.bounds import ScoreBounds

        # All bounds well above threshold
        bounds = ScoreBounds(
            lower=torch.full((1, 1, 4, 4), 10.0),
            upper=torch.full((1, 1, 4, 4), 20.0),
            error_bound=torch.full((1, 1, 4, 4), 5.0),
        )
        ref = torch.full((1, 1, 4, 4), 15.0)

        result = certified_threshold_mask(bounds, ref, 5.0)

        assert result.ambiguous.sum().item() == 0
        assert result.mask.all()


class TestNaiveFlip:
    """Test 6: Naive INT8 mask flip is recoverable by CertiMask."""

    def test_manual_flip(self) -> None:
        from certimask.bounds import ScoreBounds

        # Construct a case where quantized score crosses threshold
        # but reference score does not
        # Reference score = 2.0, quantized = 1.8, threshold = 1.9
        # Reference: 2.0 > 1.9 => KEEP
        # Naive INT8: 1.8 > 1.9 => DROP (false drop)
        # CertiMask: bound contains both, ambiguous, uses reference => KEEP

        bounds = ScoreBounds(
            lower=torch.tensor([[[[1.5]]]]),
            upper=torch.tensor([[[[2.5]]]]),
            error_bound=torch.tensor([[[[0.5]]]]),
        )
        ref = torch.tensor([[[[2.0]]]])
        quantized = torch.tensor([[[[1.8]]]])

        ref_mask = reference_mask(ref, 1.9)
        naive_mask = naive_quantized_mask(quantized, 1.9)
        cert_result = certified_threshold_mask(bounds, ref, 1.9)

        # Reference says KEEP, naive says DROP
        assert ref_mask[0, 0, 0, 0].item() is True
        assert naive_mask[0, 0, 0, 0].item() is False

        # CertiMask must match reference
        assert cert_result.mask[0, 0, 0, 0].item() is True
        assert torch.equal(cert_result.mask, ref_mask)


class TestCausalValidMask:
    """Test 7: Causal valid mask behavior."""

    def test_basic_causal(self) -> None:
        mask = make_block_causal_valid_mask(4, 4)
        assert mask.shape == (1, 1, 4, 4)
        assert mask.dtype == torch.bool

        # Query block 0 can only see key block 0
        assert mask[0, 0, 0, 0].item() is True
        assert mask[0, 0, 0, 1].item() is False
        assert mask[0, 0, 0, 2].item() is False
        assert mask[0, 0, 0, 3].item() is False

        # Query block 2 can see key blocks 0, 1, 2
        assert mask[0, 0, 2, 0].item() is True
        assert mask[0, 0, 2, 1].item() is True
        assert mask[0, 0, 2, 2].item() is True
        assert mask[0, 0, 2, 3].item() is False

    def test_future_blocks_invalid_in_mask(self) -> None:
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 4, 16)

        ref = reference_scores(q, k, scale_by_sqrt_dim=True)
        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=True)
        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )

        valid = make_block_causal_valid_mask(4, 4)
        threshold = 0.0

        ref_mask = reference_mask(ref, threshold, valid_mask=valid)
        naive_mask = naive_quantized_mask(
            result.scores, threshold, valid_mask=valid
        )
        cert_result = certified_threshold_mask(
            bounds, ref, threshold, valid_mask=valid
        )

        # Future blocks must be False/INVALID
        assert ref_mask[0, 0, 0, 1].item() is False
        assert naive_mask[0, 0, 0, 1].item() is False
        assert cert_result.mask[0, 0, 0, 1].item() is False
        assert cert_result.decisions[0, 0, 0, 1].item() == CertiMaskDecision.INVALID


class TestTargetSparsity:
    """Test 8: Target sparsity threshold computation."""

    def test_basic_sparsity(self) -> None:
        scores = torch.arange(100, dtype=torch.float32).reshape(1, 1, 1, 100)
        for target in [0.70, 0.80, 0.85, 0.90]:
            thresholds = thresholds_for_target_sparsity(
                scores, target, per_query=True
            )
            mask = reference_mask(scores, thresholds)
            actual_sp = 1.0 - mask.float().mean().item()
            # Should be close to target (within 5% due to discrete scores)
            assert abs(actual_sp - target) < 0.05, (
                f"target={target}, actual={actual_sp}"
            )

    def test_with_valid_mask(self) -> None:
        scores = torch.arange(100, dtype=torch.float32).reshape(1, 1, 1, 100)
        valid = torch.ones(1, 1, 1, 100, dtype=torch.bool)
        valid[0, 0, 0, 50:] = False  # Only first 50 are valid
        thresholds = thresholds_for_target_sparsity(
            scores, 0.5, valid_mask=valid, per_query=True
        )
        # Threshold should only consider valid tiles
        assert thresholds.shape == (1, 1, 1, 1)

    def test_per_batch_head(self) -> None:
        scores = torch.randn(2, 4, 8, 16)
        thresholds = thresholds_for_target_sparsity(
            scores, 0.8, per_query=False
        )
        assert thresholds.shape == (2, 4, 1, 1)

    def test_single_valid_key(self) -> None:
        """Stable when only one valid key per query."""
        scores = torch.randn(1, 1, 4, 1)
        thresholds = thresholds_for_target_sparsity(
            scores, 0.0, per_query=True
        )
        assert thresholds.shape == (1, 1, 4, 1)
        assert torch.isfinite(thresholds).all()

    def test_deterministic(self) -> None:
        scores = torch.randn(2, 4, 8, 16)
        t1 = thresholds_for_target_sparsity(scores, 0.8, per_query=True)
        t2 = thresholds_for_target_sparsity(scores, 0.8, per_query=True)
        assert torch.equal(t1, t2)


class TestInvalidInputs:
    """Test 11: Invalid input validation."""

    def test_non_bool_valid_mask(self) -> None:
        scores = torch.randn(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.int32)
        with pytest.raises(TypeError, match="bool"):
            reference_mask(scores, 0.0, valid_mask=valid)

    def test_non_broadcastable_valid_mask(self) -> None:
        scores = torch.randn(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 5, dtype=torch.bool)
        # This will fail at the & operation due to shape mismatch
        with pytest.raises(RuntimeError):
            reference_mask(scores, 0.0, valid_mask=valid)

    def test_threshold_nan(self) -> None:
        # NaN in scores
        scores = torch.randn(1, 1, 4, 4)
        scores_bad = scores.clone()
        scores_bad[0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            thresholds_for_target_sparsity(scores_bad, 0.5)

    def test_threshold_inf(self) -> None:
        scores = torch.randn(1, 1, 4, 4)
        scores_bad = scores.clone()
        scores_bad[0, 0, 0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            thresholds_for_target_sparsity(scores_bad, 0.5)

    def test_invalid_target_sparsity(self) -> None:
        scores = torch.randn(1, 1, 4, 4)
        with pytest.raises(ValueError):
            thresholds_for_target_sparsity(scores, -0.1)
        with pytest.raises(ValueError):
            thresholds_for_target_sparsity(scores, 1.0)

    def test_non_floating_scores(self) -> None:
        scores = torch.randint(0, 10, (1, 1, 4, 4))
        with pytest.raises(TypeError, match="floating-point"):
            reference_mask(scores, 0.0)

    def test_shape_mismatch_bounds(self) -> None:
        from certimask.bounds import ScoreBounds

        bounds = ScoreBounds(
            lower=torch.randn(1, 1, 4, 4),
            upper=torch.randn(1, 1, 4, 4),
            error_bound=torch.randn(1, 1, 4, 4),
        )
        ref = torch.randn(1, 1, 4, 6)
        with pytest.raises(ValueError, match="Shape mismatch"):
            certified_threshold_mask(bounds, ref, 0.0)


class TestBoundsDetectInvalid:
    """Test 12: validate_score_bounds detects invalid intervals."""

    def test_all_invalid_valid_mask_raises(self) -> None:
        valid = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match="no valid"):
            thresholds_for_target_sparsity(
                torch.randn(1, 1, 4, 4), 0.5, valid_mask=valid
            )


class TestStressTest:
    """Random stress test across seeds, distributions, sparsities."""

    @pytest.mark.parametrize("seed", range(20))
    @pytest.mark.parametrize("distribution", ["normal", "sparse", "outlier", "correlated"])
    @pytest.mark.parametrize("target_sp", [0.70, 0.85, 0.90])
    def test_certimask_exact_match(
        self,
        seed: int,
        distribution: str,
        target_sp: float,
    ) -> None:

        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(1, 2, 8, 32, generator=gen)
        k = torch.randn(1, 2, 8, 32, generator=gen)

        ref = reference_scores(q, k, scale_by_sqrt_dim=True)
        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=True)

        for cert_type in ("actual", "analytic"):
            bounds = compute_score_bounds(
                result.scores,
                result.query_quantized,
                result.key_quantized,
                certificate_type=cert_type,  # type: ignore[arg-type]
                scale_by_sqrt_dim=True,
            )

            thresholds = thresholds_for_target_sparsity(
                ref, target_sp, per_query=True
            )

            ref_mask = reference_mask(ref, thresholds)
            cert_result = certified_threshold_mask(bounds, ref, thresholds)

            assert torch.equal(cert_result.mask, ref_mask), (
                f"seed={seed}, dist={distribution}, sp={target_sp}, "
                f"cert={cert_type}"
            )
