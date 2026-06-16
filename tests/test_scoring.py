"""Tests for block score computation."""

from __future__ import annotations

import pytest
import torch

from certimask.scoring import quantized_int8_scores, reference_scores


class TestReferenceScoresShapeAndDtype:
    """Test 1: Reference scores shape and dtype."""

    def test_scaled(self) -> None:
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 12, 16)
        scores = reference_scores(q, k, scale_by_sqrt_dim=True)

        assert scores.shape == (2, 4, 8, 12)
        assert scores.dtype == torch.float32

    def test_unscaled(self) -> None:
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 12, 16)
        scores = reference_scores(q, k, scale_by_sqrt_dim=False)

        assert scores.shape == (2, 4, 8, 12)
        assert scores.dtype == torch.float32


class TestReferenceScoresAgainstManual:
    """Test 2: Reference scores against manual computation."""

    def test_manual_dot_product(self) -> None:
        # Small deterministic vectors
        q = torch.tensor([[[[1.0, 2.0, 3.0]]]])  # [1,1,1,3]
        k = torch.tensor([[[[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])  # [1,1,2,3]

        scores = reference_scores(q, k, scale_by_sqrt_dim=False)
        # Manual: [1*4+2*5+3*6, 1*7+2*8+3*9] = [32, 50]
        expected = torch.tensor([[[[32.0, 50.0]]]])
        torch.testing.assert_close(scores, expected, rtol=1e-6, atol=1e-6)

    def test_manual_scaled(self) -> None:
        q = torch.tensor([[[[1.0, 2.0, 3.0]]]])  # [1,1,1,3]
        k = torch.tensor([[[[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])  # [1,1,2,3]

        scores = reference_scores(q, k, scale_by_sqrt_dim=True)
        # Manual: [32, 50] / sqrt(3)
        sqrt3 = torch.sqrt(torch.tensor(3.0))
        expected = torch.tensor([[[[32.0, 50.0]]]]) / sqrt3
        torch.testing.assert_close(scores, expected, rtol=1e-6, atol=1e-6)


class TestQuantizedScoresMatchDequantized:
    """Test 3: Integer dot * scale matches dequantized dot product."""

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_cross_validation(self, dtype: torch.dtype) -> None:
        q = torch.randn(2, 4, 8, 32).to(dtype)
        k = torch.randn(2, 4, 12, 32).to(dtype)

        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=True)

        # Method 2: dequantized dot product
        q_deq = result.query_quantized.dequantized.to(torch.float32)
        k_deq = result.key_quantized.dequantized.to(torch.float32)
        scores_dequant = torch.einsum("bhqd,bhkd->bhqk", q_deq, k_deq)
        d = q.shape[-1]
        scores_dequant = scores_dequant / torch.sqrt(
            torch.tensor(float(d), dtype=torch.float32)
        )

        # They should match closely
        torch.testing.assert_close(
            result.scores,
            scores_dequant,
            rtol=1e-4,
            atol=1e-4,
        )


class TestIntegerDotNoOverflow:
    """Test 4: Integer dot product does not overflow int8/int16."""

    def test_large_dot_product(self) -> None:
        # head_dim=128, values near 127 -> dot product can be ~128*127*127 = ~2M
        # which exceeds int16 range (32767) but fits in int32
        q = torch.ones(1, 1, 2, 128) * 100.0
        k = torch.ones(1, 1, 2, 128) * 100.0

        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=False)

        # The integer dot should be large but correct
        # Each quantized value will be 127, so dot = 128 * 127 * 127 = 2064384
        # This exceeds int16 range but fits in int32
        assert result.integer_dot.dtype == torch.int32
        assert torch.isfinite(result.scores).all()

        # Verify against int64 reference
        q_int64 = result.query_quantized.values.to(torch.int64)
        k_int64 = result.key_quantized.values.to(torch.int64)
        expected_dot = torch.einsum("bhqd,bhkd->bhqk", q_int64, k_int64)
        torch.testing.assert_close(
            result.integer_dot.to(torch.int64), expected_dot, rtol=0, atol=0
        )


class TestActualBoundCoversReference:
    """Test 5: Actual certificate covers reference scores."""

    def test_random(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 12, 32)

        ref = reference_scores(q, k, scale_by_sqrt_dim=True)
        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=True)

        from certimask.bounds import compute_score_bounds, validate_score_bounds

        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="actual",
            scale_by_sqrt_dim=True,
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0


class TestAnalyticBoundCoversReference:
    """Test 6: Analytic certificate covers reference scores."""

    def _check(self, q: torch.Tensor, k: torch.Tensor) -> None:
        from certimask.bounds import compute_score_bounds, validate_score_bounds

        ref = reference_scores(q, k, scale_by_sqrt_dim=True)
        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=True)

        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0, f"violations: {violations.sum().item()}"

    def test_normal_random(self) -> None:
        self._check(torch.randn(2, 4, 8, 32), torch.randn(2, 4, 12, 32))

    def test_small_values(self) -> None:
        self._check(
            torch.randn(2, 4, 8, 32) * 1e-4,
            torch.randn(2, 4, 12, 32) * 1e-4,
        )

    def test_large_values(self) -> None:
        self._check(
            torch.randn(2, 4, 8, 32) * 1e2,
            torch.randn(2, 4, 12, 32) * 1e2,
        )

    def test_sparse_vectors(self) -> None:
        q = torch.zeros(2, 4, 8, 32)
        k = torch.zeros(2, 4, 12, 32)
        # Make sparse: only a few non-zero elements
        q[:, :, :, 0] = 1.0
        k[:, :, :, 0] = 2.0
        self._check(q, k)

    def test_all_zero(self) -> None:
        self._check(torch.zeros(2, 4, 8, 32), torch.zeros(2, 4, 12, 32))

    def test_outlier_values(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 12, 32)
        # Add some outlier values (not Inf)
        q[0, 0, 0, 0] = 50.0
        k[0, 0, 0, 0] = 60.0
        self._check(q, k)


class TestScaledBoundCoversReference:
    """Test 7: Scaled (1/sqrt(d)) bound covers reference."""

    def test_scaled(self) -> None:
        from certimask.bounds import compute_score_bounds, validate_score_bounds

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
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0


class TestUnscaledBoundCoversReference:
    """Test 8: Unscaled bound covers reference."""

    def test_unscaled(self) -> None:
        from certimask.bounds import compute_score_bounds, validate_score_bounds

        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 12, 32)

        ref = reference_scores(q, k, scale_by_sqrt_dim=False)
        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=False)

        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=False,
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0


class TestActualBoundNotLargerThanAnalytic:
    """Test 9: Actual error <= analytic bound."""

    def test_comparison(self) -> None:
        from certimask.bounds import compute_score_bounds

        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 12, 32)

        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=True)

        actual_bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="actual",
            scale_by_sqrt_dim=True,
        )
        analytic_bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )

        # actual error_bound <= analytic error_bound + tolerance
        assert torch.all(
            actual_bounds.error_bound <= analytic_bounds.error_bound + 1e-5
        )


class TestZeroVectors:
    """Test 10: Zero vectors produce finite results with no violations."""

    def _check(self, q: torch.Tensor, k: torch.Tensor) -> None:
        from certimask.bounds import compute_score_bounds, validate_score_bounds

        ref = reference_scores(q, k, scale_by_sqrt_dim=True)
        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=True)

        assert torch.isfinite(result.scores).all()
        assert torch.isfinite(ref).all()

        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )
        assert torch.isfinite(bounds.lower).all()
        assert torch.isfinite(bounds.upper).all()
        assert torch.isfinite(bounds.error_bound).all()

        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0

    def test_q_zero_k_random(self) -> None:
        self._check(torch.zeros(2, 4, 8, 32), torch.randn(2, 4, 12, 32))

    def test_q_random_k_zero(self) -> None:
        self._check(torch.randn(2, 4, 8, 32), torch.zeros(2, 4, 12, 32))

    def test_both_zero(self) -> None:
        self._check(torch.zeros(2, 4, 8, 32), torch.zeros(2, 4, 12, 32))


class TestInvalidShapes:
    """Test 11: Invalid inputs raise exceptions."""

    def test_not_4d(self) -> None:
        q = torch.randn(4, 8, 16)
        k = torch.randn(4, 12, 16)
        with pytest.raises(ValueError, match="4-D"):
            reference_scores(q, k)

    def test_batch_mismatch(self) -> None:
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(3, 4, 12, 16)
        with pytest.raises(ValueError, match="batch"):
            reference_scores(q, k)

    def test_heads_mismatch(self) -> None:
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 3, 12, 16)
        with pytest.raises(ValueError, match="heads"):
            reference_scores(q, k)

    def test_head_dim_mismatch(self) -> None:
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 12, 32)
        with pytest.raises(ValueError, match="head_dim"):
            reference_scores(q, k)

    def test_empty_tensor(self) -> None:
        q = torch.empty(0, 4, 8, 16)
        k = torch.empty(0, 4, 12, 16)
        with pytest.raises(ValueError, match="empty"):
            reference_scores(q, k)

    def test_non_floating(self) -> None:
        q = torch.randint(0, 10, (2, 4, 8, 16))
        k = torch.randn(2, 4, 12, 16)
        with pytest.raises(TypeError, match="floating-point"):
            reference_scores(q, k)

    def test_nan(self) -> None:
        q = torch.randn(2, 4, 8, 16)
        q[0, 0, 0, 0] = float("nan")
        k = torch.randn(2, 4, 12, 16)
        with pytest.raises(ValueError, match="NaN"):
            reference_scores(q, k)

    def test_inf(self) -> None:
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 12, 16)
        k[0, 0, 0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            reference_scores(q, k)


class TestBoundsDetectInvalidInterval:
    """Test 12: validate_score_bounds detects intentionally invalid bounds."""

    def test_narrowed_bounds(self) -> None:
        from certimask.bounds import ScoreBounds, validate_score_bounds

        ref = torch.randn(2, 4, 8, 12)

        # Create bounds that are too narrow (artificially shrink them)
        bounds = ScoreBounds(
            lower=ref + 1.0,  # lower above reference
            upper=ref - 1.0,  # upper below reference
            error_bound=torch.zeros_like(ref),
        )
        violations = validate_score_bounds(ref, bounds)
        # Should detect all violations
        assert violations.all()

    def test_shifted_bounds(self) -> None:
        from certimask.bounds import ScoreBounds, validate_score_bounds

        ref = torch.randn(2, 4, 8, 12)

        # Shift bounds far away from reference
        bounds = ScoreBounds(
            lower=ref + 100.0,
            upper=ref + 200.0,
            error_bound=torch.full_like(ref, 100.0),
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.all()


class TestRandomStressTest:
    """Stress test: multiple random seeds, dimensions, certificate types."""

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("head_dim", [16, 64, 128])
    @pytest.mark.parametrize("certificate_type", ["actual", "analytic"])
    @pytest.mark.parametrize("scale_by_sqrt_dim", [True, False])
    def test_no_violations(
        self,
        seed: int,
        head_dim: int,
        certificate_type: str,
        scale_by_sqrt_dim: bool,
    ) -> None:
        from certimask.bounds import compute_score_bounds, validate_score_bounds

        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(1, 2, 8, head_dim, generator=gen)
        k = torch.randn(1, 2, 8, head_dim, generator=gen)

        ref = reference_scores(q, k, scale_by_sqrt_dim=scale_by_sqrt_dim)
        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=scale_by_sqrt_dim)

        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type=certificate_type,  # type: ignore[arg-type]
            scale_by_sqrt_dim=scale_by_sqrt_dim,
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0, (
            f"seed={seed}, head_dim={head_dim}, cert={certificate_type}, "
            f"scaled={scale_by_sqrt_dim}, violations={violations.sum().item()}"
        )
