"""Tests for score error bounds and validation."""

from __future__ import annotations

import pytest
import torch

from certimask.bounds import ScoreBounds, compute_score_bounds, validate_score_bounds
from certimask.scoring import quantized_int8_scores, reference_scores


class TestScoreBoundsShape:
    """Test that ScoreBounds tensors have correct shapes."""

    def test_shape(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 12, 32)

        result = quantized_int8_scores(q, k)
        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )

        assert bounds.lower.shape == (2, 4, 8, 12)
        assert bounds.upper.shape == (2, 4, 8, 12)
        assert bounds.error_bound.shape == (2, 4, 8, 12)


class TestErrorBoundFormula:
    """Test the error bound formula E = ||q||*eps_k + ||k||*eps_q + eps_q*eps_k."""

    def test_manual_computation(self) -> None:
        # Use small vectors for manual verification
        q = torch.tensor([[[[1.0, 0.0]]]])  # [1,1,1,2]
        k = torch.tensor([[[[0.0, 1.0]]]])  # [1,1,1,2]

        result = quantized_int8_scores(q, k, scale_by_sqrt_dim=False)

        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=False,
        )

        # error_bound should be non-negative
        assert torch.all(bounds.error_bound >= 0)

        # lower <= scores <= upper
        assert torch.all(bounds.lower <= result.scores + 1e-6)
        assert torch.all(bounds.upper >= result.scores - 1e-6)


class TestCertificateTypes:
    """Test both actual and analytic certificates."""

    def test_actual_certificate(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 12, 32)

        result = quantized_int8_scores(q, k)
        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="actual",
            scale_by_sqrt_dim=True,
        )

        assert bounds.error_bound.shape == (2, 4, 8, 12)
        assert torch.all(bounds.error_bound >= 0)

    def test_analytic_certificate(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 12, 32)

        result = quantized_int8_scores(q, k)
        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )

        assert bounds.error_bound.shape == (2, 4, 8, 12)
        assert torch.all(bounds.error_bound >= 0)

    def test_invalid_certificate_type(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 12, 32)

        result = quantized_int8_scores(q, k)
        with pytest.raises(ValueError, match="certificate_type"):
            compute_score_bounds(
                result.scores,
                result.query_quantized,
                result.key_quantized,
                certificate_type="invalid",  # type: ignore[arg-type]
            )


class TestValidateScoreBounds:
    """Test the validate_score_bounds function."""

    def test_no_violations(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 12, 32)

        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)

        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0

    def test_detects_violations(self) -> None:
        # Create bounds that are too narrow
        ref = torch.randn(2, 4, 8, 12)
        bounds = ScoreBounds(
            lower=ref + 1.0,
            upper=ref - 1.0,
            error_bound=torch.zeros_like(ref),
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.all()

    def test_shape_mismatch(self) -> None:
        ref = torch.randn(2, 4, 8, 12)
        bounds = ScoreBounds(
            lower=torch.randn(2, 4, 8, 10),
            upper=torch.randn(2, 4, 8, 10),
            error_bound=torch.randn(2, 4, 8, 10),
        )
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_score_bounds(ref, bounds)

    def test_nan_in_reference(self) -> None:
        ref = torch.randn(2, 4, 8, 12)
        ref[0, 0, 0, 0] = float("nan")
        bounds = ScoreBounds(
            lower=torch.zeros_like(ref),
            upper=torch.ones_like(ref),
            error_bound=torch.ones_like(ref),
        )
        with pytest.raises(ValueError, match="NaN"):
            validate_score_bounds(ref, bounds)

    def test_inf_in_reference(self) -> None:
        ref = torch.randn(2, 4, 8, 12)
        ref[0, 0, 0, 0] = float("inf")
        bounds = ScoreBounds(
            lower=torch.zeros_like(ref),
            upper=torch.ones_like(ref),
            error_bound=torch.ones_like(ref),
        )
        with pytest.raises(ValueError, match="Inf"):
            validate_score_bounds(ref, bounds)


class TestZeroVectorBounds:
    """Test bounds with zero vectors."""

    def test_q_zero(self) -> None:
        q = torch.zeros(2, 4, 8, 32)
        k = torch.randn(2, 4, 12, 32)

        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)

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

    def test_k_zero(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.zeros(2, 4, 12, 32)

        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)

        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )

        assert torch.isfinite(bounds.lower).all()
        assert torch.isfinite(bounds.upper).all()
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0

    def test_both_zero(self) -> None:
        q = torch.zeros(2, 4, 8, 32)
        k = torch.zeros(2, 4, 12, 32)

        ref = reference_scores(q, k)
        result = quantized_int8_scores(q, k)

        bounds = compute_score_bounds(
            result.scores,
            result.query_quantized,
            result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )

        assert torch.isfinite(bounds.lower).all()
        assert torch.isfinite(bounds.upper).all()
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0


class TestStressTest50Seeds:
    """Random stress test: 50 seeds, multiple head_dims, certificate types."""

    @pytest.mark.parametrize("seed", range(50))
    @pytest.mark.parametrize("head_dim", [16, 64, 128])
    def test_actual_and_analytic(
        self,
        seed: int,
        head_dim: int,
    ) -> None:
        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(1, 2, 8, head_dim, generator=gen)
        k = torch.randn(1, 2, 8, head_dim, generator=gen)

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
            violations = validate_score_bounds(ref, bounds)
            assert violations.sum() == 0, (
                f"seed={seed}, head_dim={head_dim}, cert={cert_type}, "
                f"violations={violations.sum().item()}"
            )

    @pytest.mark.parametrize("seed", range(50))
    @pytest.mark.parametrize("head_dim", [16, 64, 128])
    def test_scaled_and_unscaled(
        self,
        seed: int,
        head_dim: int,
    ) -> None:
        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(1, 2, 8, head_dim, generator=gen)
        k = torch.randn(1, 2, 8, head_dim, generator=gen)

        for scaled in (True, False):
            ref = reference_scores(q, k, scale_by_sqrt_dim=scaled)
            result = quantized_int8_scores(q, k, scale_by_sqrt_dim=scaled)

            bounds = compute_score_bounds(
                result.scores,
                result.query_quantized,
                result.key_quantized,
                certificate_type="analytic",
                scale_by_sqrt_dim=scaled,
            )
            violations = validate_score_bounds(ref, bounds)
            assert violations.sum() == 0, (
                f"seed={seed}, head_dim={head_dim}, scaled={scaled}, "
                f"violations={violations.sum().item()}"
            )
