"""Tests for per-group INT8 quantization, scoring, and bounds."""

from __future__ import annotations

import pytest
import torch

from certimask.bounds import (
    compute_group_quantized_coordinate_bounds,
    validate_score_bounds,
)
from certimask.masking import (
    certified_threshold_mask,
    reference_mask,
)
from certimask.quantization import quantize_int8_per_group
from certimask.scoring import (
    group_quantized_int8_scores,
    quantized_int8_scores,
    reference_scores,
)


class TestGroupQuantizationBasic:
    """Test basic per-group quantization."""

    def test_shape(self) -> None:
        x = torch.randn(2, 4, 64)
        result = quantize_int8_per_group(x, group_size=16)
        assert result.values.shape == x.shape
        assert result.dequantized.shape == x.shape
        assert result.scale.shape[-1] == 4  # 64/16 = 4 groups
        assert result.group_size == 16

    def test_values_range(self) -> None:
        x = torch.randn(10, 64) * 100
        result = quantize_int8_per_group(x, group_size=8)
        assert result.values.min() >= -127
        assert result.values.max() <= 127
        assert result.values.dtype == torch.int8

    def test_dequantized_finite(self) -> None:
        x = torch.randn(4, 32)
        result = quantize_int8_per_group(x, group_size=8)
        assert torch.isfinite(result.dequantized).all()

    def test_group_size_equals_dim(self) -> None:
        """group_size == dim should behave like per-vector."""
        x = torch.randn(2, 32)
        result_g = quantize_int8_per_group(x, group_size=32)
        # Should have 1 group
        assert result_g.scale.shape[-1] == 1
        assert result_g.values.shape == x.shape

    def test_non_divisible_group_size(self) -> None:
        """Last group can be shorter than group_size."""
        x = torch.randn(2, 30)
        result = quantize_int8_per_group(x, group_size=8)
        # 30/8 = 3 groups of 8 + 1 group of 6
        assert result.scale.shape[-1] == 4
        assert result.group_lengths.tolist() == [8, 8, 8, 6]

    def test_zero_group(self) -> None:
        """All-zero group: scale=1, values=0, analytic bound=0."""
        x = torch.zeros(1, 16)
        result = quantize_int8_per_group(x, group_size=8)
        assert torch.all(result.values == 0)
        assert torch.all(result.dequantized == 0)
        assert result.is_zero_group.all()
        # Analytic bound must be 0 for zero groups
        assert torch.all(result.analytic_l2_bound == 0)
        assert torch.all(result.actual_l2_error == 0)

    def test_mixed_zero_nonzero(self) -> None:
        """Some groups zero, some not."""
        gen = torch.Generator().manual_seed(42)
        x = torch.zeros(1, 32)
        x[0, 8:16] = torch.randn(8, generator=gen)
        result = quantize_int8_per_group(x, group_size=8)
        # Group 0 and 2,3 are zero; group 1 is not
        assert result.is_zero_group[0, 0].item() is True
        assert result.is_zero_group[0, 1].item() is False
        assert result.is_zero_group[0, 2].item() is True
        assert result.is_zero_group[0, 3].item() is True
        # Zero groups have 0 analytic bound
        assert result.analytic_l2_bound[0, 0].item() == 0.0
        assert result.analytic_l2_bound[0, 1].item() > 0.0

    def test_invalid_group_size(self) -> None:
        x = torch.randn(4, 16)
        with pytest.raises(ValueError, match="positive"):
            quantize_int8_per_group(x, group_size=0)

    def test_invalid_input(self) -> None:
        with pytest.raises(TypeError, match="floating-point"):
            quantize_int8_per_group(torch.randint(0, 10, (4, 16)), group_size=4)

    def test_dtype_support(self) -> None:
        for dt in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            x = torch.randn(2, 32).to(dt)
            result = quantize_int8_per_group(x, group_size=8)
            assert result.values.dtype == torch.int8
            assert result.dequantized.dtype == torch.float32


class TestGroupQuantizedScore:
    """Test per-group quantized scoring."""

    def test_basic_shape(self) -> None:
        q = torch.randn(1, 2, 4, 32)
        k = torch.randn(1, 2, 6, 32)
        result = group_quantized_int8_scores(q, k, group_size=8)
        assert result.scores.shape == (1, 2, 4, 6)
        assert result.integer_dot.shape == (1, 2, 4, 6, 4)  # 32/8 = 4 groups

    def test_group_size_64_matches_per_vector(self) -> None:
        """group_size == head_dim should match per-vector quantized score."""
        q = torch.randn(1, 2, 4, 64)
        k = torch.randn(1, 2, 4, 64)

        g_result = group_quantized_int8_scores(q, k, group_size=64)
        v_result = quantized_int8_scores(q, k)

        # Should be close (same quantization)
        torch.testing.assert_close(
            g_result.scores, v_result.scores, rtol=1e-4, atol=1e-4,
        )

    def test_cross_validation(self) -> None:
        """integer_dot * scale matches dequantized dot product."""
        q = torch.randn(1, 2, 4, 32)
        k = torch.randn(1, 2, 4, 32)
        result = group_quantized_int8_scores(q, k, group_size=8)

        # Reconstruct from dequantized
        q_deq = result.query_quantized.dequantized
        k_deq = result.key_quantized.dequantized
        expected = torch.einsum("bhqd,bhkd->bhqk", q_deq, k_deq)
        d = q.shape[-1]
        expected = expected / torch.sqrt(torch.tensor(float(d)))

        torch.testing.assert_close(result.scores, expected, rtol=1e-4, atol=1e-4)

    def test_scaled_and_unscaled(self) -> None:
        q = torch.randn(1, 2, 4, 16)
        k = torch.randn(1, 2, 4, 16)
        r1 = group_quantized_int8_scores(q, k, group_size=4, scale_by_sqrt_dim=True)
        r2 = group_quantized_int8_scores(q, k, group_size=4, scale_by_sqrt_dim=False)
        # Unscaled should be larger
        assert r2.scores.abs().mean() > r1.scores.abs().mean()


class TestGroupCoordinateBounds:
    """Test per-group coordinate analytic bounds."""

    def test_analytic_zero_violations(self) -> None:
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        ref = reference_scores(q, k)
        g_result = group_quantized_int8_scores(q, k, group_size=8)
        bounds = compute_group_quantized_coordinate_bounds(
            g_result.scores, g_result.query_quantized,
            g_result.key_quantized, certificate_type="analytic",
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0

    def test_exact_match(self) -> None:
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 8, 32)
        ref = reference_scores(q, k)
        g_result = group_quantized_int8_scores(q, k, group_size=8)
        bounds = compute_group_quantized_coordinate_bounds(
            g_result.scores, g_result.query_quantized,
            g_result.key_quantized, certificate_type="analytic",
        )
        threshold = ref.mean()
        ref_mask = reference_mask(ref, threshold)
        cert = certified_threshold_mask(bounds, ref, threshold)
        assert torch.equal(cert.mask, ref_mask)

    def test_smaller_groups_tighter(self) -> None:
        q = torch.randn(1, 2, 4, 64)
        k = torch.randn(1, 2, 4, 64)
        g_result_32 = group_quantized_int8_scores(q, k, group_size=32)
        g_result_8 = group_quantized_int8_scores(q, k, group_size=8)

        bounds_32 = compute_group_quantized_coordinate_bounds(
            g_result_32.scores, g_result_32.query_quantized,
            g_result_32.key_quantized, certificate_type="analytic",
        )
        bounds_8 = compute_group_quantized_coordinate_bounds(
            g_result_8.scores, g_result_8.query_quantized,
            g_result_8.key_quantized, certificate_type="analytic",
        )
        # Smaller groups should give tighter or equal bounds
        assert bounds_8.error_bound.mean() <= bounds_32.error_bound.mean() + 1e-6

    def test_invalid_certificate_type(self) -> None:
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 4, 16)
        g_result = group_quantized_int8_scores(q, k, group_size=8)
        with pytest.raises(ValueError, match="actual"):
            compute_group_quantized_coordinate_bounds(
                g_result.scores, g_result.query_quantized,
                g_result.key_quantized, certificate_type="actual",
            )


class TestGroupQuantizedPipeline:
    """Test full pipeline: group quant -> scores -> bounds -> mask."""

    def test_full_pipeline(self) -> None:
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 8, 32)

        ref = reference_scores(q, k, scale_by_sqrt_dim=True)
        g_result = group_quantized_int8_scores(
            q, k, group_size=8, scale_by_sqrt_dim=True,
        )
        bounds = compute_group_quantized_coordinate_bounds(
            g_result.scores, g_result.query_quantized,
            g_result.key_quantized, certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )

        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0

        threshold = ref.mean()
        ref_mask = reference_mask(ref, threshold)
        cert = certified_threshold_mask(bounds, ref, threshold)

        assert torch.equal(cert.mask, ref_mask)

    @pytest.mark.parametrize("group_size", [4, 8, 16, 32])
    def test_all_group_sizes(self, group_size: int) -> None:
        q = torch.randn(1, 2, 4, 32)
        k = torch.randn(1, 2, 4, 32)
        ref = reference_scores(q, k)
        g_result = group_quantized_int8_scores(q, k, group_size=group_size)
        bounds = compute_group_quantized_coordinate_bounds(
            g_result.scores, g_result.query_quantized,
            g_result.key_quantized, certificate_type="analytic",
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0, f"gs={group_size} violations={violations.sum()}"
