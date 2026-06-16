"""Tests for symmetric per-vector INT8 quantization."""

from __future__ import annotations

import pytest
import torch

from certimask.quantization import quantize_int8_per_vector


class TestBasicShape:
    """Test 1: Basic shape correctness."""

    def test_3d_tensor(self) -> None:
        x = torch.randn(4, 8, 128)
        result = quantize_int8_per_vector(x)

        assert result.values.shape == x.shape
        assert result.dequantized.shape == x.shape
        assert result.scale.shape == (4, 8, 1)
        assert result.actual_l2_error.shape == (4, 8)
        assert result.analytic_l2_bound.shape == (4, 8)

    def test_2d_tensor(self) -> None:
        x = torch.randn(10, 64)
        result = quantize_int8_per_vector(x)

        assert result.values.shape == x.shape
        assert result.dequantized.shape == x.shape
        assert result.scale.shape == (10, 1)
        assert result.actual_l2_error.shape == (10,)
        assert result.analytic_l2_bound.shape == (10,)

    def test_1d_tensor(self) -> None:
        x = torch.randn(32)
        result = quantize_int8_per_vector(x)

        assert result.values.shape == x.shape
        assert result.dequantized.shape == x.shape


class TestINT8Range:
    """Test 2: Quantized values are in INT8 range."""

    def test_values_in_range(self) -> None:
        x = torch.randn(4, 8, 128)
        result = quantize_int8_per_vector(x)

        assert result.values.min() >= -127
        assert result.values.max() <= 127

    def test_dtype_is_int8(self) -> None:
        x = torch.randn(4, 8, 128)
        result = quantize_int8_per_vector(x)

        assert result.values.dtype == torch.int8


class TestZeroVector:
    """Test 3: All-zero vectors."""

    def test_zero_vector(self) -> None:
        x = torch.zeros(4, 8)
        result = quantize_int8_per_vector(x)

        assert not torch.isnan(result.scale).any()
        assert not torch.isnan(result.dequantized).any()
        assert not torch.isnan(result.actual_l2_error).any()
        assert not torch.isnan(result.analytic_l2_bound).any()

        assert torch.all(result.scale == 1.0)
        assert torch.all(result.values == 0)
        assert torch.all(result.dequantized == 0)
        assert torch.all(result.actual_l2_error == 0)


class TestAnalyticBound:
    """Test 4: Analytic L2 bound covers actual L2 error."""

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_bound_covers_error(self, dtype: torch.dtype) -> None:
        x = torch.randn(4, 8, 128).to(dtype)
        result = quantize_int8_per_vector(x)

        assert torch.all(result.actual_l2_error <= result.analytic_l2_bound + 1e-5)


class TestDifferentDim:
    """Test 5: Different quantization dimensions."""

    def test_dim_minus1(self) -> None:
        x = torch.randn(4, 8, 128)
        result = quantize_int8_per_vector(x, dim=-1)

        assert result.values.shape == x.shape
        assert result.scale.shape == (4, 8, 1)

    def test_dim_1(self) -> None:
        x = torch.randn(4, 8, 128)
        result = quantize_int8_per_vector(x, dim=1)

        assert result.values.shape == x.shape
        assert result.scale.shape == (4, 1, 128)

    def test_dim_0(self) -> None:
        x = torch.randn(4, 8, 128)
        result = quantize_int8_per_vector(x, dim=0)

        assert result.values.shape == x.shape
        assert result.scale.shape == (1, 8, 128)


class TestExtremeValues:
    """Test 6: Very small and very large values."""

    def test_small_values(self) -> None:
        x = torch.tensor([1e-8, 1e-4, 1e-8, 1e-4])
        result = quantize_int8_per_vector(x)

        assert torch.isfinite(result.dequantized).all()
        assert torch.isfinite(result.actual_l2_error).all()
        assert torch.isfinite(result.analytic_l2_bound).all()

    def test_large_values(self) -> None:
        x = torch.tensor([1e2, 1e4, 1e2, 1e4])
        result = quantize_int8_per_vector(x)

        assert torch.isfinite(result.dequantized).all()
        assert torch.isfinite(result.actual_l2_error).all()
        assert torch.isfinite(result.analytic_l2_bound).all()

    def test_mixed_values(self) -> None:
        x = torch.tensor([1e-8, 1e-4, 1e2, 1e4])
        result = quantize_int8_per_vector(x)

        assert torch.isfinite(result.dequantized).all()
        assert torch.isfinite(result.actual_l2_error).all()


class TestInvalidInput:
    """Test 7: Invalid inputs raise exceptions."""

    def test_integer_tensor(self) -> None:
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        with pytest.raises(TypeError, match="floating-point"):
            quantize_int8_per_vector(x)

    def test_nan(self) -> None:
        x = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError, match="NaN"):
            quantize_int8_per_vector(x)

    def test_inf(self) -> None:
        x = torch.tensor([1.0, float("inf"), 3.0])
        with pytest.raises(ValueError, match="Inf"):
            quantize_int8_per_vector(x)

    def test_empty_tensor(self) -> None:
        x = torch.empty(0, 128)
        with pytest.raises(ValueError, match="empty"):
            quantize_int8_per_vector(x)

    def test_invalid_dim(self) -> None:
        x = torch.randn(4, 8)
        with pytest.raises(ValueError, match="out of range"):
            quantize_int8_per_vector(x, dim=5)


class TestQuantizationReasonableness:
    """Test 8: Quantization results are reasonable."""

    def test_max_abs_error_bound(self) -> None:
        x = torch.randn(4, 8, 128)
        result = quantize_int8_per_vector(x)

        # For each vector, max absolute error should be <= scale/2 + tolerance
        # Reshape scale to broadcast with x
        scale_expanded = result.scale.expand_as(x)
        max_abs_error = (x - result.dequantized).abs().max(dim=-1).values
        max_scale = scale_expanded.max(dim=-1).values

        assert torch.all(max_abs_error <= max_scale / 2 + 1e-5)

    def test_dequantized_close_to_original(self) -> None:
        x = torch.randn(4, 8, 128)
        result = quantize_int8_per_vector(x)

        # Per-element absolute error should be bounded by scale / 2
        # (the rounding error for each element is at most half a quantization step)
        scale_expanded = result.scale.expand_as(x)
        per_elem_error = (x - result.dequantized).abs()
        assert torch.all(per_elem_error <= scale_expanded / 2 + 1e-6)
