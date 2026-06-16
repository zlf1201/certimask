"""Tests for K-only quantization and group quantization strategies."""

from __future__ import annotations

import pytest
import torch

from certimask.bounds import (
    compute_coordinate_score_bounds,
    compute_group_quantized_coordinate_bounds,
    compute_k_only_per_group_bounds,
    compute_k_only_per_vector_bounds,
    validate_score_bounds,
)
from certimask.masking import (
    certified_threshold_mask,
    make_block_causal_valid_mask,
    naive_quantized_mask,
    reference_mask,
    thresholds_for_target_sparsity,
)
from certimask.metrics import compute_mask_metrics
from certimask.quantization import quantize_int8_per_group
from certimask.scoring import (
    group_quantized_int8_scores,
    k_only_per_group_scores,
    k_only_per_vector_scores,
    quantized_int8_scores,
    reference_scores,
)


class TestGroupQuantization:
    """Test per-group quantization basics."""

    def test_shape_correct(self) -> None:
        x = torch.randn(2, 4, 8, 64)
        result = quantize_int8_per_group(x, group_size=16)
        assert result.values.shape == x.shape
        assert result.dequantized.shape == x.shape
        assert result.scale.shape == (2, 4, 8, 4)  # 64/16 = 4 groups
        assert result.group_size == 16

    def test_scale_shape(self) -> None:
        x = torch.randn(1, 2, 4, 32)
        result = quantize_int8_per_group(x, group_size=8)
        assert result.scale.shape == (1, 2, 4, 4)  # 32/8 = 4 groups

    def test_zero_group_analytic_bound_zero(self) -> None:
        x = torch.zeros(1, 2, 4, 32)
        result = quantize_int8_per_group(x, group_size=8)
        assert torch.all(result.analytic_l2_bound == 0)
        assert torch.all(result.is_zero_group)

    def test_non_divisible_group_size(self) -> None:
        x = torch.randn(1, 2, 4, 30)
        result = quantize_int8_per_group(x, group_size=8)
        # 30/8 = 3 groups of 8 + 1 group of 6
        assert result.scale.shape[-1] == 4
        assert result.group_lengths.tolist() == [8, 8, 8, 6]

    def test_group_dequantized_consistent(self) -> None:
        x = torch.randn(2, 4, 8, 64)
        result = quantize_int8_per_group(x, group_size=16)
        # Dequantized should be close to original
        error = (x.float() - result.dequantized).abs()
        assert error.max() < 1.0  # reasonable bound

    def test_invalid_group_size(self) -> None:
        x = torch.randn(1, 2, 4, 32)
        with pytest.raises(ValueError, match="group_size"):
            quantize_int8_per_group(x, group_size=0)


class TestGroupQuantizedScores:
    """Test group quantized scoring."""

    def test_shape_correct(self) -> None:
        q = torch.randn(1, 2, 8, 64)
        k = torch.randn(1, 2, 12, 64)
        result = group_quantized_int8_scores(q, k, group_size=16)
        assert result.scores.shape == (1, 2, 8, 12)

    def test_matches_dequantized(self) -> None:
        q = torch.randn(1, 2, 4, 32)
        k = torch.randn(1, 2, 6, 32)
        result = group_quantized_int8_scores(q, k, group_size=8)
        # Manually compute from dequantized
        q_deq = result.query_quantized.dequantized
        k_deq = result.key_quantized.dequantized
        expected = torch.einsum("bhqd,bhkd->bhqk", q_deq, k_deq) / torch.sqrt(torch.tensor(32.0))
        torch.testing.assert_close(result.scores, expected, rtol=1e-5, atol=1e-5)


class TestKOnlyPerVector:
    """Test K-only per-vector scoring."""

    def test_shape_correct(self) -> None:
        q = torch.randn(1, 2, 8, 64)
        k = torch.randn(1, 2, 12, 64)
        result = k_only_per_vector_scores(q, k)
        assert result.scores.shape == (1, 2, 8, 12)
        assert result.query.shape == q.shape

    def test_certificate_violations_zero(self) -> None:
        q = torch.randn(1, 2, 4, 32)
        k = torch.randn(1, 2, 6, 32)
        ref = reference_scores(q, k)
        result = k_only_per_vector_scores(q, k)
        bounds = compute_k_only_per_vector_bounds(
            result.scores, result.query, result.key_quantized,
            certificate_type="analytic",
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0


class TestKOnlyPerGroup:
    """Test K-only per-group scoring."""

    def test_shape_correct(self) -> None:
        q = torch.randn(1, 2, 8, 64)
        k = torch.randn(1, 2, 12, 64)
        result = k_only_per_group_scores(q, k, group_size=16)
        assert result.scores.shape == (1, 2, 8, 12)

    def test_certificate_violations_zero(self) -> None:
        q = torch.randn(1, 2, 4, 32)
        k = torch.randn(1, 2, 6, 32)
        ref = reference_scores(q, k)
        result = k_only_per_group_scores(q, k, group_size=8)
        bounds = compute_k_only_per_group_bounds(
            result.scores, result.query, result.key_quantized,
            certificate_type="analytic",
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0


class TestAllStrategiesExactMatch:
    """Test that all strategies produce 100% exact match with CertiMask."""

    @pytest.mark.parametrize("strategy", [
        "baseline_per_vector_qk",
        "qk_per_group_g16",
        "k_only_per_vector",
        "k_only_per_group_g8",
    ])
    def test_exact_match(self, strategy: str) -> None:
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 12, 32)

        ref = reference_scores(q, k, scale_by_sqrt_dim=True)
        valid_mask = make_block_causal_valid_mask(8, 12).expand_as(ref)
        thresholds = thresholds_for_target_sparsity(
            ref, 0.85, valid_mask=valid_mask, per_query=True,
        )
        ref_mask = reference_mask(ref, thresholds, valid_mask=valid_mask)

        if strategy == "baseline_per_vector_qk":
            result = quantized_int8_scores(q, k, scale_by_sqrt_dim=True)
            bounds = compute_coordinate_score_bounds(
                result.scores, result.query_quantized, result.key_quantized,
                certificate_type="analytic", scale_by_sqrt_dim=True,
            )
            scores = result.scores
        elif strategy == "qk_per_group_g16":
            g_result = group_quantized_int8_scores(q, k, group_size=16, scale_by_sqrt_dim=True)
            bounds = compute_group_quantized_coordinate_bounds(
                g_result.scores, g_result.query_quantized, g_result.key_quantized,
                certificate_type="analytic", scale_by_sqrt_dim=True,
            )
            scores = g_result.scores
        elif strategy == "k_only_per_vector":
            ko_result = k_only_per_vector_scores(q, k, scale_by_sqrt_dim=True)
            bounds = compute_k_only_per_vector_bounds(
                ko_result.scores, ko_result.query, ko_result.key_quantized,
                certificate_type="analytic", scale_by_sqrt_dim=True,
            )
            scores = ko_result.scores
        elif strategy == "k_only_per_group_g8":
            ko_result = k_only_per_group_scores(q, k, group_size=8, scale_by_sqrt_dim=True)
            bounds = compute_k_only_per_group_bounds(
                ko_result.scores, ko_result.query, ko_result.key_quantized,
                certificate_type="analytic", scale_by_sqrt_dim=True,
            )
            scores = ko_result.scores
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        cert = certified_threshold_mask(bounds, ref, thresholds, valid_mask=valid_mask)
        naive = naive_quantized_mask(scores, thresholds, valid_mask=valid_mask)
        mm = compute_mask_metrics(ref_mask, naive, cert, valid_mask=valid_mask)
        assert mm.certimask_match_rate == 1.0, (
            f"{strategy}: match rate = {mm.certimask_match_rate}"
        )
        assert mm.certimask_mismatch_count == 0
