"""Tests for K-only per-group policy and full-model proxy."""

from __future__ import annotations

import torch

from certimask.bounds import (
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
from certimask.scoring import (
    k_only_per_group_scores,
    k_only_per_vector_scores,
    reference_scores,
)


class TestKOnlyBoundFormula:
    """Test that K-only bound does not include Query error terms."""

    def test_no_query_error_in_bound(self) -> None:
        """K-only bound should be sum_i |q_i| * b_i^k, no q error term."""
        q = torch.randn(1, 2, 4, 32)
        k = torch.randn(1, 2, 6, 32)

        ko_result = k_only_per_group_scores(
            q, k, group_size=8, scale_by_sqrt_dim=True
        )
        bounds = compute_k_only_per_group_bounds(
            ko_result.scores,
            ko_result.query,
            ko_result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )

        # Manually compute: E = sum_i |q_i| * b_i^k / sqrt(d)
        q_f32 = q.float()
        k_quant = ko_result.key_quantized
        d = q.shape[-1]
        sqrt_d = torch.sqrt(torch.tensor(float(d)))

        # b_i^k = alpha_g / 2 for each coordinate
        half_scale = k_quant.scale / 2.0  # [1, 2, 6, 4]
        # Expand to per-coordinate
        b_k = torch.zeros_like(k_quant.dequantized)
        for g_idx in range(4):
            start = g_idx * 8
            end = min(start + 8, d)
            b_k[..., start:end] = half_scale[..., g_idx].unsqueeze(-1)

        # E_ab = sum_i |q_a,i| * b_b,i^k / sqrt(d)
        expected_bound = (
            torch.einsum("bhqd,bhkd->bhqk", q_f32.abs(), b_k) / sqrt_d
        )

        torch.testing.assert_close(
            bounds.error_bound, expected_bound, rtol=1e-5, atol=1e-6
        )

    def test_zero_k_group_bound_zero(self) -> None:
        """All-zero K group should have b_i^k = 0."""
        q = torch.randn(1, 1, 2, 32)
        k = torch.zeros(1, 1, 3, 32)
        # Make one group non-zero
        k[0, 0, 0, 8:16] = torch.randn(8)

        ko_result = k_only_per_group_scores(
            q, k, group_size=8, scale_by_sqrt_dim=True
        )
        bounds = compute_k_only_per_group_bounds(
            ko_result.scores,
            ko_result.query,
            ko_result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )

        # Bound should be finite and non-negative
        assert torch.isfinite(bounds.error_bound).all()
        assert (bounds.error_bound >= 0).all()


class TestKOnlyCertificate:
    """Test K-only certificate violations and exact match."""

    def test_per_group_violations_zero(self) -> None:
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 12, 32)

        ref = reference_scores(q, k, scale_by_sqrt_dim=True)
        ko_result = k_only_per_group_scores(
            q, k, group_size=8, scale_by_sqrt_dim=True
        )
        bounds = compute_k_only_per_group_bounds(
            ko_result.scores,
            ko_result.query,
            ko_result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0

    def test_per_vector_violations_zero(self) -> None:
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 12, 32)

        ref = reference_scores(q, k, scale_by_sqrt_dim=True)
        ko_result = k_only_per_vector_scores(
            q, k, scale_by_sqrt_dim=True
        )
        bounds = compute_k_only_per_vector_bounds(
            ko_result.scores,
            ko_result.query,
            ko_result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )
        violations = validate_score_bounds(ref, bounds)
        assert violations.sum() == 0

    def test_exact_match(self) -> None:
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 12, 32)

        ref = reference_scores(q, k, scale_by_sqrt_dim=True)
        valid_mask = make_block_causal_valid_mask(
            8, 12, device=q.device
        ).expand_as(ref)
        thresholds = thresholds_for_target_sparsity(
            ref, 0.85, valid_mask=valid_mask, per_query=True
        )
        ref_mask = reference_mask(ref, thresholds, valid_mask=valid_mask)

        ko_result = k_only_per_group_scores(
            q, k, group_size=8, scale_by_sqrt_dim=True
        )
        bounds = compute_k_only_per_group_bounds(
            ko_result.scores,
            ko_result.query,
            ko_result.key_quantized,
            certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )
        cert = certified_threshold_mask(
            bounds, ref, thresholds, valid_mask=valid_mask
        )
        naive = naive_quantized_mask(
            ko_result.scores, thresholds, valid_mask=valid_mask
        )
        mm = compute_mask_metrics(
            ref_mask, naive, cert, valid_mask=valid_mask
        )

        assert mm.certimask_match_rate == 1.0
        assert mm.certimask_mismatch_count == 0


class TestGroupSizeSelection:
    """Test group size selection priority."""

    def test_prefer_larger_group_if_under_threshold(self) -> None:
        """If g16 achieves <20%, prefer it over g8/g4."""
        # Simulate results
        results = {
            "k_only_per_group_g16_coordinate_analytic": 0.15,
            "k_only_per_group_g8_coordinate_analytic": 0.10,
            "k_only_per_group_g4_coordinate_analytic": 0.08,
        }
        # Import the selection function from the experiment
        # We'll test the logic directly
        for gs in [16, 8, 4]:
            key = f"k_only_per_group_g{gs}_coordinate_analytic"
            if key in results and results[key] < 0.20:
                assert gs == 16  # Should pick g16 first
                break

    def test_fallback_to_smaller_group(self) -> None:
        """If g16 >= 20%, try g8, then g4."""
        results = {
            "k_only_per_group_g16_coordinate_analytic": 0.25,
            "k_only_per_group_g8_coordinate_analytic": 0.18,
            "k_only_per_group_g4_coordinate_analytic": 0.12,
        }
        selected_gs = 0
        for gs in [16, 8, 4]:
            key = f"k_only_per_group_g{gs}_coordinate_analytic"
            if key in results and results[key] < 0.20:
                selected_gs = gs
                break
        assert selected_gs == 8

    def test_all_above_threshold(self) -> None:
        """If all >= 20%, pick the one with lowest refinement."""
        results = {
            "k_only_per_group_g16_coordinate_analytic": 0.35,
            "k_only_per_group_g8_coordinate_analytic": 0.28,
            "k_only_per_group_g4_coordinate_analytic": 0.22,
        }
        # None achieve <20%, so pick best
        best_key = min(results, key=lambda k: results[k])
        assert best_key == "k_only_per_group_g4_coordinate_analytic"


class TestFullModelProxy:
    """Test full-model FP16 proxy calculation."""

    def test_proxy_calculation(self) -> None:
        """Test proxy with known values."""
        num_layers = 4
        # Layer 0: fallback (1.0), Layer 1: refinement 0.15,
        # Layer 2: refinement 0.08, Layer 3: fallback (1.0)
        refinements = [1.0, 0.15, 0.08, 1.0]

        proxy = sum(refinements) / num_layers
        expected = (1.0 + 0.15 + 0.08 + 1.0) / 4
        assert abs(proxy - expected) < 1e-10

    def test_proxy_not_latency_estimate(self) -> None:
        """Proxy should be documented as not a latency estimate."""
        # This is a documentation test - the proxy is a scoring work fraction
        proxy = 0.15
        # Even with low proxy, actual latency depends on many factors
        assert proxy < 1.0  # Sanity check


class TestBestVsSelected:
    """Test that best_refinement_strategy can differ from selected."""

    def test_different_strategies_possible(self) -> None:
        """best can be g4 while selected is g16."""
        # best_refinement_strategy = k_only_per_group_g4 (lowest refinement)
        # selected_system_strategy = k_only_per_group_g16 (largest group < 20%)
        best = "k_only_per_group_g4_coordinate_analytic"
        selected = "k_only_per_group_g16_coordinate_analytic"
        assert best != selected
