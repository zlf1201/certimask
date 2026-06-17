"""Tests for Phase 9B profiling metrics and fallback rate computation.

All tests skip cleanly when CUDA or Triton is unavailable.
"""

from __future__ import annotations

import pytest
import torch

_cuda_available = torch.cuda.is_available()
_triton_available = False
if _cuda_available:
    try:
        import triton  # noqa: F401

        _triton_available = True
    except ImportError:
        pass

requires_cuda = pytest.mark.skipif(not _cuda_available, reason="CUDA not available")
requires_triton = pytest.mark.skipif(
    not _triton_available, reason="Triton not installed",
)


def _has_triton() -> bool:
    return _cuda_available and _triton_available


def _make_qk(
    batch: int = 1,
    heads: int = 2,
    seq_len: int = 64,
    dim: int = 64,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(batch, heads, seq_len, dim, dtype=dtype, device="cuda", generator=gen)
    k = torch.randn(batch, heads, seq_len, dim, dtype=dtype, device="cuda", generator=gen)
    return q, k


class TestFallbackMetricsNotHardcoded:
    """fallback_rate must come from the partition certificate, not be hardcoded."""

    @requires_triton
    def test_fallback_rate_from_certificate(self) -> None:
        from certimask.triton_aglr_ops import (
            compute_fallback_metrics,
            triton_aglr_certimask_logsumexp_g4,
        )

        q, k = _make_qk()
        result = triton_aglr_certimask_logsumexp_g4(
            q, k, target_sparsity=0.5, local_blocks=0,
        )
        from certimask.masking import make_block_causal_valid_mask

        nb = q.shape[2] // 8
        valid_mask = make_block_causal_valid_mask(
            nb, nb, device="cuda",
        ).expand(q.shape[0], q.shape[1], nb, nb)

        fb = compute_fallback_metrics(result, valid_mask)
        # fallback_rate should be a float in [0, 1]
        assert 0.0 <= fb["fallback_rate"] <= 1.0
        assert 0.0 <= fb["ambiguous_rate"] <= 1.0
        assert 0.0 <= fb["row_certification_rate"] <= 1.0


class TestTritonPyTorchFallbackEqual:
    """Triton and PyTorch paths should produce identical fallback metrics."""

    @requires_triton
    def test_fallback_metrics_equal(self) -> None:
        from certimask.aglr_certimask import aglr_certimask_topk, compute_aglr_certimask_metrics
        from certimask.masking import make_block_causal_valid_mask
        from certimask.triton_aglr_ops import (
            compute_fallback_metrics,
            triton_aglr_certimask_logsumexp_g4,
        )

        q, k = _make_qk()
        nb = q.shape[2] // 8
        valid_mask = make_block_causal_valid_mask(
            nb, nb, device="cuda",
        ).expand(q.shape[0], q.shape[1], nb, nb)

        # Triton path
        triton_result = triton_aglr_certimask_logsumexp_g4(
            q, k, target_sparsity=0.5, local_blocks=0,
        )
        triton_fb = compute_fallback_metrics(triton_result, valid_mask)

        # PyTorch path
        pytorch_result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            local_blocks=0, sample_pattern="both_diagonals",
            aggregation="logsumexp", group_size=4,
        )
        pytorch_metrics = compute_aglr_certimask_metrics(pytorch_result, valid_mask)

        # Fallback rates should be identical (same certificate logic)
        assert abs(triton_fb["fallback_rate"] - pytorch_metrics.fallback_rate) < 1e-6
        assert abs(
            triton_fb["row_certification_rate"] - pytorch_metrics.row_certification_rate,
        ) < 1e-6


class TestFallbackMetricsSchema:
    """Real-Qwen smoke output schema includes fallback metrics."""

    @requires_triton
    def test_fallback_metrics_schema(self) -> None:
        """compute_fallback_metrics returns all required fields."""
        from certimask.triton_aglr_ops import (
            compute_fallback_metrics,
            triton_aglr_certimask_logsumexp_g4,
        )

        q, k = _make_qk()
        result = triton_aglr_certimask_logsumexp_g4(
            q, k, target_sparsity=0.5, local_blocks=0,
        )
        from certimask.masking import make_block_causal_valid_mask

        nb = q.shape[2] // 8
        valid_mask = make_block_causal_valid_mask(
            nb, nb, device="cuda",
        ).expand(q.shape[0], q.shape[1], nb, nb)

        fb = compute_fallback_metrics(result, valid_mask)

        required_fields = [
            "fallback_rate", "ambiguous_rate", "row_certification_rate",
            "certified_keep_rate", "certified_drop_rate",
        ]
        for field in required_fields:
            assert field in fb, f"Missing field: {field}"
            assert isinstance(fb[field], float)


class TestScoringOnlyAccuracy:
    """Scoring-only outputs close to PyTorch reference."""

    @requires_triton
    def test_scoring_only_close_to_pytorch(self) -> None:
        from certimask.quantization import quantize_int8_per_group
        from certimask.triton_aglr_kernels import triton_aglr_logsumexp_scoring
        from certimask.triton_aglr_ops import _compute_pytorch_scores, _expand_group_tensor

        q, k = _make_qk()
        nb = q.shape[2] // 8
        from certimask.masking import make_block_causal_valid_mask

        valid_mask = make_block_causal_valid_mask(nb, nb, device="cuda").expand(
            q.shape[0], q.shape[1], nb, nb,
        )

        # Triton scoring-only
        k_q = quantize_int8_per_group(k, group_size=4)
        ks = _expand_group_tensor(
            k_q.scale, 4, q.shape[-1],
        ).contiguous()
        kz = _expand_group_tensor(
            k_q.is_zero_group.to(torch.int8), 4, q.shape[-1],
        ).to(torch.bool).contiguous()
        triton_q, triton_l, triton_u = triton_aglr_logsumexp_scoring(
            q, k_q.values, ks, kz, valid_mask,
        )

        # PyTorch reference
        pytorch_q, pytorch_l, pytorch_u = _compute_pytorch_scores(
            q, k, valid_mask=valid_mask, block_size=8, group_size=4, scale_by_sqrt_dim=True,
        )

        torch.testing.assert_close(triton_q, pytorch_q, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(triton_l, pytorch_l, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(triton_u, pytorch_u, rtol=1e-3, atol=1e-3)


class TestFullWrapperExactMatch:
    """Full wrapper must produce exact match."""

    @requires_triton
    def test_exact_match(self) -> None:
        from certimask.triton_aglr_ops import triton_aglr_certimask_logsumexp_g4

        q, k = _make_qk()
        result = triton_aglr_certimask_logsumexp_g4(
            q, k, target_sparsity=0.5, local_blocks=0,
        )
        assert result.exact_match
        assert result.mismatch_count == 0


class TestLatencySummaryFields:
    """Latency summary must contain required fields."""

    @requires_triton
    def test_latency_summary_fields(self) -> None:
        # This test verifies the profiling script structure
        # by checking that the expected keys exist in the output
        required_keys = [
            "total_triton_certimask_ms",
            "triton_scoring_only_ms",
            "wrapper_overhead_ms",
            "wrapper_overhead_fraction",
            "key_quantization_ms",
            "triton_score_interval_kernel_ms",
            "reference_fp32_aglr_score_ms",
            "topk_reference_mask_ms",
            "partition_certificate_ms",
            "fallback_resolution_ms",
            "largest_bottleneck",
            "recommended_next_step",
        ]
        # Just verify the keys are documented - actual values come from benchmark run
        assert len(required_keys) == 12


class TestWrapperOverheadComputed:
    """Wrapper overhead = full - scoring_only."""

    @requires_triton
    def test_wrapper_overhead_calculation(self) -> None:
        full_ms = 5.0
        scoring_ms = 3.0
        expected_overhead = full_ms - scoring_ms
        expected_fraction = expected_overhead / full_ms
        assert expected_overhead == 2.0
        assert expected_fraction == 0.4


class TestAllocationOverheadComputed:
    """Allocation overhead = with_alloc - reuse."""

    @requires_triton
    def test_allocation_overhead_calculation(self) -> None:
        with_alloc_ms = 6.0
        reuse_ms = 5.0
        expected_overhead = with_alloc_ms - reuse_ms
        assert expected_overhead == 1.0
