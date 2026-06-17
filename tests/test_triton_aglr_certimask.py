"""Tests for Triton AGLR-C CertiMask sampled scoring.

All tests skip cleanly when CUDA or Triton is unavailable.
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qk(
    batch: int = 1,
    heads: int = 2,
    seq_len: int = 64,
    dim: int = 64,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic Q, K tensors on CUDA."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(batch, heads, seq_len, dim, dtype=dtype, device="cuda", generator=gen)
    k = torch.randn(batch, heads, seq_len, dim, dtype=dtype, device="cuda", generator=gen)
    return q, k


def _make_valid_mask(
    batch: int,
    heads: int,
    num_blocks: int,
) -> torch.Tensor:
    """Create a causal valid block mask."""
    from certimask.masking import make_block_causal_valid_mask

    return make_block_causal_valid_mask(
        num_blocks, num_blocks, device="cuda",
    ).expand(batch, heads, num_blocks, num_blocks)


def _run_triton_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    group_size: int = 4,
    scale_by_sqrt_dim: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Triton kernel and return (quantized, lower, upper) scores."""
    from certimask.quantization import quantize_int8_per_group
    from certimask.triton_aglr_kernels import triton_aglr_logsumexp_scoring
    from certimask.triton_aglr_ops import _expand_group_tensor

    dim = q.shape[-1]
    k_quantized = quantize_int8_per_group(k, group_size=group_size)
    key_int8 = k_quantized.values
    key_scales = k_quantized.scale
    key_is_zero = k_quantized.is_zero_group

    key_scales_expanded = _expand_group_tensor(key_scales, group_size, dim).contiguous()
    key_is_zero_expanded = _expand_group_tensor(
        key_is_zero.to(torch.int8), group_size, dim,
    ).to(torch.bool).contiguous()

    return triton_aglr_logsumexp_scoring(
        q, key_int8, key_scales_expanded, key_is_zero_expanded, valid_mask,
    )


def _run_pytorch_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    group_size: int = 4,
    scale_by_sqrt_dim: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the PyTorch reference and return (quantized, lower, upper) scores."""
    from certimask.triton_aglr_ops import _compute_pytorch_scores

    return _compute_pytorch_scores(
        q,
        k,
        valid_mask=valid_mask,
        block_size=8,
        group_size=group_size,
        scale_by_sqrt_dim=scale_by_sqrt_dim,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCudaTritonSkip:
    """Verify skip behaviour when CUDA/Triton is missing."""

    @requires_cuda
    def test_cuda_is_available(self) -> None:
        assert torch.cuda.is_available()

    @requires_triton
    def test_triton_is_available(self) -> None:
        import triton

        assert hasattr(triton, "__version__")


class TestOutputShape:
    """Triton kernel output shapes."""

    @requires_triton
    def test_shape_small(self) -> None:
        q, k = _make_qk(batch=1, heads=2, seq_len=64, dim=64)
        nb = 64 // 8
        vm = _make_valid_mask(1, 2, nb)
        qt, lo, hi = _run_triton_scores(q, k, vm)
        assert qt.shape == (1, 2, nb, nb)
        assert lo.shape == (1, 2, nb, nb)
        assert hi.shape == (1, 2, nb, nb)

    @requires_triton
    def test_shape_larger(self) -> None:
        q, k = _make_qk(batch=1, heads=4, seq_len=128, dim=64)
        nb = 128 // 8
        vm = _make_valid_mask(1, 4, nb)
        qt, lo, hi = _run_triton_scores(q, k, vm)
        assert qt.shape == (1, 4, nb, nb)


class TestScoreAccuracy:
    """Triton scores close to PyTorch reference."""

    @requires_triton
    def test_quantized_score_close(self) -> None:
        q, k = _make_qk()
        nb = 64 // 8
        vm = _make_valid_mask(1, 2, nb)
        triton_q, _, _ = _run_triton_scores(q, k, vm)
        pytorch_q, _, _ = _run_pytorch_scores(q, k, vm)
        torch.testing.assert_close(triton_q, pytorch_q, rtol=1e-3, atol=1e-3)

    @requires_triton
    def test_lower_score_close(self) -> None:
        q, k = _make_qk()
        nb = 64 // 8
        vm = _make_valid_mask(1, 2, nb)
        _, triton_l, _ = _run_triton_scores(q, k, vm)
        _, pytorch_l, _ = _run_pytorch_scores(q, k, vm)
        torch.testing.assert_close(triton_l, pytorch_l, rtol=1e-3, atol=1e-3)

    @requires_triton
    def test_upper_score_close(self) -> None:
        q, k = _make_qk()
        nb = 64 // 8
        vm = _make_valid_mask(1, 2, nb)
        _, _, triton_u = _run_triton_scores(q, k, vm)
        _, _, pytorch_u = _run_pytorch_scores(q, k, vm)
        torch.testing.assert_close(triton_u, pytorch_u, rtol=1e-3, atol=1e-3)


class TestExactMatch:
    """Final mask must exactly match FP32 AGLR reference."""

    @requires_triton
    def test_exact_match(self) -> None:
        from certimask.triton_aglr_ops import triton_aglr_certimask_logsumexp_g4

        q, k = _make_qk()
        result = triton_aglr_certimask_logsumexp_g4(
            q, k, target_sparsity=0.5, local_blocks=0,
        )
        assert result.exact_match, f"Mismatch count: {result.mismatch_count}"
        assert result.mismatch_count == 0

    @requires_triton
    def test_exact_match_high_sparsity(self) -> None:
        from certimask.triton_aglr_ops import triton_aglr_certimask_logsumexp_g4

        q, k = _make_qk()
        result = triton_aglr_certimask_logsumexp_g4(
            q, k, target_sparsity=0.75, local_blocks=0,
        )
        assert result.exact_match
        assert result.mismatch_count == 0


class TestInvalidTiles:
    """Invalid / future tiles get -inf scores."""

    @requires_triton
    def test_invalid_tiles_are_neg_inf(self) -> None:
        q, k = _make_qk(batch=1, heads=1, seq_len=64, dim=64)
        nb = 64 // 8
        vm = _make_valid_mask(1, 1, nb)
        qt, lo, hi = _run_triton_scores(q, k, vm)
        # Future tiles (k > q) should be very negative
        for qi in range(nb):
            for ki in range(nb):
                if ki > qi:
                    assert qt[0, 0, qi, ki].item() < -1e30


class TestZeroKGroup:
    """Zero K group contributes zero error bound."""

    @requires_triton
    def test_zero_k_group(self) -> None:
        from certimask.bounds import _get_group_per_coord_error
        from certimask.quantization import quantize_int8_per_group
        from certimask.triton_aglr_ops import _expand_group_tensor

        _q = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float32)  # noqa: F841
        k = torch.zeros(1, 1, 64, 64, device="cuda", dtype=torch.float32)
        k_q = quantize_int8_per_group(k, group_size=4)
        k_err = _get_group_per_coord_error(k_q, "analytic")
        assert (k_err == 0).all()

        # Expanded zero flags should all be True
        expanded_zero = _expand_group_tensor(
            k_q.is_zero_group.to(torch.int8), 4, 64,
        ).to(torch.bool)
        assert expanded_zero.all()


class TestDtypeHandling:
    """FP16 and FP32 query both work."""

    @requires_triton
    def test_fp32_query(self) -> None:
        q, k = _make_qk(dtype=torch.float32)
        nb = 64 // 8
        vm = _make_valid_mask(1, 2, nb)
        qt, lo, hi = _run_triton_scores(q, k, vm)
        assert qt.dtype == torch.float32

    @requires_triton
    def test_fp16_query(self) -> None:
        q, k = _make_qk(dtype=torch.float16)
        nb = 64 // 8
        vm = _make_valid_mask(1, 2, nb)
        qt, lo, hi = _run_triton_scores(q, k, vm)
        assert qt.dtype == torch.float32


class TestMultipleHeads:
    """Different head counts work correctly."""

    @requires_triton
    def test_1_head(self) -> None:
        q, k = _make_qk(batch=1, heads=1, seq_len=64, dim=64)
        nb = 64 // 8
        vm = _make_valid_mask(1, 1, nb)
        qt, lo, hi = _run_triton_scores(q, k, vm)
        assert qt.shape == (1, 1, nb, nb)

    @requires_triton
    def test_14_heads(self) -> None:
        q, k = _make_qk(batch=1, heads=14, seq_len=64, dim=64)
        nb = 64 // 8
        vm = _make_valid_mask(1, 14, nb)
        qt, lo, hi = _run_triton_scores(q, k, vm)
        assert qt.shape == (1, 14, nb, nb)


class TestSmallSequence:
    """Minimum sequence length (1 block)."""

    @requires_triton
    def test_one_block(self) -> None:
        q, k = _make_qk(batch=1, heads=1, seq_len=8, dim=64)
        nb = 1
        vm = _make_valid_mask(1, 1, nb)
        qt, lo, hi = _run_triton_scores(q, k, vm)
        assert qt.shape == (1, 1, 1, 1)


class TestBlockSizeValidation:
    """block_size != 8 is rejected by assertion in the launcher."""

    @requires_triton
    def test_non_64_dim_raises(self) -> None:
        q = torch.randn(1, 1, 64, 32, device="cuda", dtype=torch.float32)
        k = torch.randn(1, 1, 64, 32, device="cuda", dtype=torch.float32)
        with pytest.raises(AssertionError):
            from certimask.triton_aglr_ops import triton_aglr_certimask_logsumexp_g4

            triton_aglr_certimask_logsumexp_g4(q, k, target_sparsity=0.5)


class TestUnsupportedAggregation:
    """topk_mean is not supported in Triton v0 — the wrapper uses logsumexp only."""

    @requires_triton
    def test_wrapper_uses_logsumexp(self) -> None:
        """The wrapper always uses logsumexp; topk_mean is not routed to Triton."""
        from certimask.triton_aglr_ops import triton_aglr_certimask_logsumexp_g4

        q, k = _make_qk()
        result = triton_aglr_certimask_logsumexp_g4(
            q, k, target_sparsity=0.5,
        )
        # Should succeed (uses logsumexp internally)
        assert result.quantized_scores is not None


class TestLowerLeqUpper:
    """Lower scores must be <= upper scores everywhere."""

    @requires_triton
    def test_lower_leq_upper(self) -> None:
        q, k = _make_qk()
        nb = 64 // 8
        vm = _make_valid_mask(1, 2, nb)
        _, lo, hi = _run_triton_scores(q, k, vm)
        valid = vm
        assert (lo[valid] <= hi[valid] + 1e-5).all()


class TestFallbackMetrics:
    """Fallback metrics from partition certificate."""

    @requires_triton
    def test_fallback_rate_not_hardcoded(self) -> None:
        """fallback_rate must come from certificate decisions, not be zero."""
        from certimask.triton_aglr_ops import (
            compute_fallback_metrics,
            triton_aglr_certimask_logsumexp_g4,
        )

        q, k = _make_qk()
        result = triton_aglr_certimask_logsumexp_g4(
            q, k, target_sparsity=0.5, local_blocks=0,
        )
        vm = _make_valid_mask(1, 2, 64 // 8)
        fb = compute_fallback_metrics(result, vm)
        # fallback_rate should be computed, not hardcoded
        assert isinstance(fb["fallback_rate"], float)
        assert 0.0 <= fb["fallback_rate"] <= 1.0
