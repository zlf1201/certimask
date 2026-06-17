"""Tests for fused Triton top-k partition certificate.

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


class TestTritonCertificateDecisionsMatchPyTorch:
    """Triton certificate decisions must equal PyTorch decisions."""

    @requires_triton
    def test_decisions_match(self) -> None:
        from certimask.masking import make_block_causal_valid_mask
        from certimask.triton_aglr_ops import triton_aglr_certimask_logsumexp_g4
        from certimask.triton_topk_certificate import triton_certified_topk_mask_partition

        q, k = _make_qk()
        nb = q.shape[2] // 8
        valid_mask = make_block_causal_valid_mask(
            nb, nb, device="cuda",
        ).expand(q.shape[0], q.shape[1], nb, nb)

        # Get full result with PyTorch certificate
        full_result = triton_aglr_certimask_logsumexp_g4(
            q, k, target_sparsity=0.5, local_blocks=0,
        )

        # Run Triton certificate
        triton_dec, triton_amb = triton_certified_topk_mask_partition(
            full_result.lower_scores,
            full_result.upper_scores,
            full_result.reference_mask,
            valid_mask,
        )

        # Compare
        torch.testing.assert_close(triton_dec, full_result.decisions)
        torch.testing.assert_close(triton_amb, full_result.ambiguous)


class TestTritonCertificateAmbiguousMatch:
    """Triton ambiguous mask must equal PyTorch ambiguous mask."""

    @requires_triton
    def test_ambiguous_match(self) -> None:
        from certimask.masking import make_block_causal_valid_mask
        from certimask.triton_aglr_ops import triton_aglr_certimask_logsumexp_g4
        from certimask.triton_topk_certificate import triton_certified_topk_mask_partition

        q, k = _make_qk()
        nb = q.shape[2] // 8
        valid_mask = make_block_causal_valid_mask(
            nb, nb, device="cuda",
        ).expand(q.shape[0], q.shape[1], nb, nb)

        full_result = triton_aglr_certimask_logsumexp_g4(
            q, k, target_sparsity=0.5, local_blocks=0,
        )

        triton_dec, triton_amb = triton_certified_topk_mask_partition(
            full_result.lower_scores,
            full_result.upper_scores,
            full_result.reference_mask,
            valid_mask,
        )

        torch.testing.assert_close(triton_amb, full_result.ambiguous)


class TestFallbackRateIdentical:
    """Triton and PyTorch fallback rates must be identical."""

    @requires_triton
    def test_fallback_rate_identical(self) -> None:
        from certimask.masking import make_block_causal_valid_mask
        from certimask.topk_certificate import AMBIGUOUS
        from certimask.triton_aglr_ops import (
            compute_fallback_metrics,
            triton_aglr_certimask_logsumexp_g4,
        )
        from certimask.triton_topk_certificate import triton_certified_topk_mask_partition

        q, k = _make_qk()
        nb = q.shape[2] // 8
        valid_mask = make_block_causal_valid_mask(
            nb, nb, device="cuda",
        ).expand(q.shape[0], q.shape[1], nb, nb)

        full_result = triton_aglr_certimask_logsumexp_g4(
            q, k, target_sparsity=0.5, local_blocks=0,
        )

        # PyTorch fallback
        pytorch_fb = compute_fallback_metrics(full_result, valid_mask)

        # Triton fallback (recompute decisions)
        triton_dec, triton_amb = triton_certified_topk_mask_partition(
            full_result.lower_scores,
            full_result.upper_scores,
            full_result.reference_mask,
            valid_mask,
        )

        # Compare fallback rates
        valid_decisions = triton_dec[valid_mask]
        total_valid = valid_decisions.numel()
        triton_fallback = int((valid_decisions == AMBIGUOUS).sum().item()) / total_valid

        assert abs(pytorch_fb["fallback_rate"] - triton_fallback) < 1e-6


class TestInvalidTilesRemainInvalid:
    """Invalid tiles must have INVALID decision code."""

    @requires_triton
    def test_invalid_tiles(self) -> None:
        from certimask.masking import make_block_causal_valid_mask
        from certimask.topk_certificate import INVALID
        from certimask.triton_topk_certificate import triton_certified_topk_mask_partition

        q, k = _make_qk(batch=1, heads=1, seq_len=64, dim=64)
        nb = 64 // 8
        valid_mask = make_block_causal_valid_mask(nb, nb, device="cuda").expand(1, 1, nb, nb)

        # Create simple scores
        lower = torch.randn(1, 1, nb, nb, device="cuda")
        upper = lower + torch.rand(1, 1, nb, nb, device="cuda").abs()
        selected = torch.zeros(1, 1, nb, nb, dtype=torch.bool, device="cuda")
        # Select some tiles
        for i in range(nb):
            selected[0, 0, i, :min(i + 1, nb // 2)] = True

        triton_dec, triton_amb = triton_certified_topk_mask_partition(
            lower, upper, selected, valid_mask,
        )

        # Check invalid tiles (future blocks)
        for qi in range(nb):
            for ki in range(nb):
                if ki > qi:
                    assert triton_dec[0, 0, qi, ki].item() == INVALID
                    assert not triton_amb[0, 0, qi, ki].item()


class TestLowerLeqUpper:
    """Lower scores must be <= upper scores everywhere."""

    @requires_triton
    def test_lower_leq_upper(self) -> None:
        from certimask.masking import make_block_causal_valid_mask
        from certimask.quantization import quantize_int8_per_group
        from certimask.triton_aglr_kernels import triton_aglr_logsumexp_scoring
        from certimask.triton_aglr_ops import _expand_group_tensor

        q, k = _make_qk()
        nb = q.shape[2] // 8
        valid_mask = make_block_causal_valid_mask(nb, nb, device="cuda").expand(
            q.shape[0], q.shape[1], nb, nb,
        )

        k_q = quantize_int8_per_group(k, group_size=4)
        ks = _expand_group_tensor(
            k_q.scale, 4, q.shape[-1],
        ).contiguous()
        kz = _expand_group_tensor(
            k_q.is_zero_group.to(torch.int8), 4, q.shape[-1],
        ).to(torch.bool).contiguous()

        _, lo, hi = triton_aglr_logsumexp_scoring(q, k_q.values, ks, kz, valid_mask)
        valid = valid_mask
        assert (lo[valid] <= hi[valid] + 1e-5).all()


class TestCUDARequired:
    """CUDA/Triton tests skip cleanly when unavailable."""

    @requires_cuda
    def test_cuda_available(self) -> None:
        assert torch.cuda.is_available()

    @requires_triton
    def test_triton_available(self) -> None:
        import triton

        assert hasattr(triton, "__version__")
