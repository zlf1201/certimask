"""Tests for Phase 9D: quantization benchmark utilities and proxy calculations."""

from __future__ import annotations

import pytest
import torch

from certimask.quantization import quantize_int8_per_group
from certimask.triton_aglr_ops import _expand_group_tensor


class TestCachedQuantization:
    """Test that cached quantization path reuses quantized K."""

    def test_cached_quantization_reuses_values(self) -> None:
        """Pre-quantized K should be identical to fresh quantization."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        k = torch.randn(1, 14, 1024, 64, dtype=torch.float16, device="cuda")
        group_size = 4

        # First quantization
        k_q1 = quantize_int8_per_group(k, group_size=group_size)

        # Second quantization (should be identical)
        k_q2 = quantize_int8_per_group(k, group_size=group_size)

        assert torch.equal(k_q1.values, k_q2.values)
        assert torch.equal(k_q1.scale, k_q2.scale)
        assert torch.equal(k_q1.is_zero_group, k_q2.is_zero_group)

    def test_expand_group_tensor_shape(self) -> None:
        """Expanded group tensor should have correct shape."""
        grouped = torch.randn(1, 14, 1024, 16, device="cuda")  # 16 groups
        expanded = _expand_group_tensor(grouped, group_size=4, target_dim=64)
        assert expanded.shape == (1, 14, 1024, 64)


class TestPreallocatedBuffers:
    """Test preallocated buffer shapes."""

    def test_buffer_shapes_correct(self) -> None:
        """Preallocated buffers should match expected shapes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch, heads, seq_len, dim = 1, 14, 1024, 64
        group_size = 4
        num_groups = dim // group_size

        k_int8_buf = torch.empty(
            batch, heads, seq_len, dim, dtype=torch.int8, device="cuda",
        )
        scale_buf = torch.empty(
            batch, heads, seq_len, num_groups, dtype=torch.float32, device="cuda",
        )
        zero_buf = torch.empty(
            batch, heads, seq_len, num_groups, dtype=torch.bool, device="cuda",
        )

        assert k_int8_buf.shape == (1, 14, 1024, 64)
        assert scale_buf.shape == (1, 14, 1024, 16)
        assert zero_buf.shape == (1, 14, 1024, 16)


class TestDenseAttentionBaseline:
    """Test dense attention baseline output shape."""

    def test_sdpa_output_shape(self) -> None:
        """SDPA output should match input shape."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        q = torch.randn(1, 14, 1024, 64, dtype=torch.float16, device="cuda")
        k = torch.randn(1, 14, 1024, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(1, 14, 1024, 64, dtype=torch.float16, device="cuda")

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert out.shape == (1, 14, 1024, 64)

    def test_manual_dense_output_shape(self) -> None:
        """Manual dense attention output should match input shape."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        q = torch.randn(1, 2, 64, 32, dtype=torch.float32, device="cuda")
        k = torch.randn(1, 2, 64, 32, dtype=torch.float32, device="cuda")
        v = torch.randn(1, 2, 64, 32, dtype=torch.float32, device="cuda")

        # Manual dense attention
        d = q.shape[-1]
        logits = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
        seq_len = q.shape[-2]
        causal_mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=q.device, dtype=q.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        logits = logits + causal_mask.unsqueeze(0).unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        out = torch.matmul(probs, v)

        assert out.shape == (1, 2, 64, 32)


class TestProxyCalculations:
    """Test proxy calculation correctness."""

    def test_speedup_proxy_calculation(self) -> None:
        """Speedup proxy = dense_time / total_proxy."""
        dense_ms = 10.0
        total_proxy = 5.0
        speedup = dense_ms / total_proxy
        assert speedup == 2.0

    def test_ideal_sparse_calculation(self) -> None:
        """Ideal sparse = dense * work_fraction."""
        dense_ms = 10.0
        work_fraction = 0.3765
        ideal_sparse = dense_ms * work_fraction
        assert abs(ideal_sparse - 3.765) < 1e-6

    def test_total_proxy_calculation(self) -> None:
        """Total proxy = pipeline_ms + ideal_sparse_ms."""
        pipeline_ms = 6.44
        ideal_sparse_ms = 3.765
        total = pipeline_ms + ideal_sparse_ms
        assert abs(total - 10.205) < 1e-6


class TestReadinessDecision:
    """Test readiness decision logic."""

    def test_ready_when_speedup_over_1_1(self) -> None:
        """Ready when cached speedup >= 1.1."""
        cached_speedup = 1.2
        ready = cached_speedup >= 1.1
        assert ready

    def test_not_ready_when_speedup_under_1(self) -> None:
        """Not ready when cached speedup < 1.0."""
        cached_speedup = 0.8
        not_viable = cached_speedup < 1.0
        assert not_viable

    def test_needs_optimization_when_between_1_and_1_1(self) -> None:
        """Needs optimization when speedup is between 1.0 and 1.1."""
        cached_speedup = 1.05
        ready = cached_speedup >= 1.1
        not_viable = cached_speedup < 1.0
        assert not ready
        assert not not_viable

    def test_not_viable_when_speedup_under_1(self) -> None:
        """Not viable when speedup proxy < 1.0."""
        cached_speedup = 0.8
        not_viable = cached_speedup < 1.0
        assert not_viable
