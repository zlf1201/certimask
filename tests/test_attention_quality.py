"""Tests for attention quality metrics and sparse/dense comparison."""

from __future__ import annotations

import torch

from certimask.attention_quality import (
    block_sparse_attention_output,
    compute_attention_quality,
    compute_benefit_proxy,
    dense_attention_output,
)


class TestDenseAttention:
    """Test dense attention output."""

    def test_output_shape(self) -> None:
        q = torch.randn(1, 2, 8, 16)
        k = torch.randn(1, 2, 8, 16)
        v = torch.randn(1, 2, 8, 16)
        out, probs = dense_attention_output(q, k, v)
        assert out.shape == (1, 2, 8, 16)
        assert probs.shape == (1, 2, 8, 8)

    def test_probs_sum_to_one(self) -> None:
        q = torch.randn(1, 2, 8, 16)
        k = torch.randn(1, 2, 8, 16)
        v = torch.randn(1, 2, 8, 16)
        _, probs = dense_attention_output(q, k, v)
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-5)

    def test_causal_mask(self) -> None:
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        _, probs = dense_attention_output(q, k, v, causal=True)
        # Future positions should have zero probability
        for i in range(4):
            for j in range(i + 1, 4):
                assert probs[0, 0, i, j].abs() < 1e-6

    def test_output_finite(self) -> None:
        q = torch.randn(1, 2, 8, 16)
        k = torch.randn(1, 2, 8, 16)
        v = torch.randn(1, 2, 8, 16)
        out, _ = dense_attention_output(q, k, v)
        assert torch.isfinite(out).all()


class TestBlockSparseAttention:
    """Test block sparse attention output."""

    def test_output_shape(self) -> None:
        q = torch.randn(1, 2, 8, 16)
        k = torch.randn(1, 2, 8, 16)
        v = torch.randn(1, 2, 8, 16)
        # Full block mask (all kept)
        block_mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
        out, probs = block_sparse_attention_output(q, k, v, block_mask, block_size=4)
        assert out.shape == (1, 2, 8, 16)
        assert probs.shape == (1, 2, 8, 8)

    def test_full_mask_equals_dense(self) -> None:
        q = torch.randn(1, 2, 8, 16)
        k = torch.randn(1, 2, 8, 16)
        v = torch.randn(1, 2, 8, 16)
        block_mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)

        dense_out, dense_probs = dense_attention_output(q, k, v)
        sparse_out, sparse_probs = block_sparse_attention_output(
            q, k, v, block_mask, block_size=4
        )

        torch.testing.assert_close(sparse_out, dense_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(sparse_probs, dense_probs, rtol=1e-5, atol=1e-5)

    def test_causal_mask_preserved(self) -> None:
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        block_mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)  # single block
        _, probs = block_sparse_attention_output(q, k, v, block_mask, block_size=4)
        for i in range(4):
            for j in range(i + 1, 4):
                assert probs[0, 0, i, j].abs() < 1e-6

    def test_dropped_block_masks_tokens(self) -> None:
        q = torch.randn(1, 1, 8, 8)
        k = torch.randn(1, 1, 8, 8)
        v = torch.randn(1, 1, 8, 8)
        # Keep only block (0,0) and (1,1), drop (0,1) and (1,0)
        block_mask = torch.tensor([[[[True, False], [False, True]]]])
        _, probs = block_sparse_attention_output(q, k, v, block_mask, block_size=4)

        # Token 0 (block 0) should not attend to tokens 4-7 (block 1)
        for i in range(4):
            for j in range(4, 8):
                assert probs[0, 0, i, j].abs() < 1e-6


class TestAttentionQuality:
    """Test attention quality metrics computation."""

    def test_metrics_no_nan(self) -> None:
        batch, heads, seq_len, dim = 1, 2, 8, 16
        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)

        block_mask = torch.ones(batch, heads, 2, 2, dtype=torch.bool)
        dense_out, dense_probs = dense_attention_output(q, k, v)
        sparse_out, sparse_probs = block_sparse_attention_output(
            q, k, v, block_mask, block_size=4
        )

        metrics = compute_attention_quality(
            dense_out, dense_probs, sparse_out, sparse_probs,
            block_mask, 4, layer_index=0, target_sparsity=0.0,
        )

        # All values should be finite
        for field_name, val in vars(metrics).items():
            if isinstance(val, float):
                assert val == val, f"{field_name} is NaN"  # noqa: PLR0124
                assert abs(val) < float("inf"), f"{field_name} is Inf"

    def test_full_kept_mass(self) -> None:
        """With full block mask and causal, kept mass should be 1.0."""
        batch, heads, seq_len, dim = 1, 2, 8, 16
        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)

        block_mask = torch.ones(batch, heads, 2, 2, dtype=torch.bool)
        dense_out, dense_probs = dense_attention_output(q, k, v, causal=True)
        sparse_out, sparse_probs = block_sparse_attention_output(
            q, k, v, block_mask, block_size=4, causal=True,
        )

        metrics = compute_attention_quality(
            dense_out, dense_probs, sparse_out, sparse_probs,
            block_mask, 4, layer_index=0, target_sparsity=0.0,
        )

        # Dense baseline: kept mass must be exactly 1.0
        assert abs(metrics.kept_attention_mass_mean - 1.0) < 1e-5
        assert abs(metrics.output_cosine_mean - 1.0) < 1e-5
        assert metrics.output_l2_relative_mean < 1e-5

    def test_dropped_block_reduces_mass(self) -> None:
        """Dropping blocks should reduce kept mass."""
        batch, heads, seq_len, dim = 1, 1, 8, 16
        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)

        full_mask = torch.ones(batch, heads, 2, 2, dtype=torch.bool)
        # Drop the self-attention block for tokens 4-7
        partial_mask = torch.tensor([[[[True, True], [True, False]]]])

        dense_out, dense_probs = dense_attention_output(q, k, v, causal=True)
        sparse_full, probs_full = block_sparse_attention_output(
            q, k, v, full_mask, block_size=4, causal=True,
        )
        sparse_partial, probs_partial = block_sparse_attention_output(
            q, k, v, partial_mask, block_size=4, causal=True,
        )

        m_full = compute_attention_quality(
            dense_out, dense_probs, sparse_full, probs_full,
            full_mask, 4, layer_index=0, target_sparsity=0.0,
        )
        m_partial = compute_attention_quality(
            dense_out, dense_probs, sparse_partial, probs_partial,
            partial_mask, 4, layer_index=0, target_sparsity=0.25,
        )

        # Full mask should have kept mass = 1.0
        assert abs(m_full.kept_attention_mass_mean - 1.0) < 1e-5
        # Partial mask should have less
        assert m_partial.kept_attention_mass_mean < 1.0


class TestBenefitProxy:
    """Test benefit proxy metrics."""

    def test_fallback_score_work_is_one(self) -> None:
        mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        bp = compute_benefit_proxy(0, "FP16 fallback", "none", mask, 0.5)
        assert bp.score_work_fraction_proxy == 1.0

    def test_go_score_work_is_refinement(self) -> None:
        mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        bp = compute_benefit_proxy(5, "Go", "k_only_g16", mask, 0.08)
        assert bp.score_work_fraction_proxy == 0.08

    def test_tile_sparsity(self) -> None:
        mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        mask[0, 0, 0, 1] = False
        mask[0, 0, 1, 0] = False
        bp = compute_benefit_proxy(0, "Go", "test", mask, 0.1)
        assert abs(bp.tile_sparsity - 2 / 16) < 1e-6
        assert abs(bp.kept_tile_fraction - 14 / 16) < 1e-6
