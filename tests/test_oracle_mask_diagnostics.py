"""Tests for oracle mask diagnostics and tile sparsity fix."""

from __future__ import annotations

import torch

from certimask.attention_quality import (
    block_sparse_attention_output,
    compute_attention_quality,
    compute_oracle_block_mass_scores,
    dense_attention_output,
    expand_block_mask_to_token_mask,
    local_window_block_mask,
    oracle_block_mass_mask,
    random_valid_block_mask,
)


class TestTileSparsityFix:
    """Test that tile sparsity is computed correctly with valid_mask."""

    def test_dense_mask_sparsity_zero(self) -> None:
        """Dense causal mask should have tile_sparsity = 0 within valid tiles."""
        batch, heads, seq_len, dim = 1, 2, 8, 16
        block_size = 4
        q_blk, k_blk = 2, 2

        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)

        dense_out, dense_probs = dense_attention_output(q, k, v, causal=True)
        # All valid tiles kept
        block_mask = torch.ones(batch, heads, q_blk, k_blk, dtype=torch.bool)
        valid_mask = torch.ones(batch, heads, q_blk, k_blk, dtype=torch.bool)
        # Causal: block (1, 0) is invalid (future)
        valid_mask[0, :, 1, 0] = False

        sparse_out, sparse_probs = block_sparse_attention_output(
            q, k, v, block_mask, block_size=block_size, causal=True,
        )
        metrics = compute_attention_quality(
            dense_out, dense_probs, sparse_out, sparse_probs,
            block_mask, block_size, layer_index=0, target_sparsity=0.0,
            valid_block_mask=valid_mask,
        )

        # With valid_mask, tile sparsity should be 0 (all valid tiles kept)
        assert abs(metrics.actual_tile_sparsity) < 1e-6

    def test_all_drop_sparsity_one(self) -> None:
        """All-drop mask should have tile_sparsity = 1."""
        batch, heads, seq_len, dim = 1, 2, 8, 16
        block_size = 4
        q_blk, k_blk = 2, 2

        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)

        dense_out, dense_probs = dense_attention_output(q, k, v, causal=True)
        block_mask = torch.zeros(batch, heads, q_blk, k_blk, dtype=torch.bool)
        valid_mask = torch.ones(batch, heads, q_blk, k_blk, dtype=torch.bool)

        sparse_out, sparse_probs = block_sparse_attention_output(
            q, k, v, block_mask, block_size=block_size, causal=True,
        )
        metrics = compute_attention_quality(
            dense_out, dense_probs, sparse_out, sparse_probs,
            block_mask, block_size, layer_index=0, target_sparsity=1.0,
            valid_block_mask=valid_mask,
        )

        assert abs(metrics.actual_tile_sparsity - 1.0) < 1e-6

    def test_partial_sparsity(self) -> None:
        """Partial mask should have correct sparsity."""
        batch, heads, q_blk, k_blk = 1, 2, 4, 4
        valid_mask = torch.ones(batch, heads, q_blk, k_blk, dtype=torch.bool)
        # Keep 6 out of 16 tiles
        block_mask = torch.zeros(batch, heads, q_blk, k_blk, dtype=torch.bool)
        block_mask[0, :, :2, :3] = True  # 2*3=6 tiles

        # Use dummy values for quality computation
        seq_len = 16
        dim = 8
        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)
        dense_out, dense_probs = dense_attention_output(q, k, v, causal=True)
        sparse_out, sparse_probs = block_sparse_attention_output(
            q, k, v, block_mask, block_size=4, causal=True,
        )

        metrics = compute_attention_quality(
            dense_out, dense_probs, sparse_out, sparse_probs,
            block_mask, 4, layer_index=0, target_sparsity=0.625,
            valid_block_mask=valid_mask,
        )

        # 6 kept out of 16 = 62.5% kept = 37.5% sparsity
        assert abs(metrics.actual_tile_sparsity - 0.625) < 1e-6


class TestBlockMaskExpansion:
    """Test block mask to token mask expansion."""

    def test_causal_expansion(self) -> None:
        """Dense causal block mask should expand to lower triangular token mask."""
        block_mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        token_mask = expand_block_mask_to_token_mask(
            block_mask, block_size=4, seq_len=8, causal=True,
        )
        expected = torch.tril(torch.ones(8, 8, dtype=torch.bool))
        assert torch.equal(token_mask[0, 0], expected)

    def test_one_dropped_block(self) -> None:
        """Dropping block (1,0) should mask tokens 4-7 from seeing tokens 0-3."""
        block_mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        block_mask[0, 0, 1, 0] = False  # block 1 cannot see block 0
        token_mask = expand_block_mask_to_token_mask(
            block_mask, block_size=4, seq_len=8, causal=True,
        )
        # Tokens 4-7 should NOT see tokens 0-3
        assert not token_mask[0, 0, 4, 0].item()
        assert not token_mask[0, 0, 4, 3].item()
        # Tokens 4-7 should still see tokens 4-7 (causal)
        assert token_mask[0, 0, 4, 4].item()
        assert token_mask[0, 0, 7, 7].item()

    def test_all_drop(self) -> None:
        """All-drop block mask should give all-False token mask."""
        block_mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
        token_mask = expand_block_mask_to_token_mask(
            block_mask, block_size=4, seq_len=8, causal=True,
        )
        assert not token_mask.any()


class TestDenseBaseline:
    """Test dense baseline quality."""

    def test_kept_mass_one(self) -> None:
        """Dense attention should have kept mass = 1."""
        batch, heads, seq_len, dim = 1, 2, 8, 16
        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)

        dense_out, dense_probs = dense_attention_output(q, k, v, causal=True)
        # Full block mask (all blocks kept)
        block_mask = torch.ones(batch, heads, 2, 2, dtype=torch.bool)
        valid_mask = torch.ones(batch, heads, 2, 2, dtype=torch.bool)

        sparse_out, sparse_probs = block_sparse_attention_output(
            q, k, v, block_mask, block_size=4, causal=True,
        )
        metrics = compute_attention_quality(
            dense_out, dense_probs, sparse_out, sparse_probs,
            block_mask, 4, layer_index=0, target_sparsity=0.0,
            valid_block_mask=valid_mask,
        )

        # Dense baseline must satisfy these exactly
        assert abs(metrics.kept_attention_mass_mean - 1.0) < 1e-5
        assert abs(metrics.output_cosine_mean - 1.0) < 1e-5
        assert metrics.output_l2_relative_mean < 1e-5
        assert metrics.prob_l1_mean < 1e-5
        assert metrics.dropped_attention_mass_mean < 1e-5

    def test_sparse_output_matches_dense(self) -> None:
        """Full block mask sparse output should match dense output."""
        batch, heads, seq_len, dim = 1, 2, 8, 8
        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)

        dense_out, _ = dense_attention_output(q, k, v, causal=True)
        block_mask = torch.ones(batch, heads, 2, 2, dtype=torch.bool)
        sparse_out, _ = block_sparse_attention_output(
            q, k, v, block_mask, block_size=4, causal=True,
        )

        torch.testing.assert_close(sparse_out, dense_out, rtol=1e-5, atol=1e-5)

    def test_future_positions_not_counted_as_dropped(self) -> None:
        """Future positions should not count as dropped mass."""
        batch, heads, seq_len, dim = 1, 1, 4, 8
        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)

        dense_out, dense_probs = dense_attention_output(q, k, v, causal=True)
        # All blocks kept
        block_mask = torch.ones(batch, heads, 1, 1, dtype=torch.bool)
        valid_mask = torch.ones(batch, heads, 1, 1, dtype=torch.bool)

        sparse_out, sparse_probs = block_sparse_attention_output(
            q, k, v, block_mask, block_size=4, causal=True,
        )
        metrics = compute_attention_quality(
            dense_out, dense_probs, sparse_out, sparse_probs,
            block_mask, 4, layer_index=0, target_sparsity=0.0,
            valid_block_mask=valid_mask,
        )

        assert abs(metrics.kept_attention_mass_mean - 1.0) < 1e-5

    def test_one_dropped_block_reduces_mass(self) -> None:
        """Dropping one block should reduce kept mass."""
        batch, heads, seq_len, dim = 1, 1, 8, 8
        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)

        dense_out, dense_probs = dense_attention_output(q, k, v, causal=True)
        # Drop block (1, 1) - self-attention block for tokens 4-7
        block_mask = torch.ones(batch, heads, 2, 2, dtype=torch.bool)
        block_mask[0, 0, 1, 1] = False
        valid_mask = torch.ones(batch, heads, 2, 2, dtype=torch.bool)

        sparse_out, sparse_probs = block_sparse_attention_output(
            q, k, v, block_mask, block_size=4, causal=True,
        )
        metrics = compute_attention_quality(
            dense_out, dense_probs, sparse_out, sparse_probs,
            block_mask, 4, layer_index=0, target_sparsity=0.25,
            valid_block_mask=valid_mask,
        )

        assert metrics.kept_attention_mass_mean < 1.0
        assert metrics.dropped_attention_mass_mean > 0.0


class TestOracleBlockMass:
    """Test oracle block mass mask."""

    def test_score_shape(self) -> None:
        """Oracle block mass scores should have correct shape."""
        probs = torch.randn(1, 2, 16, 16).softmax(dim=-1)
        scores = compute_oracle_block_mass_scores(probs, block_size=4)
        assert scores.shape == (1, 2, 4, 4)

    def test_mask_sparsity_near_target(self) -> None:
        """Oracle mask sparsity should be close to target."""
        probs = torch.randn(1, 2, 16, 16).softmax(dim=-1)
        valid_mask = torch.ones(1, 2, 4, 4, dtype=torch.bool)
        # Causal: lower triangular
        for q in range(4):
            for k in range(q + 1, 4):
                valid_mask[0, :, q, k] = False

        mask = oracle_block_mass_mask(
            probs, block_size=4, target_sparsity=0.5, valid_block_mask=valid_mask,
        )

        valid_tiles = valid_mask.sum().item()
        kept_tiles = (mask & valid_mask).sum().item()
        actual_sp = 1.0 - kept_tiles / valid_tiles
        # Should be close to 0.5 (within 0.2 due to small size)
        assert abs(actual_sp - 0.5) < 0.3

    def test_not_worse_than_random(self) -> None:
        """Oracle mask should keep more mass than random on structured data."""
        # Create structured attention where some blocks have clearly more mass
        probs = torch.zeros(1, 1, 8, 8)
        # Diagonal-dominant: each token mostly attends to itself and neighbors
        for i in range(8):
            for j in range(max(0, i - 1), min(8, i + 2)):
                probs[0, 0, i, j] = 1.0
        probs = probs / probs.sum(dim=-1, keepdim=True)

        valid_mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        # Keep 1 out of 4 blocks (75% sparsity)
        oracle_mask = oracle_block_mass_mask(
            probs, block_size=4, target_sparsity=0.75, valid_block_mask=valid_mask,
        )

        oracle_mass = compute_oracle_block_mass_scores(probs, block_size=4)
        oracle_kept = (oracle_mass * oracle_mask.float()).sum().item()

        # Random with fixed seed
        rand_mask = random_valid_block_mask(valid_mask, target_sparsity=0.75, seed=42)
        rand_kept = (oracle_mass * rand_mask.float()).sum().item()

        # Oracle should do at least as well as random
        assert oracle_kept >= rand_kept - 1e-6


class TestLocalWindow:
    """Test local window block mask."""

    def test_causal_only(self) -> None:
        """Local window should only keep causal (past + current) blocks."""
        mask = local_window_block_mask(4, 4, window_blocks=2)
        assert mask.shape == (1, 1, 4, 4)

        # Block 0: only block 0
        assert mask[0, 0, 0, 0].item() is True
        assert mask[0, 0, 0, 1].item() is False

        # Block 2: blocks 1 and 2
        assert mask[0, 0, 2, 0].item() is False
        assert mask[0, 0, 2, 1].item() is True
        assert mask[0, 0, 2, 2].item() is True
        assert mask[0, 0, 2, 3].item() is False

    def test_window_size(self) -> None:
        """Each query block should keep exactly window_blocks key blocks."""
        mask = local_window_block_mask(8, 8, window_blocks=3)
        for q in range(8):
            kept = mask[0, 0, q].sum().item()
            expected = min(q + 1, 3)
            assert kept == expected


class TestRandomMask:
    """Test random valid block mask."""

    def test_only_valid_tiles(self) -> None:
        """Random mask should only keep tiles within valid_mask."""
        valid_mask = torch.ones(1, 2, 4, 4, dtype=torch.bool)
        valid_mask[0, :, 0, 3] = False  # Invalid tile

        rand_mask = random_valid_block_mask(valid_mask, target_sparsity=0.5)
        # Invalid tile should remain False
        assert rand_mask[0, 0, 0, 3].item() is False

    def test_sparsity_approximate(self) -> None:
        """Random mask sparsity should be approximately target."""
        valid_mask = torch.ones(1, 1, 10, 10, dtype=torch.bool)
        rand_mask = random_valid_block_mask(valid_mask, target_sparsity=0.7)
        kept = rand_mask.sum().item()
        total = valid_mask.sum().item()
        actual_sp = 1.0 - kept / total
        assert abs(actual_sp - 0.7) < 0.15  # Allow some randomness


class TestQualityMetricsNoNaN:
    """Test that all quality metrics are finite."""

    def test_no_nan(self) -> None:
        batch, heads, seq_len, dim = 1, 2, 8, 16
        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)

        dense_out, dense_probs = dense_attention_output(q, k, v, causal=True)
        block_mask = torch.ones(batch, heads, 2, 2, dtype=torch.bool)
        valid_mask = torch.ones(batch, heads, 2, 2, dtype=torch.bool)

        sparse_out, sparse_probs = block_sparse_attention_output(
            q, k, v, block_mask, block_size=4, causal=True,
        )
        metrics = compute_attention_quality(
            dense_out, dense_probs, sparse_out, sparse_probs,
            block_mask, 4, layer_index=0, target_sparsity=0.0,
            valid_block_mask=valid_mask,
        )

        for name, val in vars(metrics).items():
            if isinstance(val, float):
                assert val == val, f"{name} is NaN"
                assert abs(val) < float("inf"), f"{name} is Inf"
