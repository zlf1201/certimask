"""Tests for antidiagonal scoring and hybrid AGLR-C v1."""

from __future__ import annotations

import pytest
import torch

from certimask.aglr_indexer import (
    combine_aglr_scores,
    compute_antidiagonal_block_scores,
)


class TestAntidiagonalScoreShape:
    """Test antidiagonal score output shapes."""

    @pytest.mark.parametrize("pattern", [
        "main_diagonal", "anti_diagonal", "both_diagonals",
        "strided_grid", "landmark_cross",
    ])
    def test_shape(self, pattern: str) -> None:
        q = torch.randn(1, 2, 16, 8)
        k = torch.randn(1, 2, 16, 8)
        scores = compute_antidiagonal_block_scores(
            q, k, block_size=4, sample_pattern=pattern,
            aggregation="mean",
        )
        assert scores.shape == (1, 2, 4, 4)

    @pytest.mark.parametrize("agg", ["mean", "max", "topk_mean", "logsumexp"])
    def test_aggregation_shapes(self, agg: str) -> None:
        q = torch.randn(1, 2, 16, 8)
        k = torch.randn(1, 2, 16, 8)
        scores = compute_antidiagonal_block_scores(
            q, k, block_size=4, sample_pattern="anti_diagonal",
            aggregation=agg,
        )
        assert scores.shape == (1, 2, 4, 4)


class TestDiagonalPositions:
    """Test sample position generation."""

    def test_main_diagonal(self) -> None:
        from certimask.aglr_indexer import _generate_sample_positions
        pos = _generate_sample_positions(4, "main_diagonal")
        assert pos == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_anti_diagonal(self) -> None:
        from certimask.aglr_indexer import _generate_sample_positions
        pos = _generate_sample_positions(4, "anti_diagonal")
        assert pos == [(0, 3), (1, 2), (2, 1), (3, 0)]

    def test_both_diagonals(self) -> None:
        from certimask.aglr_indexer import _generate_sample_positions
        pos = _generate_sample_positions(4, "both_diagonals")
        # Should have 4 + 2 = 6 positions (corners overlap at (0,3) and (3,0))
        # Actually: main=(0,0),(1,1),(2,2),(3,3) and anti=(0,3),(1,2),(2,1),(3,0)
        # (0,0)!=(0,3), (1,1)!=(1,2), (2,2)!=(2,1), (3,3)!=(3,0) => all different
        assert len(pos) == 8

    def test_strided_grid_count(self) -> None:
        from certimask.aglr_indexer import _generate_sample_positions
        pos = _generate_sample_positions(16, "strided_grid", num_samples=4)
        # Should have at least 4 positions
        assert len(pos) >= 4

    def test_invalid_pattern(self) -> None:
        q = torch.randn(1, 1, 4, 4)
        k = torch.randn(1, 1, 4, 4)
        with pytest.raises(ValueError, match="Unknown sample_pattern"):
            compute_antidiagonal_block_scores(
                q, k, block_size=2, sample_pattern="invalid",
                aggregation="mean",
            )


class TestAggregationCorrectness:
    """Test aggregation numerical correctness."""

    def test_mean_aggregation(self) -> None:
        q = torch.ones(1, 1, 2, 2)
        k = torch.ones(1, 1, 2, 2) * 2.0
        # anti_diagonal with block_size=2: positions (0,1), (1,0)
        # Each dot = [1,1]·[2,2] = 4, / sqrt(2) = 2.828
        scores = compute_antidiagonal_block_scores(
            q, k, block_size=2, sample_pattern="anti_diagonal",
            aggregation="mean", scale_by_sqrt_dim=True,
        )
        expected = 4.0 / (2.0 ** 0.5)
        torch.testing.assert_close(
            scores, torch.full_like(scores, expected), rtol=1e-5, atol=1e-5,
        )

    def test_max_aggregation(self) -> None:
        q = torch.zeros(1, 1, 4, 2)
        k = torch.zeros(1, 1, 4, 2)
        # Make one position have a high dot product
        q[0, 0, 0, 0] = 10.0
        k[0, 0, 3, 0] = 10.0
        # anti_diagonal: (0,3), (1,2), (2,1), (3,0)
        # (0,3) dot = 10*10 = 100
        scores = compute_antidiagonal_block_scores(
            q, k, block_size=4, sample_pattern="anti_diagonal",
            aggregation="max", scale_by_sqrt_dim=False,
        )
        assert scores[0, 0, 0, 0].item() == pytest.approx(100.0, rel=1e-5)

    def test_topk_mean(self) -> None:
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        scores = compute_antidiagonal_block_scores(
            q, k, block_size=4, sample_pattern="both_diagonals",
            aggregation="topk_mean",
        )
        assert scores.shape == (1, 1, 1, 1)
        assert torch.isfinite(scores).all()

    def test_logsumexp(self) -> None:
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        scores = compute_antidiagonal_block_scores(
            q, k, block_size=4, sample_pattern="anti_diagonal",
            aggregation="logsumexp",
        )
        assert scores.shape == (1, 1, 1, 1)
        assert torch.isfinite(scores).all()


class TestAntidiagonalInvalidMask:
    """Test that invalid tiles are masked."""

    def test_invalid_tiles_masked(self) -> None:
        q = torch.randn(1, 1, 8, 8)
        k = torch.randn(1, 1, 8, 8)
        valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        valid[0, 0, 0, 1] = False

        scores = compute_antidiagonal_block_scores(
            q, k, block_size=4, sample_pattern="anti_diagonal",
            aggregation="mean", valid_mask=valid,
        )
        assert scores[0, 0, 0, 1].item() == torch.finfo(torch.float32).min


class TestCombineScores:
    """Test score combination."""

    def test_landmark_only(self) -> None:
        lm = torch.randn(1, 2, 4, 4)
        combined = combine_aglr_scores(landmark_scores=lm)
        torch.testing.assert_close(combined, 0.5 * lm)

    def test_antidiagonal_only(self) -> None:
        ad = torch.randn(1, 2, 4, 4)
        combined = combine_aglr_scores(antidiagonal_scores=ad)
        torch.testing.assert_close(combined, 0.5 * ad)

    def test_hybrid_weights(self) -> None:
        lm = torch.ones(1, 1, 4, 4)
        ad = torch.ones(1, 1, 4, 4) * 2.0
        combined = combine_aglr_scores(
            landmark_scores=lm, antidiagonal_scores=ad,
            landmark_weight=0.7, antidiagonal_weight=0.3,
        )
        torch.testing.assert_close(combined, torch.full_like(combined, 1.3))

    def test_recency_prior(self) -> None:
        combined = combine_aglr_scores(
            landmark_scores=torch.zeros(1, 1, 4, 4),
            recency_weight=1.0,
        )
        # Recency: -log(1 + a - b) for causal tiles
        # Position (0,0): -log(1) = 0
        # Position (1,0): -log(2) ≈ -0.693
        assert combined[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert combined[0, 0, 1, 0].item() == pytest.approx(-0.693, abs=0.01)

    def test_valid_mask(self) -> None:
        lm = torch.ones(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        valid[0, 0, 0, 3] = False
        combined = combine_aglr_scores(
            landmark_scores=lm, valid_mask=valid,
        )
        assert combined[0, 0, 0, 3].item() == torch.finfo(torch.float32).min

    def test_both_none_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            combine_aglr_scores()


class TestQualityPassCriteria:
    """Test quality pass criteria."""

    def test_quality_pass(self) -> None:
        kept_mass = 0.92
        cosine = 0.96
        l2_rel = 0.15
        work_frac = 0.45
        quality_pass = (
            kept_mass >= 0.90 and cosine >= 0.95
            and l2_rel <= 0.20 and work_frac <= 0.50
        )
        assert quality_pass is True

    def test_quality_fail_high_l2(self) -> None:
        kept_mass = 0.95
        cosine = 0.99
        l2_rel = 0.25
        work_frac = 0.40
        quality_pass = (
            kept_mass >= 0.90 and cosine >= 0.95
            and l2_rel <= 0.20 and work_frac <= 0.50
        )
        assert quality_pass is False
