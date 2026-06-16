"""Tests for AGLR-C v0 reference indexer."""

from __future__ import annotations

import pytest
import torch

from certimask.aglr_indexer import (
    aglr_adaptive_mass_budget_mask,
    aglr_local_plus_landmark_mask,
    compute_landmark_block_scores,
    select_block_landmarks,
)


class TestSelectBlockLandmarks:
    """Test landmark selection methods."""

    def test_mean_shape(self) -> None:
        states = torch.randn(1, 2, 16, 8)
        lm = select_block_landmarks(states, block_size=4, method="mean")
        assert lm.landmarks.shape == (1, 2, 4, 1, 8)
        assert lm.num_blocks == 4
        assert lm.num_landmarks == 1

    def test_last_shape(self) -> None:
        states = torch.randn(1, 2, 16, 8)
        lm = select_block_landmarks(states, block_size=4, method="last")
        assert lm.landmarks.shape == (1, 2, 4, 1, 8)

    def test_max_norm_selects_highest_norm(self) -> None:
        states = torch.zeros(1, 1, 4, 2)
        states[0, 0, 0] = torch.tensor([1.0, 0.0])
        states[0, 0, 1] = torch.tensor([0.0, 0.0])
        states[0, 0, 2] = torch.tensor([0.0, 0.0])
        states[0, 0, 3] = torch.tensor([0.0, 3.0])  # highest norm
        lm = select_block_landmarks(states, block_size=4, method="max_norm")
        assert lm.landmarks.shape == (1, 1, 1, 1, 2)
        expected = torch.tensor([0.0, 3.0])
        torch.testing.assert_close(lm.landmarks[0, 0, 0, 0], expected)

    def test_topk_norm_shape(self) -> None:
        states = torch.randn(1, 2, 16, 8)
        lm = select_block_landmarks(
            states, block_size=4, method="topk_norm", num_landmarks=2,
        )
        assert lm.landmarks.shape == (1, 2, 4, 2, 8)

    def test_mean_plus_max_norm_r2(self) -> None:
        states = torch.randn(1, 2, 16, 8)
        lm = select_block_landmarks(states, block_size=4, method="mean_plus_max_norm")
        assert lm.landmarks.shape == (1, 2, 4, 2, 8)
        assert lm.num_landmarks == 2

    def test_mean_plus_topk_norm_shape(self) -> None:
        states = torch.randn(1, 2, 16, 8)
        lm = select_block_landmarks(
            states, block_size=4, method="mean_plus_topk_norm", num_landmarks=2,
        )
        assert lm.landmarks.shape == (1, 2, 4, 3, 8)  # 1 mean + 2 topk
        assert lm.num_landmarks == 3

    def test_invalid_method(self) -> None:
        states = torch.randn(1, 1, 8, 4)
        with pytest.raises(ValueError, match="Unknown method"):
            select_block_landmarks(states, block_size=4, method="invalid")

    def test_invalid_block_size(self) -> None:
        states = torch.randn(1, 1, 8, 4)
        with pytest.raises(ValueError, match="positive"):
            select_block_landmarks(states, block_size=0, method="mean")

    def test_seq_shorter_than_block(self) -> None:
        states = torch.randn(1, 1, 3, 4)
        with pytest.raises(ValueError, match="no complete blocks"):
            select_block_landmarks(states, block_size=4, method="mean")

    def test_non_floating(self) -> None:
        states = torch.randint(0, 10, (1, 1, 8, 4))
        with pytest.raises(TypeError, match="floating-point"):
            select_block_landmarks(states, block_size=4, method="mean")


class TestComputeLandmarkBlockScores:
    """Test landmark block scoring methods."""

    def test_max_score_shape(self) -> None:
        q = torch.randn(1, 2, 16, 8)
        k = torch.randn(1, 2, 16, 8)
        q_lm = select_block_landmarks(q, block_size=4, method="mean")
        k_lm = select_block_landmarks(k, block_size=4, method="mean")
        scores = compute_landmark_block_scores(
            q_lm, k_lm, score_method="max",
        )
        assert scores.shape == (1, 2, 4, 4)

    def test_mean_score_shape(self) -> None:
        q = torch.randn(1, 2, 16, 8)
        k = torch.randn(1, 2, 16, 8)
        q_lm = select_block_landmarks(q, block_size=4, method="mean")
        k_lm = select_block_landmarks(k, block_size=4, method="mean")
        scores = compute_landmark_block_scores(
            q_lm, k_lm, score_method="mean",
        )
        assert scores.shape == (1, 2, 4, 4)

    def test_hybrid_score_shape(self) -> None:
        q = torch.randn(1, 2, 16, 8)
        k = torch.randn(1, 2, 16, 8)
        q_lm = select_block_landmarks(q, block_size=4, method="mean_plus_max_norm")
        k_lm = select_block_landmarks(k, block_size=4, method="mean_plus_max_norm")
        scores = compute_landmark_block_scores(
            q_lm, k_lm, score_method="hybrid",
        )
        assert scores.shape == (1, 2, 4, 4)

    def test_invalid_tiles_masked(self) -> None:
        q = torch.randn(1, 1, 8, 4)
        k = torch.randn(1, 1, 8, 4)
        q_lm = select_block_landmarks(q, block_size=4, method="mean")
        k_lm = select_block_landmarks(k, block_size=4, method="mean")
        valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        valid[0, 0, 0, 1] = False
        scores = compute_landmark_block_scores(
            q_lm, k_lm, score_method="max", valid_mask=valid,
        )
        assert scores[0, 0, 0, 1].item() == torch.finfo(torch.float32).min

    def test_max_score_correctness(self) -> None:
        """With single-landmark mean, max score = mean dot product."""
        q = torch.ones(1, 1, 4, 2)
        k = torch.ones(1, 1, 4, 2) * 2.0
        q_lm = select_block_landmarks(q, block_size=2, method="mean")
        k_lm = select_block_landmarks(k, block_size=2, method="mean")
        scores = compute_landmark_block_scores(
            q_lm, k_lm, score_method="max", scale_by_sqrt_dim=True,
        )
        # mean q = [1,1], mean k = [2,2], dot = 4, / sqrt(2) ≈ 2.83
        expected = 4.0 / (2.0 ** 0.5)
        torch.testing.assert_close(
            scores, torch.full_like(scores, expected), rtol=1e-5, atol=1e-5,
        )


class TestAGRLocalPlusLandmarkMask:
    """Test AGLR local + landmark mask."""

    def test_no_future_blocks(self) -> None:
        scores = torch.ones(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        for q in range(4):
            for k in range(q + 1, 4):
                valid[0, 0, q, k] = False
        result = aglr_local_plus_landmark_mask(
            scores, target_keep_fraction=0.5, local_blocks=1, valid_mask=valid,
        )
        for q in range(4):
            for k in range(q + 1, 4):
                assert result.mask[0, 0, q, k].item() is False

    def test_keeps_local_blocks(self) -> None:
        scores = torch.randn(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        for q in range(4):
            for k in range(q + 1, 4):
                valid[0, 0, q, k] = False
        result = aglr_local_plus_landmark_mask(
            scores, target_keep_fraction=0.5, local_blocks=2, valid_mask=valid,
        )
        # Query block 3: local blocks 2, 3
        assert result.local_mask[0, 0, 3, 2].item() is True
        assert result.local_mask[0, 0, 3, 3].item() is True

    def test_budget_correct(self) -> None:
        scores = torch.randn(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        for q in range(4):
            for k in range(q + 1, 4):
                valid[0, 0, q, k] = False
        result = aglr_local_plus_landmark_mask(
            scores, target_keep_fraction=0.5, local_blocks=1, valid_mask=valid,
        )
        # Row 3: 4 valid keys, budget = ceil(0.5 * 4) = 2
        kept_row3 = result.mask[0, 0, 3].sum().item()
        assert kept_row3 == 2

    def test_overflow_preserves_local(self) -> None:
        scores = torch.randn(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        for q in range(4):
            for k in range(q + 1, 4):
                valid[0, 0, q, k] = False
        result = aglr_local_plus_landmark_mask(
            scores, target_keep_fraction=0.1, local_blocks=2, valid_mask=valid,
        )
        # Local blocks should still be kept even if overflow
        assert result.local_mask[0, 0, 3, 2].item() is True
        assert result.local_mask[0, 0, 3, 3].item() is True

    def test_sparsity_reasonable(self) -> None:
        scores = torch.randn(1, 2, 8, 8)
        valid = torch.ones(1, 2, 8, 8, dtype=torch.bool)
        for q in range(8):
            for k in range(q + 1, 8):
                valid[0, :, q, k] = False
        result = aglr_local_plus_landmark_mask(
            scores, target_keep_fraction=0.5, local_blocks=1, valid_mask=valid,
        )
        # Work fraction should be reasonable
        assert 0.3 < result.attention_tile_work_fraction < 0.8

    def test_both_sparsity_and_keep_raises(self) -> None:
        scores = torch.randn(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match="only one"):
            aglr_local_plus_landmark_mask(
                scores, target_sparsity=0.5, target_keep_fraction=0.5,
                local_blocks=1, valid_mask=valid,
            )

    def test_neither_sparsity_nor_keep_raises(self) -> None:
        scores = torch.randn(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match="Must specify"):
            aglr_local_plus_landmark_mask(
                scores, local_blocks=1, valid_mask=valid,
            )


class TestAGLRAdaptiveMassBudget:
    """Test adaptive mass budget mask."""

    def test_keeps_local_blocks(self) -> None:
        scores = torch.randn(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        for q in range(4):
            for k in range(q + 1, 4):
                valid[0, 0, q, k] = False
        result = aglr_adaptive_mass_budget_mask(
            scores, target_proxy_mass=0.9, local_blocks=2, valid_mask=valid,
        )
        assert result.local_mask[0, 0, 3, 2].item() is True
        assert result.local_mask[0, 0, 3, 3].item() is True

    def test_mask_shape(self) -> None:
        scores = torch.randn(1, 2, 4, 4)
        valid = torch.ones(1, 2, 4, 4, dtype=torch.bool)
        for q in range(4):
            for k in range(q + 1, 4):
                valid[0, :, q, k] = False
        result = aglr_adaptive_mass_budget_mask(
            scores, target_proxy_mass=0.8, local_blocks=1, valid_mask=valid,
        )
        assert result.mask.shape == (1, 2, 4, 4)


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

    def test_strong_quality_pass(self) -> None:
        kept_mass = 0.96
        cosine = 0.99
        l2_rel = 0.08
        work_frac = 0.40
        strong = (
            kept_mass >= 0.95 and cosine >= 0.98
            and l2_rel <= 0.10 and work_frac <= 0.50
        )
        assert strong is True

    def test_quality_fail_high_work(self) -> None:
        kept_mass = 0.95
        cosine = 0.99
        l2_rel = 0.05
        work_frac = 0.55
        quality_pass = (
            kept_mass >= 0.90 and cosine >= 0.95
            and l2_rel <= 0.20 and work_frac <= 0.50
        )
        assert quality_pass is False
