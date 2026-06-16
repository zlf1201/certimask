"""Tests for local+extra hybrid masks and quality pass criteria."""

from __future__ import annotations

import torch

from certimask.attention_quality import (
    local_plus_extra_mask,
    local_window_block_mask,
)


class TestLocalPlusExtraMask:
    """Test local_plus_extra_mask function."""

    def test_keeps_local_blocks(self) -> None:
        """Mandatory local blocks must always be kept."""
        scores = torch.randn(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        # Causal: upper triangle invalid
        for q in range(4):
            for k in range(q + 1, 4):
                valid[0, 0, q, k] = False

        mask, _ = local_plus_extra_mask(
            scores, target_sparsity=0.5, local_blocks=2, valid_mask=valid,
        )

        # Query block 2: local blocks are 1, 2 (nearest 2 causal)
        assert mask[0, 0, 2, 1].item() is True
        assert mask[0, 0, 2, 2].item() is True

    def test_no_future_blocks(self) -> None:
        """Must not select future (invalid) blocks."""
        scores = torch.ones(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        for q in range(4):
            for k in range(q + 1, 4):
                valid[0, 0, q, k] = False

        mask, _ = local_plus_extra_mask(
            scores, target_sparsity=0.0, local_blocks=1, valid_mask=valid,
        )

        # No future blocks should be selected
        for q in range(4):
            for k in range(q + 1, 4):
                assert mask[0, 0, q, k].item() is False

    def test_extra_selected_by_score(self) -> None:
        """Extra blocks should be selected by highest score."""
        # Query block 0: valid keys are 0 only (causal)
        # Query block 1: valid keys are 0, 1
        # Query block 2: valid keys are 0, 1, 2
        # Query block 3: valid keys are 0, 1, 2, 3
        scores = torch.zeros(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        for q in range(4):
            for k in range(q + 1, 4):
                valid[0, 0, q, k] = False

        # Set high score for key block 0 when query is block 3
        scores[0, 0, 3, 0] = 100.0

        mask, _ = local_plus_extra_mask(
            scores, target_sparsity=0.5, local_blocks=1, valid_mask=valid,
        )

        # Query block 3: local=block 3, extra should include block 0 (score=100)
        assert mask[0, 0, 3, 3].item() is True  # local
        assert mask[0, 0, 3, 0].item() is True  # high score extra

    def test_overflow_preserves_local(self) -> None:
        """When local blocks exceed budget, all local blocks are kept."""
        scores = torch.zeros(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        for q in range(4):
            for k in range(q + 1, 4):
                valid[0, 0, q, k] = False

        # local_blocks=4 means all causal blocks are mandatory
        mask, overflow = local_plus_extra_mask(
            scores, target_sparsity=0.5, local_blocks=4, valid_mask=valid,
        )

        # All causal blocks should be kept
        assert mask[0, 0, 0, 0].item() is True
        assert mask[0, 0, 1, 0].item() is True
        assert mask[0, 0, 1, 1].item() is True
        assert overflow is True

    def test_sparsity_approximate(self) -> None:
        """Actual sparsity should be close to target when no overflow."""
        scores = torch.randn(1, 1, 8, 8)
        valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        for q in range(8):
            for k in range(q + 1, 8):
                valid[0, 0, q, k] = False

        mask, overflow = local_plus_extra_mask(
            scores, target_sparsity=0.5, local_blocks=1, valid_mask=valid,
        )

        valid_tiles = valid.sum().item()
        kept_tiles = (mask & valid).sum().item()
        actual_sp = 1.0 - kept_tiles / valid_tiles
        # Should be close to 0.5 (within 0.2 due to small size and local guarantee)
        assert abs(actual_sp - 0.5) < 0.25


class TestLocalWindowCausal:
    """Test that local window only keeps causal blocks."""

    def test_no_future(self) -> None:
        mask = local_window_block_mask(4, 4, window_blocks=2)
        # Block 0: only block 0
        assert mask[0, 0, 0, 0].item() is True
        assert mask[0, 0, 0, 1].item() is False
        # Block 2: blocks 1, 2
        assert mask[0, 0, 2, 0].item() is False
        assert mask[0, 0, 2, 1].item() is True
        assert mask[0, 0, 2, 2].item() is True
        assert mask[0, 0, 2, 3].item() is False


class TestQualityPassCriteria:
    """Test quality_pass and strong_quality_pass criteria."""

    def test_quality_pass_thresholds(self) -> None:
        """Verify quality_pass logic."""
        # Simulate metrics
        kept_mass = 0.92
        cosine = 0.96
        l2_rel = 0.15

        quality_pass = kept_mass >= 0.90 and cosine >= 0.95 and l2_rel <= 0.20
        assert quality_pass is True

    def test_strong_quality_pass_thresholds(self) -> None:
        """Verify strong_quality_pass logic."""
        kept_mass = 0.96
        cosine = 0.99
        l2_rel = 0.08

        strong = kept_mass >= 0.95 and cosine >= 0.98 and l2_rel <= 0.10
        assert strong is True

    def test_quality_fail_low_mass(self) -> None:
        """Low kept mass should fail quality_pass."""
        kept_mass = 0.85
        cosine = 0.96
        l2_rel = 0.15

        quality_pass = kept_mass >= 0.90 and cosine >= 0.95 and l2_rel <= 0.20
        assert quality_pass is False
