"""Tests for candidate-pruned AGLR-C v2 candidate generation and scoring."""

from __future__ import annotations

import pytest
import torch

from certimask.candidate_pruning import (
    compute_candidate_antidiagonal_scores,
    compute_teacher_mask_overlap,
    compute_teacher_selected_coverage,
    generate_candidate_mask,
)


def _causal_valid_mask(num_blocks: int, device: torch.device) -> torch.Tensor:
    """Create a standard causal valid mask [1, 1, Q, K]."""
    q_idx = torch.arange(num_blocks, device=device).unsqueeze(1)
    k_idx = torch.arange(num_blocks, device=device).unsqueeze(0)
    return (k_idx <= q_idx).unsqueeze(0).unsqueeze(0)


class TestCandidateMaskShape:
    """Test that candidate masks have correct shapes."""

    @pytest.mark.parametrize("mode", [
        "local_stride", "block_norm", "coarse_to_fine", "head_pattern",
    ])
    def test_shape(self, mode: str) -> None:
        q = torch.randn(1, 2, 32, 8)
        k = torch.randn(1, 2, 32, 8)
        result = generate_candidate_mask(q, k, mode=mode, block_size=8)
        assert result.candidate_mask.shape == (1, 2, 4, 4)
        assert result.mode == mode

    def test_shape_with_valid_mask(self) -> None:
        q = torch.randn(2, 4, 64, 8)
        k = torch.randn(2, 4, 64, 8)
        valid = _causal_valid_mask(8, q.device).expand(2, 4, 8, 8)
        result = generate_candidate_mask(
            q, k, mode="local_stride", block_size=8, valid_mask=valid,
        )
        assert result.candidate_mask.shape == (2, 4, 8, 8)


class TestCandidateMaskCausal:
    """Test that candidate masks respect causal valid_mask."""

    @pytest.mark.parametrize("mode", [
        "local_stride", "block_norm", "coarse_to_fine", "head_pattern",
    ])
    def test_no_future_blocks(self, mode: str) -> None:
        q = torch.randn(1, 2, 32, 8)
        k = torch.randn(1, 2, 32, 8)
        result = generate_candidate_mask(q, k, mode=mode, block_size=8)
        # No candidate should be set for future blocks (k > q)
        for qi in range(4):
            for ki in range(qi + 1, 4):
                assert not result.candidate_mask[:, :, qi, ki].all()


class TestLocalStride:
    """Test local_stride mode specifics."""

    def test_includes_local_blocks(self) -> None:
        q = torch.randn(1, 1, 32, 8)
        k = torch.randn(1, 1, 32, 8)
        result = generate_candidate_mask(
            q, k, mode="local_stride", block_size=8, local_blocks=2,
        )
        # Query block 3: local blocks 2, 3 should be candidates
        assert result.candidate_mask[0, 0, 3, 2].item() is True
        assert result.candidate_mask[0, 0, 3, 3].item() is True

    def test_includes_first_block(self) -> None:
        q = torch.randn(1, 1, 32, 8)
        k = torch.randn(1, 1, 32, 8)
        result = generate_candidate_mask(
            q, k, mode="local_stride", block_size=8, local_blocks=2,
        )
        # Block 0 should be a candidate for all query blocks
        for qi in range(4):
            assert result.candidate_mask[0, 0, qi, 0].item() is True

    def test_candidate_fraction_controlled(self) -> None:
        q = torch.randn(1, 1, 128, 8)
        k = torch.randn(1, 1, 128, 8)
        result = generate_candidate_mask(
            q, k, mode="local_stride", block_size=8, local_blocks=2, stride=32,
        )
        # Should be well below 1.0
        assert result.candidate_fraction < 1.0
        # Should be above 0
        assert result.candidate_fraction > 0

    def test_no_full_pair_scoring(self) -> None:
        q = torch.randn(1, 1, 32, 8)
        k = torch.randn(1, 1, 32, 8)
        result = generate_candidate_mask(q, k, mode="local_stride")
        assert result.metadata["uses_full_pair_scoring"] is False
        assert result.metadata["uses_full_pair_proxy"] is False


class TestCoarseToFine:
    """Test coarse_to_fine mode specifics."""

    def test_output_shape(self) -> None:
        q = torch.randn(1, 2, 64, 8)
        k = torch.randn(1, 2, 64, 8)
        result = generate_candidate_mask(
            q, k, mode="coarse_to_fine", block_size=8,
            coarse_block_size=32, topk_coarse=4,
        )
        assert result.candidate_mask.shape == (1, 2, 8, 8)

    def test_candidate_fraction_roughly_controlled(self) -> None:
        q = torch.randn(1, 2, 512, 8)
        k = torch.randn(1, 2, 512, 8)
        result = generate_candidate_mask(
            q, k, mode="coarse_to_fine", block_size=8,
            coarse_block_size=64, topk_coarse=2,
        )
        # With 8 coarse blocks and topk=2, fraction should be less than 1.0
        # (local coarse blocks inflate the fraction)
        assert result.candidate_fraction < 1.0
        assert result.candidate_fraction > 0

    def test_no_full_pair_scoring(self) -> None:
        q = torch.randn(1, 1, 32, 8)
        k = torch.randn(1, 1, 32, 8)
        result = generate_candidate_mask(q, k, mode="coarse_to_fine")
        assert result.metadata["uses_full_pair_scoring"] is False
        assert result.metadata["uses_full_pair_proxy"] is False


class TestHeadPattern:
    """Test head_pattern mode specifics."""

    def test_different_heads_differ(self) -> None:
        q = torch.randn(1, 5, 32, 8)
        k = torch.randn(1, 5, 32, 8)
        result = generate_candidate_mask(q, k, mode="head_pattern", block_size=8)
        # Different heads should have different candidate patterns
        # (because they route to different sub-modes)
        h0_count = result.candidate_mask[0, 0].sum().item()
        h1_count = result.candidate_mask[0, 1].sum().item()
        h4_count = result.candidate_mask[0, 4].sum().item()
        # At least some heads should differ
        assert not (h0_count == h1_count == h4_count)


class TestCandidateOnlyScoring:
    """Test candidate-only AGLR scoring."""

    def test_non_candidates_set_to_neginf(self) -> None:
        q = torch.randn(1, 1, 32, 8)
        k = torch.randn(1, 1, 32, 8)
        # Create a sparse candidate mask
        candidate_mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        candidate_mask[0, 0, 0, 0] = True
        candidate_mask[0, 0, 1, 0] = True
        candidate_mask[0, 0, 1, 1] = True

        scores, meta = compute_candidate_antidiagonal_scores(
            q, k, candidate_mask, block_size=8,
        )
        # Non-candidate valid tiles should be -inf
        assert scores[0, 0, 0, 1].item() == float("-inf")
        assert scores[0, 0, 2, 0].item() == float("-inf")
        # Candidate tiles should have finite scores
        assert torch.isfinite(scores[0, 0, 0, 0]).item()
        assert torch.isfinite(scores[0, 0, 1, 1]).item()

    def test_invalid_tiles_set_to_neginf(self) -> None:
        q = torch.randn(1, 1, 32, 8)
        k = torch.randn(1, 1, 32, 8)
        # All tiles are candidates, but future tiles are invalid
        candidate_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        candidate_mask = candidate_mask & _causal_valid_mask(4, q.device)

        scores, meta = compute_candidate_antidiagonal_scores(
            q, k, candidate_mask, block_size=8,
        )
        # Future tiles should be -inf
        assert scores[0, 0, 0, 1].item() == float("-inf")
        assert scores[0, 0, 0, 3].item() == float("-inf")

    def test_computed_tile_count(self) -> None:
        q = torch.randn(1, 1, 32, 8)
        k = torch.randn(1, 1, 32, 8)
        candidate_mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        candidate_mask[0, 0, 0, 0] = True
        candidate_mask[0, 0, 1, 1] = True
        candidate_mask[0, 0, 2, 2] = True

        scores, meta = compute_candidate_antidiagonal_scores(
            q, k, candidate_mask, block_size=8,
        )
        assert meta["computed_tile_count"] == 3
        assert meta["uses_full_pair_scoring"] is False

    def test_empty_candidates(self) -> None:
        q = torch.randn(1, 1, 32, 8)
        k = torch.randn(1, 1, 32, 8)
        candidate_mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)

        scores, meta = compute_candidate_antidiagonal_scores(
            q, k, candidate_mask, block_size=8,
        )
        assert meta["computed_tile_count"] == 0
        assert (scores == float("-inf")).all()


class TestTeacherCoverage:
    """Test teacher coverage metrics."""

    def test_full_coverage(self) -> None:
        # Candidate covers all teacher-selected tiles
        candidate = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        teacher = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        valid = _causal_valid_mask(4, torch.device("cpu")).expand(1, 1, 4, 4)

        coverage = compute_teacher_selected_coverage(candidate, teacher, valid)
        assert coverage == 1.0

    def test_partial_coverage(self) -> None:
        # Teacher selects 4 tiles, candidate covers 2 of them
        valid = _causal_valid_mask(4, torch.device("cpu")).expand(1, 1, 4, 4)
        teacher = valid.clone()
        candidate = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        candidate[0, 0, 0, 0] = True
        candidate[0, 0, 1, 1] = True

        coverage = compute_teacher_selected_coverage(candidate, teacher, valid)
        # Teacher has 10 valid tiles (causal), candidate covers 2
        assert coverage == pytest.approx(2.0 / 10.0)

    def test_no_teacher_selected(self) -> None:
        candidate = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        teacher = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        valid = _causal_valid_mask(4, torch.device("cpu")).expand(1, 1, 4, 4)

        coverage = compute_teacher_selected_coverage(candidate, teacher, valid)
        assert coverage == 1.0  # vacuously true

    def test_overlap_identical_masks(self) -> None:
        mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        valid = _causal_valid_mask(4, torch.device("cpu")).expand(1, 1, 4, 4)

        overlap = compute_teacher_mask_overlap(mask, mask, valid)
        assert overlap == 1.0

    def test_overlap_disjoint_masks(self) -> None:
        valid = _causal_valid_mask(4, torch.device("cpu")).expand(1, 1, 4, 4)
        pruned = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        pruned[0, 0, 0, 0] = True
        teacher = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        teacher[0, 0, 1, 1] = True

        overlap = compute_teacher_mask_overlap(pruned, teacher, valid)
        assert overlap == 0.0


class TestCandidateMaskResult:
    """Test CandidateMaskResult dataclass."""

    def test_metadata_populated(self) -> None:
        q = torch.randn(1, 1, 32, 8)
        k = torch.randn(1, 1, 32, 8)
        result = generate_candidate_mask(q, k, mode="local_stride")
        assert "candidate_tiles" in result.metadata
        assert "valid_tiles" in result.metadata
        assert isinstance(result.candidate_fraction, float)
        assert isinstance(result.valid_fraction, float)


class TestInvalidMode:
    """Test error handling for invalid mode."""

    def test_unknown_mode_raises(self) -> None:
        q = torch.randn(1, 1, 32, 8)
        k = torch.randn(1, 1, 32, 8)
        with pytest.raises(ValueError, match="Unknown mode"):
            generate_candidate_mask(q, k, mode="invalid_mode")
