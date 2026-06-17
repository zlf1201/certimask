"""Tests for vectorized top-k mask construction."""

import pytest
import torch

from certimask.vectorized_topk import VectorizedTopKMaskResult, vectorized_topk_mask


def _make_causal_mask(q_blk: int, k_blk: int, device: str = "cpu") -> torch.Tensor:
    """Create a simple causal valid mask."""
    q_idx = torch.arange(q_blk, device=device).unsqueeze(1)
    k_idx = torch.arange(k_blk, device=device).unsqueeze(0)
    mask = k_idx <= q_idx
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, K]


class TestVectorizedTopKBasic:
    """Basic functionality tests."""

    def test_output_shape(self) -> None:
        """Output mask has correct shape."""
        scores = torch.randn(2, 4, 8, 8)
        valid_mask = _make_causal_mask(8, 8).expand(2, 4, 8, 8)
        k_per_row = torch.full((2, 4, 8), 3, dtype=torch.long)

        result = vectorized_topk_mask(scores, k_per_row=k_per_row, valid_mask=valid_mask)

        assert isinstance(result, VectorizedTopKMaskResult)
        assert result.mask.shape == (2, 4, 8, 8)
        assert result.mask.dtype == torch.bool

    def test_k_equals_one(self) -> None:
        """Selecting k=1 keeps exactly one block per row."""
        scores = torch.tensor([[[[5.0, 3.0, 1.0, 0.0]]]])
        valid_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
        k_per_row = torch.tensor([[[1]]])

        result = vectorized_topk_mask(scores, k_per_row=k_per_row, valid_mask=valid_mask)

        assert result.mask.sum().item() == 1
        assert result.mask[0, 0, 0, 0].item()  # highest score selected

    def test_k_equals_all_valid(self) -> None:
        """Selecting k=all keeps all valid blocks."""
        scores = torch.tensor([[[[5.0, 3.0, 1.0, 0.0]]]])
        valid_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
        k_per_row = torch.tensor([[[4]]])

        result = vectorized_topk_mask(scores, k_per_row=k_per_row, valid_mask=valid_mask)

        assert result.mask.all()

    def test_invalid_tiles_never_selected(self) -> None:
        """Invalid (future) tiles are never selected."""
        scores = torch.tensor([[[[5.0, 3.0, 1.0, 0.0]]]])
        # Only first two tiles are valid (causal)
        valid_mask = torch.tensor([[[[True, True, False, False]]]])
        k_per_row = torch.tensor([[[4]]])  # want 4 but only 2 valid

        result = vectorized_topk_mask(scores, k_per_row=k_per_row, valid_mask=valid_mask)

        assert result.mask[0, 0, 0, 0].item()
        assert result.mask[0, 0, 0, 1].item()
        assert not result.mask[0, 0, 0, 2].item()
        assert not result.mask[0, 0, 0, 3].item()


class TestVectorizedTopKVariableK:
    """Tests with variable k_per_row across rows."""

    def test_variable_k_per_row(self) -> None:
        """Different rows can have different k values."""
        scores = torch.tensor([[[[5.0, 3.0, 1.0],
                                  [2.0, 4.0, 6.0]]]])
        valid_mask = torch.ones(1, 1, 2, 3, dtype=torch.bool)
        k_per_row = torch.tensor([[[1, 2]]])  # row 0: k=1, row 1: k=2

        result = vectorized_topk_mask(scores, k_per_row=k_per_row, valid_mask=valid_mask)

        # Row 0: keep only top-1 (index 0, score 5.0)
        assert result.mask[0, 0, 0, 0].item()
        assert result.mask[0, 0, 0, 1:].sum().item() == 0

        # Row 1: keep top-2 (indices 2, 1 with scores 6.0, 4.0)
        assert result.mask[0, 0, 1, 2].item()
        assert result.mask[0, 0, 1, 1].item()
        assert not result.mask[0, 0, 1, 0].item()


class TestVectorizedTopKMandatory:
    """Tests with mandatory keep mask."""

    def test_mandatory_blocks_always_kept(self) -> None:
        """Mandatory blocks are kept even if their score is low."""
        scores = torch.tensor([[[[1.0, 5.0, 3.0, 4.0]]]])
        valid_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
        # Keep 2 total, but block 0 is mandatory (low score)
        mandatory = torch.tensor([[[[True, False, False, False]]]])
        k_per_row = torch.tensor([[[2]]])

        result = vectorized_topk_mask(
            scores, k_per_row=k_per_row, valid_mask=valid_mask,
            mandatory_keep_mask=mandatory,
        )

        # Mandatory block 0 must be kept
        assert result.mask[0, 0, 0, 0].item()
        # Plus one more from top-scored (block 1 with score 5.0)
        assert result.mask[0, 0, 0, 1].item()
        assert result.mask.sum().item() == 2

    def test_mandatory_overflow_retained(self) -> None:
        """When mandatory count > k, all mandatory blocks are still kept."""
        scores = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        valid_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
        # k=1 but 3 mandatory blocks
        mandatory = torch.tensor([[[[True, True, True, False]]]])
        k_per_row = torch.tensor([[[1]]])

        result = vectorized_topk_mask(
            scores, k_per_row=k_per_row, valid_mask=valid_mask,
            mandatory_keep_mask=mandatory,
        )

        # All 3 mandatory blocks kept despite k=1
        assert result.mask[0, 0, 0, 0].item()
        assert result.mask[0, 0, 0, 1].item()
        assert result.mask[0, 0, 0, 2].item()
        assert not result.mask[0, 0, 0, 3].item()


class TestVectorizedTopKDeterministic:
    """Tests for deterministic tie-breaking."""

    def test_ties_deterministic(self) -> None:
        """Equal scores produce consistent selection."""
        scores = torch.tensor([[[[5.0, 5.0, 5.0, 5.0]]]])
        valid_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
        k_per_row = torch.tensor([[[2]]])

        # Run multiple times - should get same result
        results = []
        for _ in range(5):
            r = vectorized_topk_mask(scores, k_per_row=k_per_row, valid_mask=valid_mask)
            results.append(r.mask.clone())

        for r in results[1:]:
            assert torch.equal(r, results[0])


class TestVectorizedTopKNoSync:
    """Tests that no CPU-GPU sync is required."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_no_cpu_transfer(self) -> None:
        """Operations stay on GPU without .item() calls in hot path."""
        device = "cuda"
        scores = torch.randn(1, 14, 128, 128, device=device)
        valid_mask = _make_causal_mask(128, 128, device=device).expand(1, 14, 128, 128)
        k_per_row = torch.full((1, 14, 128), 64, dtype=torch.long, device=device)

        # This should complete without error
        result = vectorized_topk_mask(scores, k_per_row=k_per_row, valid_mask=valid_mask)

        assert result.mask.device.type == "cuda"
        assert result.mask.shape == (1, 14, 128, 128)


class TestVectorizedTopKEdgeCases:
    """Edge case tests."""

    def test_zero_k(self) -> None:
        """k=0 selects nothing (except mandatory)."""
        scores = torch.tensor([[[[5.0, 3.0, 1.0]]]])
        valid_mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
        k_per_row = torch.tensor([[[0]]])

        result = vectorized_topk_mask(scores, k_per_row=k_per_row, valid_mask=valid_mask)

        assert result.mask.sum().item() == 0

    def test_single_valid_tile(self) -> None:
        """Works with only one valid tile."""
        scores = torch.tensor([[[[5.0, 3.0, 1.0]]]])
        valid_mask = torch.tensor([[[[True, False, False]]]])
        k_per_row = torch.tensor([[[2]]])

        result = vectorized_topk_mask(scores, k_per_row=k_per_row, valid_mask=valid_mask)

        assert result.mask[0, 0, 0, 0].item()
        assert result.mask.sum().item() == 1

    def test_batch_heads_dimensions(self) -> None:
        """Works correctly across batch and head dimensions."""
        scores = torch.randn(3, 8, 4, 4)
        valid_mask = _make_causal_mask(4, 4).expand(3, 8, 4, 4)
        k_per_row = torch.full((3, 8, 4), 2, dtype=torch.long)

        result = vectorized_topk_mask(scores, k_per_row=k_per_row, valid_mask=valid_mask)

        assert result.mask.shape == (3, 8, 4, 4)
        # Each row should have exactly 2 selected (or fewer if fewer valid)
        for b in range(3):
            for h in range(8):
                for q in range(4):
                    valid_count = valid_mask[b, h, q].sum().item()
                    expected = min(2, valid_count)
                    assert result.mask[b, h, q].sum().item() == expected
