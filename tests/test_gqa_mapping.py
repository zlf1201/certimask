"""Tests for GQA head expansion."""

from __future__ import annotations

import pytest
import torch

from certimask.block_summary import expand_kv_heads


class TestExpandKVHeads:
    """Test GQA head expansion."""

    def test_basic_expansion(self) -> None:
        """2 KV heads -> 8 query heads: each KV head repeated 4 times."""
        kv = torch.arange(16, dtype=torch.float32).reshape(1, 2, 4, 2)
        # KV head 0: [[0,1],[2,3],[4,5],[6,7]]
        # KV head 1: [[8,9],[10,11],[12,13],[14,15]]
        expanded = expand_kv_heads(kv, num_query_heads=8)
        assert expanded.shape == (1, 8, 4, 2)

        # Query heads 0-3 should all equal KV head 0
        for h in range(4):
            assert torch.equal(expanded[:, h], kv[:, 0])

        # Query heads 4-7 should all equal KV head 1
        for h in range(4, 8):
            assert torch.equal(expanded[:, h], kv[:, 1])

    def test_no_expansion_needed(self) -> None:
        """H_q == H_kv: return unchanged."""
        kv = torch.randn(1, 4, 8, 16)
        result = expand_kv_heads(kv, num_query_heads=4)
        assert result.shape == (1, 4, 8, 16)
        assert torch.equal(result, kv)

    def test_batch_and_seq(self) -> None:
        """Expansion preserves batch and sequence dimensions."""
        kv = torch.randn(3, 2, 10, 32)
        result = expand_kv_heads(kv, num_query_heads=6)
        assert result.shape == (3, 6, 10, 32)

    def test_values_preserved(self) -> None:
        """Expanded values are exact copies, not averaged."""
        kv = torch.randn(1, 2, 5, 8)
        result = expand_kv_heads(kv, num_query_heads=4)
        # Head 0 and 1 in result should both be kv[:, 0]
        assert torch.equal(result[:, 0], result[:, 1])
        assert torch.equal(result[:, 0], kv[:, 0])

    def test_invalid_dim(self) -> None:
        kv = torch.randn(4, 8, 16)
        with pytest.raises(ValueError, match="4-D"):
            expand_kv_heads(kv, num_query_heads=8)

    def test_empty_tensor(self) -> None:
        kv = torch.empty(1, 2, 0, 16)
        with pytest.raises(ValueError, match="empty"):
            expand_kv_heads(kv, num_query_heads=4)

    def test_not_divisible(self) -> None:
        kv = torch.randn(1, 3, 8, 16)
        with pytest.raises(ValueError, match="divisible"):
            expand_kv_heads(kv, num_query_heads=8)
