"""Tests for block mean pooling."""

from __future__ import annotations

import pytest
import torch

from certimask.block_summary import mean_pool_qk_blocks


class TestMeanPoolBasic:
    """Test basic mean pooling with hand-crafted tensors."""

    def test_block_size_1(self) -> None:
        """block_size=1: each token becomes its own block."""
        q = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])  # [1,1,3,2]
        k = torch.tensor([[[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]])
        result = mean_pool_qk_blocks(q, k, block_size=1)

        assert result.num_blocks == 3
        assert result.dropped_tail_tokens == 0
        torch.testing.assert_close(result.query, q)
        torch.testing.assert_close(result.key, k)

    def test_block_size_2(self) -> None:
        """block_size=2: mean of pairs."""
        q = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]])
        k = torch.tensor([[[[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]]])
        result = mean_pool_qk_blocks(q, k, block_size=2)

        assert result.num_blocks == 2
        assert result.dropped_tail_tokens == 0
        # Block 0: mean([1,2], [3,4]) = [2,3]
        expected_q = torch.tensor([[[[2.0, 3.0], [6.0, 7.0]]]])
        expected_k = torch.tensor([[[[10.0, 11.0], [14.0, 15.0]]]])
        torch.testing.assert_close(result.query, expected_q)
        torch.testing.assert_close(result.key, expected_k)

    def test_block_size_equals_seq_len(self) -> None:
        """block_size=L: entire sequence is one block."""
        q = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        k = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]])
        result = mean_pool_qk_blocks(q, k, block_size=2)

        assert result.num_blocks == 1
        expected_q = torch.tensor([[[[2.0, 3.0]]]])
        expected_k = torch.tensor([[[[6.0, 7.0]]]])
        torch.testing.assert_close(result.query, expected_q)
        torch.testing.assert_close(result.key, expected_k)

    def test_dropped_tail(self) -> None:
        """Tail tokens beyond complete blocks are dropped."""
        q = torch.ones(1, 1, 5, 2)
        k = torch.ones(1, 1, 5, 2) * 2
        result = mean_pool_qk_blocks(q, k, block_size=2)

        assert result.num_blocks == 2
        assert result.used_sequence_length == 4
        assert result.dropped_tail_tokens == 1
        assert result.query.shape == (1, 1, 2, 2)
        assert result.key.shape == (1, 1, 2, 2)

    def test_seq_shorter_than_block_raises(self) -> None:
        q = torch.randn(1, 1, 3, 8)
        k = torch.randn(1, 1, 3, 8)
        with pytest.raises(ValueError, match="block_size"):
            mean_pool_qk_blocks(q, k, block_size=4)

    def test_multiple_heads_and_batch(self) -> None:
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        result = mean_pool_qk_blocks(q, k, block_size=4)

        assert result.num_blocks == 4
        assert result.query.shape == (2, 4, 4, 32)
        assert result.key.shape == (2, 4, 4, 32)

    def test_dtype_preserved(self) -> None:
        q = torch.randn(1, 1, 4, 8, dtype=torch.float64)
        k = torch.randn(1, 1, 4, 8, dtype=torch.float64)
        result = mean_pool_qk_blocks(q, k, block_size=2)
        assert result.query.dtype == torch.float64
        assert result.key.dtype == torch.float64


class TestMeanPoolInvalidInputs:
    """Test input validation."""

    def test_shape_mismatch(self) -> None:
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 6, 8)
        with pytest.raises(ValueError, match="shape"):
            mean_pool_qk_blocks(q, k, block_size=2)

    def test_not_4d(self) -> None:
        q = torch.randn(4, 8)
        k = torch.randn(4, 8)
        with pytest.raises(ValueError, match="4-D"):
            mean_pool_qk_blocks(q, k, block_size=2)

    def test_zero_block_size(self) -> None:
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        with pytest.raises(ValueError, match="positive"):
            mean_pool_qk_blocks(q, k, block_size=0)

    def test_non_floating(self) -> None:
        q = torch.randint(0, 10, (1, 1, 4, 8))
        k = torch.randn(1, 1, 4, 8)
        with pytest.raises(TypeError, match="floating-point"):
            mean_pool_qk_blocks(q, k, block_size=2)
