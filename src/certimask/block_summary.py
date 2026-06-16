"""Block summarization and GQA head expansion for Q/K tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlockSummaries:
    """Result of block-pooling Q/K tensors.

    Attributes:
        query: Mean-pooled query, shape [B, H, num_blocks, D].
        key: Mean-pooled key, shape [B, H, num_blocks, D].
        num_blocks: Number of complete blocks.
        block_size: Block size used.
        used_sequence_length: Number of tokens used (excluding dropped tail).
        dropped_tail_tokens: Number of tail tokens dropped.
    """

    query: torch.Tensor
    key: torch.Tensor
    num_blocks: int
    block_size: int
    used_sequence_length: int
    dropped_tail_tokens: int


def expand_kv_heads(
    tensor: torch.Tensor,
    num_query_heads: int,
) -> torch.Tensor:
    """Expand KV heads to match query heads for grouped-query attention.

    Maps each KV head to `group_size = num_query_heads // num_kv_heads`
    consecutive query heads.

    Args:
        tensor: KV tensor of shape [B, H_kv, L, D].
        num_query_heads: Number of query heads.

    Returns:
        Tensor of shape [B, H_q, L, D] with KV heads expanded.

    Raises:
        ValueError: If tensor is not 4-D, empty, or H_q is not divisible by H_kv.
    """
    if tensor.dim() != 4:
        raise ValueError(f"tensor must be 4-D [B, H_kv, L, D], got {tensor.dim()}-D")

    if tensor.numel() == 0:
        raise ValueError("tensor is empty")

    num_kv_heads = tensor.shape[1]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads ({num_query_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )

    if num_query_heads == num_kv_heads:
        return tensor

    group_size = num_query_heads // num_kv_heads
    # repeat_interleave along dim=1: each KV head repeated group_size times
    return tensor.repeat_interleave(group_size, dim=1)


def mean_pool_qk_blocks(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    block_size: int,
    drop_incomplete_block: bool = True,
) -> BlockSummaries:
    """Mean-pool Q/K tensors into blocks along the sequence dimension.

    Args:
        query: Query tensor, shape [B, H, L, D].
        key: Key tensor, shape [B, H, L, D].
        block_size: Number of tokens per block.
        drop_incomplete_block: If True, drop the last incomplete block.
            If False, the incomplete block is excluded (raises if L < block_size).

    Returns:
        BlockSummaries with mean-pooled Q/K and metadata.

    Raises:
        ValueError: If inputs are invalid or fewer than block_size tokens.
    """
    _validate_pool_inputs(query, key, block_size)

    seq_len = query.shape[2]
    num_complete_blocks = seq_len // block_size

    if num_complete_blocks == 0:
        raise ValueError(
            f"Sequence length {seq_len} < block_size {block_size}. "
            "At least one complete block is required."
        )

    used_length = num_complete_blocks * block_size
    dropped_tail = seq_len - used_length

    # Truncate to complete blocks
    q_truncated = query[:, :, :used_length, :]
    k_truncated = key[:, :, :used_length, :]

    # Reshape to [batch, heads, num_blocks, block_size, dim] and mean
    batch, heads, _, dim = q_truncated.shape
    q_blocks = q_truncated.reshape(batch, heads, num_complete_blocks, block_size, dim)
    k_blocks = k_truncated.reshape(batch, heads, num_complete_blocks, block_size, dim)

    q_pooled = q_blocks.mean(dim=3)  # [B, H, num_blocks, D]
    k_pooled = k_blocks.mean(dim=3)  # [B, H, num_blocks, D]

    return BlockSummaries(
        query=q_pooled,
        key=k_pooled,
        num_blocks=num_complete_blocks,
        block_size=block_size,
        used_sequence_length=used_length,
        dropped_tail_tokens=dropped_tail,
    )


def _validate_pool_inputs(
    query: torch.Tensor, key: torch.Tensor, block_size: int
) -> None:
    """Validate inputs for block pooling.

    Raises:
        TypeError: If inputs are not floating-point.
        ValueError: If shapes are incompatible or parameters are invalid.
    """
    if not query.is_floating_point():
        raise TypeError(f"query must be floating-point, got {query.dtype}")
    if not key.is_floating_point():
        raise TypeError(f"key must be floating-point, got {key.dtype}")

    if query.dim() != 4:
        raise ValueError(f"query must be 4-D [B, H, L, D], got {query.dim()}-D")
    if key.dim() != 4:
        raise ValueError(f"key must be 4-D [B, H, L, D], got {key.dim()}-D")

    if query.shape != key.shape:
        raise ValueError(
            f"query shape {query.shape} != key shape {key.shape}"
        )

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    if query.numel() == 0:
        raise ValueError("query tensor is empty")
