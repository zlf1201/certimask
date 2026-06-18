"""Block masking utilities."""

from __future__ import annotations

import torch


def make_block_causal_valid_mask(
    num_query_blocks: int,
    num_key_blocks: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a block-causal valid mask where valid(a, b) = (b <= a).

    Args:
        num_query_blocks: Number of query blocks.
        num_key_blocks: Number of key blocks.
        device: Device for the mask tensor.

    Returns:
        Boolean tensor of shape [1, 1, Q, K], broadcastable to [B, H, Q, K].

    Raises:
        ValueError: If num_query_blocks <= 0 or num_key_blocks <= 0.
    """
    if num_query_blocks <= 0:
        raise ValueError(
            f"num_query_blocks must be positive, got {num_query_blocks}"
        )
    if num_key_blocks <= 0:
        raise ValueError(f"num_key_blocks must be positive, got {num_key_blocks}")

    q_idx = torch.arange(num_query_blocks, device=device).unsqueeze(1)  # [Q, 1]
    k_idx = torch.arange(num_key_blocks, device=device).unsqueeze(0)  # [1, K]
    mask = k_idx <= q_idx  # [Q, K]
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, K]


