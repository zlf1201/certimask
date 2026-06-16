"""Synthetic data generation for block score experiments."""

from __future__ import annotations

import torch


def generate_synthetic_summaries(
    *,
    batch_size: int,
    num_heads: int,
    num_query_blocks: int,
    num_key_blocks: int,
    head_dim: int,
    distribution: str,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic query and key block summaries.

    Args:
        batch_size: Batch size.
        num_heads: Number of attention heads.
        num_query_blocks: Number of query blocks.
        num_key_blocks: Number of key blocks.
        head_dim: Head dimension.
        distribution: One of 'normal', 'small_variance', 'large_variance',
            'sparse', 'outlier', 'correlated', 'near_zero'.
        seed: Random seed.
        device: Device for tensors.
        dtype: Data type for tensors.

    Returns:
        Tuple of (query, key) tensors, each of shape
        [batch_size, num_heads, num_blocks, head_dim].

    Raises:
        ValueError: If distribution is not supported or parameters are invalid.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if num_query_blocks <= 0:
        raise ValueError(
            f"num_query_blocks must be positive, got {num_query_blocks}"
        )
    if num_key_blocks <= 0:
        raise ValueError(
            f"num_key_blocks must be positive, got {num_key_blocks}"
        )
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")

    gen = torch.Generator(device="cpu").manual_seed(seed)

    q_shape = (batch_size, num_heads, num_query_blocks, head_dim)
    k_shape = (batch_size, num_heads, num_key_blocks, head_dim)

    if distribution == "normal":
        query = torch.randn(q_shape, generator=gen, device="cpu")
        key = torch.randn(k_shape, generator=gen, device="cpu")

    elif distribution == "small_variance":
        query = torch.randn(q_shape, generator=gen, device="cpu") * 0.05
        key = torch.randn(k_shape, generator=gen, device="cpu") * 0.05

    elif distribution == "large_variance":
        query = torch.randn(q_shape, generator=gen, device="cpu") * 5.0
        key = torch.randn(k_shape, generator=gen, device="cpu") * 5.0

    elif distribution == "sparse":
        query = torch.randn(q_shape, generator=gen, device="cpu")
        key = torch.randn(k_shape, generator=gen, device="cpu")
        # ~80% zero
        q_mask = torch.rand(q_shape, generator=gen) > 0.8
        k_mask = torch.rand(k_shape, generator=gen) > 0.8
        query = query * q_mask
        key = key * k_mask

    elif distribution == "outlier":
        query = torch.randn(q_shape, generator=gen, device="cpu")
        key = torch.randn(k_shape, generator=gen, device="cpu")
        # ~5% of elements get multiplied by 10
        q_outlier = torch.rand(q_shape, generator=gen) < 0.05
        k_outlier = torch.rand(k_shape, generator=gen) < 0.05
        query = torch.where(q_outlier, query * 10.0, query)
        key = torch.where(k_outlier, key * 10.0, key)

    elif distribution == "correlated":
        query = torch.randn(q_shape, generator=gen, device="cpu")
        key = torch.randn(k_shape, generator=gen, device="cpu")
        # For matching block indices, key = query + noise
        # Use min(Q, K) blocks for correlation
        n_corr = min(num_query_blocks, num_key_blocks)
        noise = torch.randn(
            (batch_size, num_heads, n_corr, head_dim), generator=gen, device="cpu"
        ) * 0.1
        key[:, :, :n_corr, :] = query[:, :, :n_corr, :] + noise

    elif distribution == "near_zero":
        query = torch.randn(q_shape, generator=gen, device="cpu") * 1e-6
        key = torch.randn(k_shape, generator=gen, device="cpu") * 1e-6

    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. Supported: "
            "normal, small_variance, large_variance, sparse, outlier, "
            "correlated, near_zero"
        )

    return query.to(dtype).to(device), key.to(dtype).to(device)
