"""AGLR-C v0: Anchor-Guided Local-Retrieval Certifiable Indexer.

Components:
1. Mandatory local blocks
2. Landmark global block scoring
3. Adaptive per-row budget
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AGLRMaskResult:
    """Result of AGLR mask selection.

    Attributes:
        mask: Boolean block mask [B, H, Q_blk, K_blk].
        scores: Landmark scores [B, H, Q_blk, K_blk].
        local_mask: Boolean mask for mandatory local blocks.
        selected_extra_mask: Boolean mask for extra selected blocks.
        actual_tile_sparsity: Fraction of valid tiles dropped.
        attention_tile_work_fraction: Fraction of valid tiles kept.
        local_budget_overflow_rate: Fraction of rows where local exceeds budget.
        selected_blocks_per_row_mean: Mean number of selected blocks per row.
    """

    mask: torch.Tensor
    scores: torch.Tensor
    local_mask: torch.Tensor
    selected_extra_mask: torch.Tensor
    actual_tile_sparsity: float
    attention_tile_work_fraction: float
    local_budget_overflow_rate: float
    selected_blocks_per_row_mean: float


def aglr_local_plus_landmark_mask(
    landmark_scores: torch.Tensor,
    *,
    target_sparsity: float | None = None,
    target_keep_fraction: float | None = None,
    local_blocks: int,
    valid_mask: torch.Tensor,
    per_query_budget: bool = True,
) -> AGLRMaskResult:
    """Create mask with mandatory local blocks + landmark-scored extras.

    .. warning::
        **Historical/reference mask construction.** This function uses a
        Python triple-nested loop over (batch, heads, q_blocks) and is
        slow for large tensors. Use :func:`certimask.vectorized_topk.vectorized_topk_mask`
        for optimized top-k mask construction.

    Args:
        landmark_scores: [B, H, Q_blk, K_blk] block scores.
        target_sparsity: Fraction of valid tiles to drop (mutually exclusive
            with target_keep_fraction).
        target_keep_fraction: Fraction of valid tiles to keep.
        local_blocks: Number of mandatory local blocks per query row.
        valid_mask: [B, H, Q_blk, K_blk] boolean causal valid mask.
        per_query_budget: If True, budget is per query row.

    Returns:
        AGLRMaskResult.
    """
    if target_sparsity is not None and target_keep_fraction is not None:
        raise ValueError("Specify only one of target_sparsity, target_keep_fraction")
    if target_sparsity is None and target_keep_fraction is None:
        raise ValueError("Must specify target_sparsity or target_keep_fraction")

    if target_keep_fraction is None:
        target_keep_fraction = 1.0 - target_sparsity  # type: ignore[operator]

    batch, heads, q_blk, k_blk = landmark_scores.shape
    device = landmark_scores.device

    # Per-row budget
    valid_per_row = valid_mask.sum(dim=-1)  # [B, H, Q]
    if per_query_budget:
        keep_per_row = (valid_per_row.float() * target_keep_fraction).ceil().long()
        keep_per_row = torch.clamp(keep_per_row, min=1)
    else:
        # Global budget per head
        total_valid = valid_mask.sum(dim=(-2, -1)).float()  # [B, H]
        total_keep = (total_valid * target_keep_fraction).ceil().long()
        keep_per_row = total_keep.unsqueeze(-1).expand_as(valid_per_row)

    local_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    extra_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    overflow_count = 0

    for b in range(batch):
        for h in range(heads):
            for q in range(q_blk):
                n_keep = int(keep_per_row[b, h, q].item())
                valid_k = valid_mask[b, h, q]

                # Step 1: mandatory local blocks
                local_start = max(0, q - local_blocks + 1)
                local_range = torch.arange(k_blk, device=device)
                is_local = (local_range >= local_start) & (local_range <= q) & valid_k
                local_mask[b, h, q, is_local] = True
                n_local = int(is_local.sum().item())

                if n_local >= n_keep:
                    overflow_count += 1
                    continue

                # Step 2: fill remaining with highest landmark scores
                remaining = n_keep - n_local
                scores_row = landmark_scores[b, h, q].clone()
                scores_row[is_local] = torch.finfo(torch.float32).min
                scores_row[~valid_k] = torch.finfo(torch.float32).min

                n_available = int((scores_row > torch.finfo(torch.float32).min).sum().item())
                n_extra = min(remaining, n_available)

                if n_extra > 0:
                    _, extra_idx = scores_row.topk(n_extra)
                    extra_mask[b, h, q, extra_idx] = True

    mask = local_mask | extra_mask

    # Compute stats
    valid_count = int(valid_mask.sum().item())
    kept_count = int((mask & valid_mask).sum().item())
    total_rows = batch * heads * q_blk

    return AGLRMaskResult(
        mask=mask,
        scores=landmark_scores,
        local_mask=local_mask,
        selected_extra_mask=extra_mask,
        actual_tile_sparsity=1.0 - kept_count / valid_count if valid_count > 0 else 0.0,
        attention_tile_work_fraction=kept_count / valid_count if valid_count > 0 else 0.0,
        local_budget_overflow_rate=overflow_count / total_rows if total_rows > 0 else 0.0,
        selected_blocks_per_row_mean=kept_count / total_rows if total_rows > 0 else 0.0,
    )


def compute_antidiagonal_block_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    block_size: int,
    sample_pattern: str,
    aggregation: str,
    num_samples: int | None = None,
    valid_mask: torch.Tensor | None = None,
    scale_by_sqrt_dim: bool = True,
) -> torch.Tensor:
    """Compute block scores using sampled token interactions within tiles.

    Args:
        query: [B, H, L, D] query tensor.
        key: [B, H, L, D] key tensor.
        block_size: Tokens per block.
        sample_pattern: One of 'main_diagonal', 'anti_diagonal',
            'both_diagonals', 'strided_grid', 'landmark_cross'.
        aggregation: One of 'mean', 'max', 'topk_mean', 'logsumexp'.
        num_samples: Number of samples for strided_grid.
        valid_mask: Optional [B, H, Q_blk, K_blk] boolean mask.
        scale_by_sqrt_dim: Whether to scale by 1/sqrt(d).

    Returns:
        scores: [B, H, Q_blk, K_blk] block scores.
    """
    if not query.is_floating_point():
        raise TypeError(f"query must be floating-point, got {query.dtype}")
    if not key.is_floating_point():
        raise TypeError(f"key must be floating-point, got {key.dtype}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    batch, heads, seq_len, dim = query.shape

    num_blocks = seq_len // block_size
    if num_blocks == 0:
        raise ValueError(f"seq_len {seq_len} < block_size {block_size}")

    used_len = num_blocks * block_size
    query = query[:, :, :used_len, :]
    key = key[:, :, :used_len, :]

    # Reshape to [B, H, num_blocks, block_size, D]
    q_blocks = query.reshape(batch, heads, num_blocks, block_size, dim)
    k_blocks = key.reshape(batch, heads, num_blocks, block_size, dim)

    # Generate sample positions within a tile
    positions = _generate_sample_positions(block_size, sample_pattern, num_samples)
    q_indices = torch.tensor([p[0] for p in positions], device=query.device)
    k_indices = torch.tensor([p[1] for p in positions], device=query.device)

    # Gather query and key at sampled positions
    q_sampled = q_blocks[:, :, :, q_indices, :]  # [B, H, Q, P, D]
    k_sampled = k_blocks[:, :, :, k_indices, :]  # [B, H, K, P, D]

    # Paired dot products: [B, H, Q, K, P]
    dots = torch.einsum("bhqpd,bhkpd->bhqkp", q_sampled, k_sampled)

    if scale_by_sqrt_dim:
        sqrt_d = torch.sqrt(torch.tensor(float(dim), dtype=torch.float32))
        dots = dots / sqrt_d

    # Aggregate over sampled positions
    if aggregation == "mean":
        scores = dots.mean(dim=-1)
    elif aggregation == "max":
        scores = dots.max(dim=-1).values
    elif aggregation == "topk_mean":
        topk = min(4, dots.shape[-1])
        topk_vals, _ = dots.topk(topk, dim=-1)
        scores = topk_vals.mean(dim=-1)
    elif aggregation == "logsumexp":
        max_val = dots.max(dim=-1).values
        scores = (
            torch.logsumexp(dots - max_val.unsqueeze(-1), dim=-1) + max_val
        )
    else:
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. "
            "Supported: mean, max, topk_mean, logsumexp"
        )

    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, torch.finfo(torch.float32).min)

    return scores


def _generate_sample_positions(
    block_size: int,
    pattern: str,
    num_samples: int | None = None,
) -> list[tuple[int, int]]:
    """Generate (query_pos, key_pos) sample positions within a tile."""
    positions: list[tuple[int, int]] = []

    if pattern == "main_diagonal":
        for t in range(block_size):
            positions.append((t, t))

    elif pattern == "anti_diagonal":
        for t in range(block_size):
            positions.append((t, block_size - 1 - t))

    elif pattern == "both_diagonals":
        for t in range(block_size):
            positions.append((t, t))
            if t != block_size - 1 - t:
                positions.append((t, block_size - 1 - t))

    elif pattern == "strided_grid":
        n = num_samples if num_samples else 4
        step = max(1, block_size // n)
        for i in range(0, block_size, step):
            if len(positions) >= n:
                break
            positions.append((i, i))
        for i in range(0, block_size, step):
            if len(positions) >= n * 2:
                break
            j = block_size - 1 - i
            if (i, j) not in positions:
                positions.append((i, j))

    elif pattern == "landmark_cross":
        n = min(4, block_size)
        step = max(1, block_size // n)
        lm_indices = list(range(0, block_size, step))[:n]
        for qi in lm_indices:
            for ki in lm_indices:
                positions.append((qi, ki))
    else:
        raise ValueError(
            f"Unknown sample_pattern '{pattern}'. Supported: "
            "main_diagonal, anti_diagonal, both_diagonals, strided_grid, "
            "landmark_cross"
        )

    return positions


def combine_aglr_scores(
    *,
    landmark_scores: torch.Tensor | None = None,
    antidiagonal_scores: torch.Tensor | None = None,
    recency_weight: float = 0.0,
    landmark_weight: float = 0.5,
    antidiagonal_weight: float = 0.5,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine landmark and antidiagonal scores with optional recency prior.

    Args:
        landmark_scores: [B, H, Q, K] landmark-based scores.
        antidiagonal_scores: [B, H, Q, K] antidiagonal-based scores.
        recency_weight: Weight for recency prior.
        landmark_weight: Weight for landmark scores.
        antidiagonal_weight: Weight for antidiagonal scores.
        valid_mask: [B, H, Q, K] boolean mask.

    Returns:
        Combined scores [B, H, Q, K].
    """
    if landmark_scores is None and antidiagonal_scores is None:
        raise ValueError("At least one score source required")

    if landmark_scores is not None:
        shape = landmark_scores.shape
        device = landmark_scores.device
    elif antidiagonal_scores is not None:
        shape = antidiagonal_scores.shape
        device = antidiagonal_scores.device
    else:
        raise ValueError("At least one score source required")

    combined = torch.zeros(shape, dtype=torch.float32, device=device)

    if landmark_scores is not None:
        combined = combined + landmark_weight * landmark_scores
    if antidiagonal_scores is not None:
        combined = combined + antidiagonal_weight * antidiagonal_scores

    if recency_weight > 0:
        q_blk, k_blk = shape[2], shape[3]
        q_idx = torch.arange(q_blk, device=device).float()
        k_idx = torch.arange(k_blk, device=device).float()
        recency = -torch.log1p(
            (q_idx.unsqueeze(1) - k_idx.unsqueeze(0)).clamp(min=0)
        )
        recency = recency.unsqueeze(0).unsqueeze(0)
        combined = combined + recency_weight * recency

    if valid_mask is not None:
        combined = combined.masked_fill(~valid_mask, torch.finfo(torch.float32).min)

    return combined
