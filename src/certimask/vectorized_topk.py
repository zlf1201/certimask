"""Vectorized top-k mask construction without Python row loops.

Eliminates per-row .item() CPU-GPU synchronization points by using
batched torch.topk and rank-based masking operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VectorizedTopKMaskResult:
    """Result of vectorized top-k mask construction.

    Attributes:
        mask: Boolean block mask [B, H, Q_blk, K_blk].
        k_per_row: Number of blocks kept per row [B, H, Q_blk].
        topk_indices: Indices of top-k blocks [B, H, Q_blk, max_k].
        topk_scores: Scores of top-k blocks [B, H, Q_blk, max_k].
    """

    mask: torch.Tensor
    k_per_row: torch.Tensor
    topk_indices: torch.Tensor
    topk_scores: torch.Tensor


def vectorized_topk_mask(
    scores: torch.Tensor,
    *,
    k_per_row: torch.Tensor,
    valid_mask: torch.Tensor,
    mandatory_keep_mask: torch.Tensor | None = None,
) -> VectorizedTopKMaskResult:
    """Construct top-k block mask without Python row loops.

    Uses batched torch.topk with rank-based selection to handle
    variable k_per_row across rows.

    Args:
        scores: [B, H, Q_blk, K_blk] block scores.
        k_per_row: [B, H, Q_blk] number of blocks to keep per row.
        valid_mask: [B, H, Q_blk, K_blk] boolean causal valid mask.
        mandatory_keep_mask: Optional [B, H, Q_blk, K_blk] boolean mask
            for blocks that must be kept regardless of score (e.g., local blocks).

    Returns:
        VectorizedTopKMaskResult with mask and metadata.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    if scores.shape != valid_mask.shape:
        raise ValueError(
            f"Shape mismatch: scores {scores.shape} vs valid_mask {valid_mask.shape}"
        )
    if k_per_row.shape != scores.shape[:3]:
        raise ValueError(
            f"Shape mismatch: k_per_row {k_per_row.shape} vs scores {scores.shape[:3]}"
        )

    batch, heads, q_blk, k_blk = scores.shape
    device = scores.device

    # Handle mandatory keep mask
    if mandatory_keep_mask is not None:
        # Count mandatory blocks per row
        mandatory_count = mandatory_keep_mask.long().sum(dim=-1)  # [B, H, Q]
        # Extra budget: how many more blocks to select
        extra_budget = torch.clamp(k_per_row - mandatory_count, min=0)
        # Mask out mandatory and invalid positions for scoring
        available_mask = valid_mask & ~mandatory_keep_mask
        # Zero groups that are already mandatory or invalid
        masked_scores = scores.masked_fill(~available_mask, float("-inf"))
    else:
        mandatory_keep_mask = torch.zeros_like(valid_mask)
        extra_budget = k_per_row
        masked_scores = scores.masked_fill(~valid_mask, float("-inf"))

    # Determine max_k for topk operation
    max_k = int(extra_budget.max().item())

    if max_k == 0:
        # All rows have zero or negative extra budget
        # (all valid blocks are mandatory or k_per_row <= mandatory_count)
        mask = mandatory_keep_mask & valid_mask
        topk_indices = torch.zeros(
            batch, heads, q_blk, 1, dtype=torch.long, device=device,
        )
        topk_scores = torch.full(
            (batch, heads, q_blk, 1), float("-inf"), device=device,
        )
        return VectorizedTopKMaskResult(
            mask=mask,
            k_per_row=k_per_row,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
        )

    # Batched topk: get top max_k indices per row
    # topk_scores shape: [B, H, Q, max_k]
    # topk_indices shape: [B, H, Q, max_k]
    topk_scores, topk_indices = masked_scores.topk(max_k, dim=-1)

    # Build rank tensor: for each position in topk, what's its rank (0-indexed)
    # rank[i] = i for i in 0..max_k-1
    ranks = torch.arange(max_k, device=device).view(1, 1, 1, max_k)
    ranks = ranks.expand(batch, heads, q_blk, max_k)

    # extra_budget shape: [B, H, Q] -> [B, H, Q, 1] for broadcasting
    extra_budget_expanded = extra_budget.unsqueeze(-1)

    # A top-k entry is selected if its rank < extra_budget for that row
    selected_by_rank = ranks < extra_budget_expanded  # [B, H, Q, max_k]

    # Build extra mask using one-hot encoding to avoid scatter conflicts
    # when multiple topk entries map to the same index
    clamped_indices = topk_indices.clamp(0, k_blk - 1)

    # Create one-hot: [B, H, Q, max_k, K_blk]
    one_hot = torch.zeros(
        batch, heads, q_blk, max_k, k_blk, dtype=torch.bool, device=device,
    )
    one_hot.scatter_(4, clamped_indices.unsqueeze(-1), True)

    # Combine: selected entries where rank < budget
    # selected_by_rank: [B, H, Q, max_k] -> [B, H, Q, max_k, 1]
    selected_expanded = selected_by_rank.unsqueeze(-1)
    # OR over max_k dimension: any selected entry contributes
    extra_mask = (one_hot & selected_expanded).any(dim=3)  # [B, H, Q, K_blk]

    # Combine mandatory + extra, masked by valid
    mask = (mandatory_keep_mask | extra_mask) & valid_mask

    return VectorizedTopKMaskResult(
        mask=mask,
        k_per_row=k_per_row,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
    )
