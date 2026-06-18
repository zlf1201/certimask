"""Candidate-pruned AGLR-C v2: cheap candidate generation + candidate-only scoring.

Generates a candidate mask that selects a subset of block pairs for expensive
AGLR sampled scoring, instead of scoring all O(N²) causal block pairs.

Candidate generation modes:
- local_stride: local window + first block + strided historical + diagonal band
- block_norm: norm-product proxy (baseline, uses full-pair proxy)
- coarse_to_fine: two-level hierarchy (primary mode)
- head_pattern: fixed per-head routing
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CandidateMaskResult:
    """Result of candidate mask generation.

    Attributes:
        candidate_mask: Boolean block mask [B, H, Q_blk, K_blk].
            True = this tile is a candidate for AGLR scoring.
        candidate_fraction: Fraction of valid tiles marked as candidates.
        valid_fraction: Fraction of all tiles that are causally valid.
        mode: Candidate generation mode used.
        metadata: Additional metadata about the generation process.
    """

    candidate_mask: torch.Tensor
    candidate_fraction: float
    valid_fraction: float
    mode: str
    metadata: dict[str, float | int | str | bool] = field(default_factory=dict)


def generate_candidate_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    mode: str,
    block_size: int = 8,
    valid_mask: torch.Tensor | None = None,
    target_candidate_fraction: float = 0.25,
    local_blocks: int = 4,
    stride: int = 16,
    coarse_block_size: int = 64,
    topk_coarse: int = 8,
    diagonal_band: int = 8,
) -> CandidateMaskResult:
    """Generate a candidate block mask for selective AGLR scoring.

    Args:
        query: [B, H, L, D] query tensor.
        key: [B, H, L, D] key tensor.
        mode: Candidate generation mode. One of 'local_stride',
            'block_norm', 'coarse_to_fine', 'head_pattern'.
        block_size: Fine block size (tokens per block).
        valid_mask: Optional [B, H, Q_blk, K_blk] boolean causal mask.
            If None, a standard causal mask is created.
        target_candidate_fraction: Target fraction of valid tiles to
            select as candidates (used by block_norm and coarse_to_fine).
        local_blocks: Number of mandatory local blocks per query row.
        stride: Stride for strided-historical candidates (local_stride mode).
        coarse_block_size: Coarse block size in tokens (coarse_to_fine mode).
        topk_coarse: Number of top coarse regions to select (coarse_to_fine mode).
        diagonal_band: Width of diagonal band in blocks (local_stride mode).

    Returns:
        CandidateMaskResult with candidate mask and metadata.

    Raises:
        ValueError: If mode is unknown or parameters are invalid.
    """
    if mode not in ("local_stride", "block_norm", "coarse_to_fine", "head_pattern"):
        raise ValueError(
            f"Unknown mode '{mode}'. Supported: local_stride, block_norm, "
            "coarse_to_fine, head_pattern"
        )

    batch, heads, seq_len, dim = query.shape
    num_blocks = seq_len // block_size
    if num_blocks == 0:
        raise ValueError(f"seq_len {seq_len} < block_size {block_size}")

    device = query.device

    # Build valid mask if not provided
    if valid_mask is None:
        q_idx = torch.arange(num_blocks, device=device).unsqueeze(1)
        k_idx = torch.arange(num_blocks, device=device).unsqueeze(0)
        causal = k_idx <= q_idx
        valid_mask = causal.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1)

    valid_count = int(valid_mask.sum().item())
    total_count = valid_mask.numel()
    valid_fraction = valid_count / total_count if total_count > 0 else 0.0

    # Dispatch to mode-specific generator
    if mode == "local_stride":
        candidate_mask, meta = _generate_local_stride(
            batch, heads, num_blocks, device,
            valid_mask=valid_mask,
            local_blocks=local_blocks,
            stride=stride,
            diagonal_band=diagonal_band,
        )
    elif mode == "block_norm":
        candidate_mask, meta = _generate_block_norm(
            query, key,
            block_size=block_size,
            valid_mask=valid_mask,
            target_candidate_fraction=target_candidate_fraction,
        )
    elif mode == "coarse_to_fine":
        candidate_mask, meta = _generate_coarse_to_fine(
            query, key,
            block_size=block_size,
            valid_mask=valid_mask,
            coarse_block_size=coarse_block_size,
            topk_coarse=topk_coarse,
            local_blocks=local_blocks,
        )
    elif mode == "head_pattern":
        candidate_mask, meta = _generate_head_pattern(
            query, key,
            block_size=block_size,
            valid_mask=valid_mask,
            local_blocks=local_blocks,
            stride=stride,
            coarse_block_size=coarse_block_size,
            topk_coarse=topk_coarse,
            target_candidate_fraction=target_candidate_fraction,
            diagonal_band=diagonal_band,
        )
    else:
        raise ValueError(f"Unhandled mode '{mode}'")

    # Ensure candidates are a subset of valid tiles
    candidate_mask = candidate_mask & valid_mask

    candidate_count = int(candidate_mask.sum().item())
    candidate_fraction = candidate_count / valid_count if valid_count > 0 else 0.0

    meta["candidate_tiles"] = candidate_count
    meta["valid_tiles"] = valid_count

    return CandidateMaskResult(
        candidate_mask=candidate_mask,
        candidate_fraction=candidate_fraction,
        valid_fraction=valid_fraction,
        mode=mode,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Mode: local_stride
# ---------------------------------------------------------------------------

def _generate_local_stride(
    batch: int,
    heads: int,
    num_blocks: int,
    device: torch.device,
    *,
    valid_mask: torch.Tensor,
    local_blocks: int,
    stride: int,
    diagonal_band: int = 8,
) -> tuple[torch.Tensor, dict[str, float | int | str | bool]]:
    """Local window + first block + strided historical + diagonal band."""
    q_idx = torch.arange(num_blocks, device=device).unsqueeze(1)  # [Q, 1]
    k_idx = torch.arange(num_blocks, device=device).unsqueeze(0)  # [1, K]

    # Local window: for each q, keep blocks [q - local_blocks + 1, q]
    local_start = torch.clamp(q_idx - local_blocks + 1, min=0)
    is_local = (k_idx >= local_start) & (k_idx <= q_idx)

    # First/global block: always include block 0
    is_first = k_idx == 0

    # Strided historical: every stride-th block
    is_strided = (k_idx % stride == 0) & (k_idx <= q_idx)

    # Diagonal band: blocks within diagonal_band of the main diagonal
    is_diagonal = (q_idx - k_idx).abs() <= diagonal_band

    # Combine
    combined = is_local | is_first | is_strided | is_diagonal
    mask = combined.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1)

    meta: dict[str, float | int | str | bool] = {
        "uses_full_pair_scoring": False,
        "uses_full_pair_proxy": False,
        "local_blocks": local_blocks,
        "stride": stride,
        "diagonal_band": diagonal_band,
    }
    return mask, meta


# ---------------------------------------------------------------------------
# Mode: block_norm
# ---------------------------------------------------------------------------

def _generate_block_norm(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    block_size: int,
    valid_mask: torch.Tensor,
    target_candidate_fraction: float,
) -> tuple[torch.Tensor, dict[str, float | int | str | bool]]:
    """Norm-product proxy: score = ||q_block|| * ||k_block||.

    Uses vectorized top-k selection (no Python loops).
    """
    batch, heads, seq_len, dim = query.shape
    num_blocks = seq_len // block_size
    device = query.device

    used_len = num_blocks * block_size
    q_blocks = query[:, :, :used_len, :].reshape(
        batch, heads, num_blocks, block_size, dim,
    )
    k_blocks = key[:, :, :used_len, :].reshape(
        batch, heads, num_blocks, block_size, dim,
    )

    # Per-block L2 norms: [B, H, num_blocks]
    q_norms = q_blocks.float().norm(dim=(-2, -1))
    k_norms = k_blocks.float().norm(dim=(-2, -1))

    # Pairwise norm product: [B, H, Q_blk, K_blk]
    proxy_scores = q_norms.unsqueeze(-1) * k_norms.unsqueeze(-2)

    # Mask invalid tiles
    proxy_scores = proxy_scores.masked_fill(~valid_mask, float("-inf"))

    # Vectorized top-k per row
    valid_per_row = valid_mask.sum(dim=-1)  # [B, H, Q]
    k_per_row = (valid_per_row.float() * target_candidate_fraction).ceil().long()
    k_per_row = torch.clamp(k_per_row, min=1)

    max_k = int(k_per_row.max().item())
    max_k = min(max_k, num_blocks)

    if max_k == 0:
        mask = torch.zeros(
            batch, heads, num_blocks, num_blocks, dtype=torch.bool, device=device,
        )
    else:
        topk_scores, topk_idx = proxy_scores.topk(max_k, dim=-1)
        ranks = torch.arange(max_k, device=device).view(1, 1, 1, max_k)
        k_expanded = k_per_row.unsqueeze(-1)
        selected_by_rank = ranks < k_expanded

        clamped_idx = topk_idx.clamp(0, num_blocks - 1)
        one_hot = torch.zeros(
            batch, heads, num_blocks, max_k, num_blocks,
            dtype=torch.bool, device=device,
        )
        one_hot.scatter_(4, clamped_idx.unsqueeze(-1), True)
        selected_expanded = selected_by_rank.unsqueeze(-1)
        mask = (one_hot & selected_expanded).any(dim=3)

    meta: dict[str, float | int | str | bool] = {
        "uses_full_pair_scoring": False,
        "uses_full_pair_proxy": True,
        "target_candidate_fraction": target_candidate_fraction,
    }
    return mask, meta


# ---------------------------------------------------------------------------
# Mode: coarse_to_fine
# ---------------------------------------------------------------------------

def _generate_coarse_to_fine(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    block_size: int,
    valid_mask: torch.Tensor,
    coarse_block_size: int,
    topk_coarse: int,
    local_blocks: int,
) -> tuple[torch.Tensor, dict[str, float | int | str | bool]]:
    """Two-level hierarchy: coarse selection then fine expansion.

    Uses vectorized operations throughout (no Python loops for mask construction).
    """
    batch, heads, seq_len, dim = query.shape
    num_blocks = seq_len // block_size
    device = query.device

    # Coarse blocks contain multiple fine blocks
    fine_per_coarse = max(1, coarse_block_size // block_size)
    num_coarse = (num_blocks + fine_per_coarse - 1) // fine_per_coarse

    used_len = num_blocks * block_size
    q_blocks = query[:, :, :used_len, :].reshape(
        batch, heads, num_blocks, block_size, dim,
    )
    k_blocks = key[:, :, :used_len, :].reshape(
        batch, heads, num_blocks, block_size, dim,
    )

    # Per-fine-block L2 norms
    q_norms = q_blocks.float().norm(dim=(-2, -1))  # [B, H, num_blocks]
    k_norms = k_blocks.float().norm(dim=(-2, -1))

    # Pad to multiple of fine_per_coarse
    pad_len = num_coarse * fine_per_coarse - num_blocks
    if pad_len > 0:
        q_norms_pad = torch.nn.functional.pad(q_norms, (0, pad_len), value=0.0)
        k_norms_pad = torch.nn.functional.pad(k_norms, (0, pad_len), value=0.0)
    else:
        q_norms_pad = q_norms
        k_norms_pad = k_norms

    # Aggregate to coarse level
    q_coarse = q_norms_pad.reshape(
        batch, heads, num_coarse, fine_per_coarse,
    ).sum(dim=-1)  # [B, H, num_coarse]
    k_coarse = k_norms_pad.reshape(
        batch, heads, num_coarse, fine_per_coarse,
    ).sum(dim=-1)

    # Coarse proxy scores: [B, H, num_coarse, num_coarse]
    coarse_scores = q_coarse.unsqueeze(-1) * k_coarse.unsqueeze(-2)

    # Coarse causal mask
    cq_idx = torch.arange(num_coarse, device=device).unsqueeze(1)
    ck_idx = torch.arange(num_coarse, device=device).unsqueeze(0)
    coarse_causal = (ck_idx <= cq_idx).unsqueeze(0).unsqueeze(0)

    # Coarse valid: causal only (simplification — fine valid mask is applied later)
    coarse_valid = coarse_causal.expand(batch, heads, -1, -1)
    coarse_scores = coarse_scores.masked_fill(~coarse_valid, float("-inf"))

    # Vectorized top-k coarse selection
    coarse_valid_per_row = coarse_valid.sum(dim=-1)  # [B, H, num_coarse]
    ck_per_row = torch.clamp(
        torch.full_like(coarse_valid_per_row, topk_coarse), max=coarse_valid_per_row,
    )
    max_ck = min(topk_coarse, num_coarse)

    coarse_topk_scores, coarse_topk_idx = coarse_scores.topk(max_ck, dim=-1)
    ck_ranks = torch.arange(max_ck, device=device).view(1, 1, 1, max_ck)
    ck_selected = ck_ranks < ck_per_row.unsqueeze(-1)

    coarse_mask = torch.zeros(
        batch, heads, num_coarse, num_coarse, dtype=torch.bool, device=device,
    )
    ck_clamped = coarse_topk_idx.clamp(0, num_coarse - 1)
    ck_one_hot = torch.zeros(
        batch, heads, num_coarse, max_ck, num_coarse,
        dtype=torch.bool, device=device,
    )
    ck_one_hot.scatter_(4, ck_clamped.unsqueeze(-1), True)
    coarse_mask = (ck_one_hot & ck_selected.unsqueeze(-1)).any(dim=3)

    # Add local coarse blocks
    local_coarse = max(1, local_blocks // fine_per_coarse + 1)
    lc_start = torch.clamp(cq_idx - local_coarse + 1, min=0)
    is_local_coarse = (ck_idx >= lc_start) & (ck_idx <= cq_idx)
    coarse_mask = coarse_mask | is_local_coarse.unsqueeze(0).unsqueeze(0)

    # Expand coarse mask to fine mask using repeat_interleave
    # coarse_mask: [B, H, num_coarse, num_coarse]
    # -> repeat_interleave along both coarse dims
    fine_mask = coarse_mask.repeat_interleave(fine_per_coarse, dim=2)
    fine_mask = fine_mask.repeat_interleave(fine_per_coarse, dim=3)

    # Trim to actual num_blocks
    fine_mask = fine_mask[:, :, :num_blocks, :num_blocks]

    # Apply fine valid mask (important: coarse selection was causal-only)
    fine_mask = fine_mask & valid_mask

    meta: dict[str, float | int | str | bool] = {
        "uses_full_pair_scoring": False,
        "uses_full_pair_proxy": False,
        "coarse_block_size": coarse_block_size,
        "fine_per_coarse": fine_per_coarse,
        "num_coarse": num_coarse,
        "topk_coarse": topk_coarse,
    }
    return fine_mask, meta


# ---------------------------------------------------------------------------
# Mode: head_pattern
# ---------------------------------------------------------------------------

def _generate_head_pattern(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    block_size: int,
    valid_mask: torch.Tensor,
    local_blocks: int,
    stride: int,
    coarse_block_size: int,
    topk_coarse: int,
    target_candidate_fraction: float,
    diagonal_band: int = 8,
) -> tuple[torch.Tensor, dict[str, float | int | str | bool]]:
    """Fixed per-head routing: hash head index to sub-mode."""
    batch, heads, seq_len, dim = query.shape
    num_blocks = seq_len // block_size
    device = query.device

    mask = torch.zeros(
        batch, heads, num_blocks, num_blocks, dtype=torch.bool, device=device,
    )

    # Route heads: 40% local_stride, 40% coarse_to_fine, 20% block_norm
    for h in range(heads):
        route = h % 5  # 0,1,2,3,4
        if route < 2:
            sub_mask, _ = _generate_local_stride(
                batch, 1, num_blocks, device,
                valid_mask=valid_mask[:, h:h + 1],
                local_blocks=local_blocks,
                stride=stride,
                diagonal_band=diagonal_band,
            )
            mask[:, h:h + 1] = sub_mask
        elif route < 4:
            sub_mask, _ = _generate_coarse_to_fine(
                query[:, h:h + 1], key[:, h:h + 1],
                block_size=block_size,
                valid_mask=valid_mask[:, h:h + 1],
                coarse_block_size=coarse_block_size,
                topk_coarse=topk_coarse,
                local_blocks=local_blocks,
            )
            mask[:, h:h + 1] = sub_mask
        else:
            sub_mask, _ = _generate_block_norm(
                query[:, h:h + 1], key[:, h:h + 1],
                block_size=block_size,
                valid_mask=valid_mask[:, h:h + 1],
                target_candidate_fraction=target_candidate_fraction,
            )
            mask[:, h:h + 1] = sub_mask

    meta: dict[str, float | int | str | bool] = {
        "uses_full_pair_scoring": False,
        "uses_full_pair_proxy": True,
        "route_local_stride": 2,
        "route_coarse_to_fine": 2,
        "route_block_norm": 1,
    }
    return mask, meta


# ---------------------------------------------------------------------------
# Candidate-only AGLR scoring
# ---------------------------------------------------------------------------

def compute_candidate_antidiagonal_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    block_size: int = 8,
    sample_pattern: str = "both_diagonals",
    aggregation: str = "logsumexp",
    scale_by_sqrt_dim: bool = True,
) -> tuple[torch.Tensor, dict[str, float | int | str | bool]]:
    """Compute AGLR antidiagonal scores only for candidate block pairs.

    Non-candidate valid tiles receive -inf. Invalid/future tiles receive -inf.

    Args:
        query: [B, H, L, D] query tensor.
        key: [B, H, L, D] key tensor.
        candidate_mask: [B, H, Q_blk, K_blk] boolean candidate mask.
        block_size: Tokens per block.
        sample_pattern: Sample pattern for antidiagonal scoring.
        aggregation: Score aggregation method.
        scale_by_sqrt_dim: Whether to scale by 1/sqrt(d).

    Returns:
        scores: [B, H, Q_blk, K_blk] block scores.
        metadata: Dict with computed_tile_count, uses_full_pair_scoring.
    """
    from certimask.aglr_indexer import _generate_sample_positions

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    batch, heads, seq_len, dim = query.shape
    num_blocks = seq_len // block_size
    device = query.device

    used_len = num_blocks * block_size
    q_blocks = query[:, :, :used_len, :].reshape(
        batch, heads, num_blocks, block_size, dim,
    )
    k_blocks = key[:, :, :used_len, :].reshape(
        batch, heads, num_blocks, block_size, dim,
    )

    # Initialize scores to -inf
    scores = torch.full(
        (batch, heads, num_blocks, num_blocks),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )

    # Generate sample positions
    positions = _generate_sample_positions(block_size, sample_pattern)
    q_indices = torch.tensor([p[0] for p in positions], device=device)
    k_indices = torch.tensor([p[1] for p in positions], device=device)
    num_samples = len(positions)

    sqrt_d = torch.sqrt(torch.tensor(float(dim), dtype=torch.float32, device=device))

    # Only score candidate tiles
    candidate_positions = candidate_mask.nonzero(as_tuple=False)

    if candidate_positions.numel() == 0:
        return scores, {
            "computed_tile_count": 0,
            "uses_full_pair_scoring": False,
        }

    # Gather candidate q/k blocks
    b_idx = candidate_positions[:, 0]
    h_idx = candidate_positions[:, 1]
    q_idx = candidate_positions[:, 2]
    k_idx = candidate_positions[:, 3]

    q_sel = q_blocks[b_idx, h_idx, q_idx]  # [N, block_size, D]
    k_sel = k_blocks[b_idx, h_idx, k_idx]

    # Sample positions
    q_sampled = q_sel[:, q_indices, :]  # [N, P, D]
    k_sampled = k_sel[:, k_indices, :]

    # Dot products: [N, P]
    dots = (q_sampled * k_sampled).sum(dim=-1)
    if scale_by_sqrt_dim:
        dots = dots / sqrt_d

    # Aggregate
    if aggregation == "mean":
        tile_scores = dots.mean(dim=-1)
    elif aggregation == "max":
        tile_scores = dots.max(dim=-1).values
    elif aggregation == "topk_mean":
        topk = min(4, dots.shape[-1])
        topk_vals, _ = dots.topk(topk, dim=-1)
        tile_scores = topk_vals.mean(dim=-1)
    elif aggregation == "logsumexp":
        max_val = dots.max(dim=-1).values
        tile_scores = (
            torch.logsumexp(dots - max_val.unsqueeze(-1), dim=-1) + max_val
        )
    else:
        raise ValueError(f"Unknown aggregation '{aggregation}'")

    # Scatter back
    scores[b_idx, h_idx, q_idx, k_idx] = tile_scores
    computed_count = candidate_positions.shape[0]

    meta: dict[str, float | int | str | bool] = {
        "computed_tile_count": computed_count,
        "uses_full_pair_scoring": False,
        "num_samples": num_samples,
    }
    return scores, meta


# ---------------------------------------------------------------------------
# Teacher coverage metrics
# ---------------------------------------------------------------------------

def compute_teacher_selected_coverage(
    candidate_mask: torch.Tensor,
    teacher_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """Compute fraction of teacher-selected tiles that are covered by candidates.

    coverage = |candidate ∩ teacher_selected| / |teacher_selected|

    A coverage of 1.0 means every tile the teacher would select is also
    a candidate. This is the minimum requirement for the candidate-pruned
    indexer to reproduce the teacher's mask exactly.

    Args:
        candidate_mask: [B, H, Q_blk, K_blk] boolean candidate mask.
        teacher_mask: [B, H, Q_blk, K_blk] boolean teacher-selected mask.
        valid_mask: [B, H, Q_blk, K_blk] boolean causal valid mask.

    Returns:
        Coverage fraction in [0, 1].
    """
    teacher_selected = teacher_mask & valid_mask
    teacher_count = int(teacher_selected.sum().item())

    if teacher_count == 0:
        return 1.0

    covered = candidate_mask & teacher_selected
    covered_count = int(covered.sum().item())

    return covered_count / teacher_count


def compute_teacher_mask_overlap(
    candidate_pruned_mask: torch.Tensor,
    teacher_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """Compute overlap between candidate-pruned final mask and teacher mask.

    Uses selected-overlap (Jaccard-like):
    overlap = |pruned ∩ teacher| / |pruned ∪ teacher|

    Both masks are restricted to valid tiles only.

    Args:
        candidate_pruned_mask: [B, H, Q_blk, K_blk] boolean final mask
            after candidate-pruned scoring + top-k.
        teacher_mask: [B, H, Q_blk, K_blk] boolean teacher mask.
        valid_mask: [B, H, Q_blk, K_blk] boolean causal valid mask.

    Returns:
        Overlap fraction in [0, 1].
    """
    pruned_valid = candidate_pruned_mask & valid_mask
    teacher_valid = teacher_mask & valid_mask

    intersection = int((pruned_valid & teacher_valid).sum().item())
    union = int((pruned_valid | teacher_valid).sum().item())

    if union == 0:
        return 1.0

    return intersection / union
