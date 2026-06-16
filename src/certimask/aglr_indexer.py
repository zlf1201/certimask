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
class BlockLandmarks:
    """Landmark vectors for blocks.

    Attributes:
        landmarks: [B, H, num_blocks, R, D] landmark vectors.
        method: Landmark selection method.
        num_blocks: Number of blocks.
        block_size: Block size.
        num_landmarks: Number of landmarks per block (R).
    """

    landmarks: torch.Tensor
    method: str
    num_blocks: int
    block_size: int
    num_landmarks: int


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


def select_block_landmarks(
    states: torch.Tensor,
    *,
    block_size: int,
    method: str,
    num_landmarks: int = 2,
    drop_incomplete_block: bool = True,
) -> BlockLandmarks:
    """Select landmark vectors for each block.

    Args:
        states: [B, H, L, D] float tensor.
        block_size: Number of tokens per block.
        method: One of 'mean', 'last', 'max_norm', 'topk_norm',
            'mean_plus_max_norm', 'mean_plus_topk_norm'.
        num_landmarks: Number of landmarks for topk methods.
        drop_incomplete_block: If True, drop trailing incomplete block.

    Returns:
        BlockLandmarks with landmarks [B, H, num_blocks, R, D].

    Raises:
        TypeError: If states is not floating-point.
        ValueError: If states is empty, contains NaN/Inf, or params invalid.
    """
    if not states.is_floating_point():
        raise TypeError(f"states must be floating-point, got {states.dtype}")
    if states.numel() == 0:
        raise ValueError("states is empty")
    if torch.isnan(states).any():
        raise ValueError("states contains NaN")
    if torch.isinf(states).any():
        raise ValueError("states contains Inf")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    states = states.to(torch.float32)
    batch, heads, seq_len, dim = states.shape
    num_blocks = seq_len // block_size
    if num_blocks == 0:
        raise ValueError(
            f"seq_len {seq_len} < block_size {block_size}, no complete blocks"
        )

    used_len = num_blocks * block_size
    states = states[:, :, :used_len, :]

    # Reshape to [B, H, num_blocks, block_size, D]
    blocks = states.reshape(batch, heads, num_blocks, block_size, dim)

    if method == "mean":
        landmarks = blocks.mean(dim=3, keepdim=True)  # [B,H,nb,1,D]
    elif method == "last":
        landmarks = blocks[:, :, :, -1:, :]  # [B,H,nb,1,D]
    elif method == "max_norm":
        norms = blocks.norm(dim=-1)  # [B,H,nb,bs]
        idx = norms.argmax(dim=-1, keepdim=True)  # [B,H,nb,1]
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, 1, dim)
        landmarks = blocks.gather(3, idx_expanded)  # [B,H,nb,1,D]
    elif method == "topk_norm":
        norms = blocks.norm(dim=-1)  # [B,H,nb,bs]
        k = min(num_landmarks, block_size)
        _, topk_idx = norms.topk(k, dim=-1)  # [B,H,nb,k]
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, dim)
        landmarks = blocks.gather(3, topk_idx_expanded)  # [B,H,nb,k,D]
    elif method == "mean_plus_max_norm":
        mean_vec = blocks.mean(dim=3, keepdim=True)
        norms = blocks.norm(dim=-1)
        idx = norms.argmax(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, -1, 1, dim)
        max_vec = blocks.gather(3, idx)
        landmarks = torch.cat([mean_vec, max_vec], dim=3)  # [B,H,nb,2,D]
    elif method == "mean_plus_topk_norm":
        mean_vec = blocks.mean(dim=3, keepdim=True)
        norms = blocks.norm(dim=-1)
        k = min(num_landmarks, block_size)
        _, topk_idx = norms.topk(k, dim=-1)
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, dim)
        topk_vec = blocks.gather(3, topk_idx_expanded)
        landmarks = torch.cat([mean_vec, topk_vec], dim=3)  # [B,H,nb,1+k,D]
    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported: mean, last, max_norm, "
            "topk_norm, mean_plus_max_norm, mean_plus_topk_norm"
        )

    return BlockLandmarks(
        landmarks=landmarks,
        method=method,
        num_blocks=num_blocks,
        block_size=block_size,
        num_landmarks=landmarks.shape[3],
    )


def compute_landmark_block_scores(
    query_landmarks: BlockLandmarks,
    key_landmarks: BlockLandmarks,
    *,
    score_method: str,
    scale_by_sqrt_dim: bool = True,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute block scores from landmark vectors.

    Args:
        query_landmarks: Query block landmarks.
        key_landmarks: Key block landmarks.
        score_method: One of 'max', 'mean', 'logsumexp', 'hybrid'.
        scale_by_sqrt_dim: Whether to scale by 1/sqrt(d).
        valid_mask: Optional [B, H, Q_blk, K_blk] boolean mask.

    Returns:
        scores: [B, H, Q_blk, K_blk] block scores.
    """
    q_lm = query_landmarks.landmarks  # [B,H,Q,Rq,D]
    k_lm = key_landmarks.landmarks  # [B,H,K,Rk,D]

    dim = q_lm.shape[-1]
    # Compute all pairwise dot products: [B,H,Q,Rq,K,Rk]
    dots = torch.einsum("bhqrd,bhksd->bhqrks", q_lm, k_lm)

    if scale_by_sqrt_dim:
        sqrt_d = torch.sqrt(torch.tensor(float(dim), dtype=torch.float32))
        dots = dots / sqrt_d

    # dots shape: [B, H, Q, Rq, K, Rk]
    # For each (Q, K) block pair, reduce over landmark dims (Rq, Rk).
    # Rearrange to [B, H, Q, K, Rq*Rk] so we reduce over the last dim.
    batch, heads = dots.shape[0], dots.shape[1]
    q_blk = dots.shape[2]
    k_blk = dots.shape[4]
    dots_pairwise = dots.permute(0, 1, 2, 4, 3, 5)  # [B,H,Q,K,Rq,Rk]
    dots_flat = dots_pairwise.reshape(batch, heads, q_blk, k_blk, -1)  # [B,H,Q,K,Rq*Rk]

    if score_method == "max":
        scores = dots_flat.max(dim=-1).values
    elif score_method == "mean":
        scores = dots_flat.mean(dim=-1)
    elif score_method == "logsumexp":
        max_val = dots_flat.max(dim=-1).values  # [B,H,Q,K]
        scores = (
            torch.logsumexp(dots_flat - max_val.unsqueeze(-1), dim=-1)
            + max_val
        )
    elif score_method == "hybrid":
        s_max = dots_flat.max(dim=-1).values
        s_mean = dots_flat.mean(dim=-1)
        # Recency prior: -log(1 + a - b) for causal-valid tiles
        q_idx = torch.arange(
            query_landmarks.num_blocks, device=dots.device
        ).float()
        k_idx = torch.arange(
            key_landmarks.num_blocks, device=dots.device
        ).float()
        recency = -torch.log1p(
            (q_idx.unsqueeze(1) - k_idx.unsqueeze(0)).clamp(min=0)
        )
        recency = recency.unsqueeze(0).unsqueeze(0)  # [1,1,Q,K]
        scores = 0.6 * s_max + 0.3 * s_mean + 0.1 * recency
    else:
        raise ValueError(
            f"Unknown score_method '{score_method}'. "
            "Supported: max, mean, logsumexp, hybrid"
        )

    # Mask invalid/future tiles
    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, torch.finfo(torch.float32).min)

    return scores


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


def aglr_adaptive_mass_budget_mask(
    landmark_scores: torch.Tensor,
    *,
    target_proxy_mass: float,
    local_blocks: int,
    valid_mask: torch.Tensor,
    min_keep_blocks: int = 1,
    max_keep_fraction: float = 0.60,
) -> AGLRMaskResult:
    """Create mask with adaptive proxy-mass budget.

    For each query row, keeps blocks until cumulative softmax proxy mass
    reaches target_proxy_mass.

    Args:
        landmark_scores: [B, H, Q_blk, K_blk] block scores.
        target_proxy_mass: Target cumulative proxy mass to retain.
        local_blocks: Number of mandatory local blocks.
        valid_mask: [B, H, Q_blk, K_blk] boolean causal valid mask.
        min_keep_blocks: Minimum blocks to keep per row.
        max_keep_fraction: Maximum fraction of valid blocks to keep.

    Returns:
        AGLRMaskResult.
    """
    batch, heads, q_blk, k_blk = landmark_scores.shape
    device = landmark_scores.device

    # Proxy mass: softmax over valid scores per row
    proxy_scores = landmark_scores.clone()
    proxy_scores[~valid_mask] = torch.finfo(torch.float32).min
    proxy_mass = torch.softmax(proxy_scores, dim=-1)  # [B, H, Q, K]

    local_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    extra_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    overflow_count = 0

    for b in range(batch):
        for h in range(heads):
            for q in range(q_blk):
                valid_k = valid_mask[b, h, q]
                n_valid = int(valid_k.sum().item())
                max_keep = max(min_keep_blocks, int(n_valid * max_keep_fraction))

                # Mandatory local blocks
                local_start = max(0, q - local_blocks + 1)
                local_range = torch.arange(k_blk, device=device)
                is_local = (local_range >= local_start) & (local_range <= q) & valid_k
                local_mask[b, h, q, is_local] = True
                n_local = int(is_local.sum().item())

                if n_local >= max_keep:
                    overflow_count += 1
                    continue

                # Sort remaining valid blocks by score
                scores_row = landmark_scores[b, h, q].clone()
                scores_row[is_local] = torch.finfo(torch.float32).min
                scores_row[~valid_k] = torch.finfo(torch.float32).min
                mass_row = proxy_mass[b, h, q].clone()
                mass_row[is_local] = 0.0
                mass_row[~valid_k] = 0.0

                # Sort by score descending
                sorted_scores, sorted_idx = scores_row.sort(descending=True)

                cumulative_mass = 0.0
                selected: list[int] = []
                for idx in sorted_idx.tolist():
                    if sorted_scores[len(selected)].item() <= torch.finfo(torch.float32).min:
                        break
                    if len(selected) + n_local >= max_keep:
                        overflow_count += 1
                        break
                    cumulative_mass += mass_row[idx].item()
                    selected.append(idx)
                    if cumulative_mass >= target_proxy_mass:
                        break

                if selected:
                    extra_mask[b, h, q, torch.tensor(selected, device=device)] = True

    mask = local_mask | extra_mask

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
