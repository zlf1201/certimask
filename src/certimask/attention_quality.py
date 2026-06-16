"""Attention quality metrics and sparse/dense attention comparison."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttentionQualityMetrics:
    """Quality metrics comparing sparse vs dense attention.

    Attributes:
        layer_index: Model layer index.
        target_sparsity: Requested tile sparsity.
        actual_tile_sparsity: Fraction of tiles dropped.
        token_mask_sparsity: Fraction of token pairs masked.
        kept_attention_mass_mean: Mean kept attention mass per query.
        kept_attention_mass_p50: P50 of kept attention mass.
        kept_attention_mass_p90: P90 of kept attention mass.
        kept_attention_mass_p99: P99 of kept attention mass.
        dropped_attention_mass_mean: Mean dropped attention mass (1 - kept).
        output_l2_relative_mean: Mean relative L2 error of output.
        output_l2_relative_p90: P90 of relative L2 error.
        output_cosine_mean: Mean cosine similarity of output.
        output_cosine_p10: P10 of cosine similarity.
        prob_l1_mean: Mean L1 distance of attention probabilities.
        prob_l1_p90: P90 of L1 distance.
        prob_kl_mean: Mean KL divergence (dense || sparse).
        prob_kl_p90: P90 of KL divergence.
    """

    layer_index: int
    target_sparsity: float
    actual_tile_sparsity: float
    token_mask_sparsity: float
    kept_attention_mass_mean: float
    kept_attention_mass_p50: float
    kept_attention_mass_p90: float
    kept_attention_mass_p99: float
    dropped_attention_mass_mean: float
    output_l2_relative_mean: float
    output_l2_relative_p90: float
    output_cosine_mean: float
    output_cosine_p10: float
    prob_l1_mean: float
    prob_l1_p90: float
    prob_kl_mean: float
    prob_kl_p90: float


@dataclass
class BenefitProxyMetrics:
    """Proxy metrics for potential compute benefit.

    Attributes:
        layer_index: Model layer index.
        decision: Layer decision (Go / Conditional / Fallback).
        selected_strategy: Selected quantization strategy.
        tile_sparsity: Fraction of tiles dropped.
        kept_tile_fraction: Fraction of tiles kept.
        score_work_fraction_proxy: Scoring work proxy (refinement rate or 1.0).
        attention_tile_work_fraction: Attention tile work fraction.
    """

    layer_index: int
    decision: str
    selected_strategy: str
    tile_sparsity: float
    kept_tile_fraction: float
    score_work_fraction_proxy: float
    attention_tile_work_fraction: float


def dense_attention_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    scale_by_sqrt_dim: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dense attention output and probabilities.

    Args:
        query: [B, H, L, D]
        key: [B, H, L, D] (already GQA-expanded)
        value: [B, H, L, D] (already GQA-expanded)
        causal: Whether to apply causal mask.
        scale_by_sqrt_dim: Whether to scale by 1/sqrt(d).

    Returns:
        (attention_output [B, H, L, D], attention_probs [B, H, L, L])
    """
    q = query.float()
    k = key.float()
    v = value.float()
    d = q.shape[-1]

    logits = torch.matmul(q, k.transpose(-2, -1))  # [B, H, L, L]
    if scale_by_sqrt_dim:
        logits = logits / (d ** 0.5)

    if causal:
        seq_len = q.shape[-2]
        causal_mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=logits.device, dtype=logits.dtype
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        logits = logits + causal_mask.unsqueeze(0).unsqueeze(0)

    probs = torch.softmax(logits, dim=-1)  # [B, H, L, L]
    output = torch.matmul(probs, v)  # [B, H, L, D]

    return output, probs


def expand_block_mask_to_token_mask(
    block_mask: torch.Tensor,
    *,
    block_size: int,
    seq_len: int,
    causal: bool = True,
) -> torch.Tensor:
    """Expand block-level mask to token-level mask.

    Args:
        block_mask: [B, H, Q_blk, K_blk] boolean.
        block_size: Tokens per block.
        seq_len: Actual sequence length.
        causal: If True, also apply causal mask (future tokens = False).

    Returns:
        Boolean token mask [B, H, seq_len, seq_len].
    """
    token_mask = block_mask.repeat_interleave(block_size, dim=-2)
    token_mask = token_mask.repeat_interleave(block_size, dim=-1)
    token_mask = token_mask[:, :, :seq_len, :seq_len]

    if causal:
        ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=block_mask.device)
        causal_mask = torch.tril(ones)
        token_mask = token_mask & causal_mask.unsqueeze(0).unsqueeze(0)

    return token_mask


def block_sparse_attention_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    block_size: int,
    causal: bool = True,
    scale_by_sqrt_dim: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute block-sparse attention output.

    Args:
        query: [B, H, L, D]
        key: [B, H, L, D] (already GQA-expanded)
        value: [B, H, L, D] (already GQA-expanded)
        block_mask: [B, H, Q_blocks, K_blocks] boolean.
            True = block is kept.
        block_size: Tokens per block.
        causal: Whether to also apply causal mask.
        scale_by_sqrt_dim: Whether to scale by 1/sqrt(d).

    Returns:
        (attention_output [B, H, L, D], attention_probs [B, H, L, L])
    """
    q = query.float()
    k = key.float()
    v = value.float()
    dim = q.shape[-1]
    seq_len = q.shape[-2]

    logits = torch.matmul(q, k.transpose(-2, -1))  # [B, H, L, L]
    if scale_by_sqrt_dim:
        logits = logits / (dim ** 0.5)

    # Expand block_mask to token-level mask
    # block_mask: [B, H, Q_blk, K_blk]
    # Use repeat_interleave for correct block tiling (expand+reshape is buggy
    # on non-contiguous tensors)
    token_mask = block_mask.repeat_interleave(block_size, dim=-2)
    token_mask = token_mask.repeat_interleave(block_size, dim=-1)
    # Trim to actual seq_len (in case seq_len is not a multiple of block_size)
    token_mask = token_mask[:, :, :seq_len, :seq_len]

    # Apply block mask: masked positions get -inf
    logits = logits.masked_fill(~token_mask, float("-inf"))

    if causal:
        causal_mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=logits.device, dtype=logits.dtype
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        logits = logits + causal_mask.unsqueeze(0).unsqueeze(0)

    probs = torch.softmax(logits, dim=-1)  # [B, H, L, L]
    # NaN from all-inf rows -> 0
    probs = torch.nan_to_num(probs, nan=0.0)

    output = torch.matmul(probs, v)  # [B, H, L, D]

    return output, probs


def compute_attention_quality(
    dense_output: torch.Tensor,
    dense_probs: torch.Tensor,
    sparse_output: torch.Tensor,
    sparse_probs: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
    *,
    layer_index: int,
    target_sparsity: float,
    valid_block_mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> AttentionQualityMetrics:
    """Compute attention quality metrics.

    Args:
        dense_output: [B, H, L, D]
        dense_probs: [B, H, L, L]
        sparse_output: [B, H, L, D]
        sparse_probs: [B, H, L, L]
        block_mask: [B, H, Q_blk, K_blk]
        block_size: Tokens per block.
        layer_index: Layer index.
        target_sparsity: Target tile sparsity.
        valid_block_mask: Optional [B, H, Q_blk, K_blk] boolean mask
            indicating causally valid tiles. If provided, tile sparsity
            is computed only over valid tiles.
        eps: Small value for numerical stability.

    Returns:
        AttentionQualityMetrics.
    """
    batch, heads, seq_len, _ = dense_output.shape
    _, _, q_blk, k_blk = block_mask.shape

    # Tile sparsity: only count valid tiles
    if valid_block_mask is not None:
        valid_tiles = valid_block_mask.sum().item()
        kept_tiles = (block_mask & valid_block_mask).sum().item()
    else:
        valid_tiles = q_blk * k_blk
        kept_tiles = block_mask.sum().item()
    actual_tile_sparsity = 1.0 - kept_tiles / valid_tiles if valid_tiles > 0 else 0.0

    # Token mask sparsity - expand block mask to token level
    # Use causal=True to ensure future positions are excluded
    token_mask = expand_block_mask_to_token_mask(
        block_mask, block_size=block_size, seq_len=seq_len, causal=True,
    )
    total_token_pairs = batch * heads * seq_len * seq_len
    kept_token_pairs = token_mask.sum().item()
    token_mask_sparsity = 1.0 - kept_token_pairs / total_token_pairs

    # Kept attention mass per query token
    # For each query position i, sum dense_probs over kept key positions
    kept_mass_per_query = (dense_probs * token_mask.float()).sum(dim=-1)
    kept_mass_flat = kept_mass_per_query.reshape(-1)

    # Dropped mass
    dropped_mass_flat = 1.0 - kept_mass_flat

    # Output relative L2 error
    output_diff = (sparse_output - dense_output).norm(dim=-1)
    output_norm = dense_output.norm(dim=-1)
    output_l2_rel = output_diff / (output_norm + eps)
    output_l2_rel_flat = output_l2_rel.reshape(-1)

    # Output cosine similarity
    dense_flat = dense_output.reshape(batch * heads * seq_len, -1)
    sparse_flat = sparse_output.reshape(batch * heads * seq_len, -1)
    cosine = torch.nn.functional.cosine_similarity(dense_flat, sparse_flat, dim=-1)

    # Probability L1 distance
    prob_l1 = (sparse_probs - dense_probs).abs().sum(dim=-1)
    prob_l1_flat = prob_l1.reshape(-1)

    # KL divergence: KL(dense || sparse)
    # Use log with epsilon to avoid log(0)
    dense_log = (dense_probs + eps).log()
    sparse_log = (sparse_probs + eps).log()
    kl_per_pos = (dense_probs * (dense_log - sparse_log)).sum(dim=-1)
    kl_flat = kl_per_pos.reshape(-1)

    def _q(t: torch.Tensor, q: float) -> float:
        return float(t.quantile(q).item())

    return AttentionQualityMetrics(
        layer_index=layer_index,
        target_sparsity=target_sparsity,
        actual_tile_sparsity=actual_tile_sparsity,
        token_mask_sparsity=token_mask_sparsity,
        kept_attention_mass_mean=float(kept_mass_flat.mean().item()),
        kept_attention_mass_p50=_q(kept_mass_flat, 0.50),
        kept_attention_mass_p90=_q(kept_mass_flat, 0.90),
        kept_attention_mass_p99=_q(kept_mass_flat, 0.99),
        dropped_attention_mass_mean=float(dropped_mass_flat.mean().item()),
        output_l2_relative_mean=float(output_l2_rel_flat.mean().item()),
        output_l2_relative_p90=_q(output_l2_rel_flat, 0.90),
        output_cosine_mean=float(cosine.mean().item()),
        output_cosine_p10=_q(cosine, 0.10),
        prob_l1_mean=float(prob_l1_flat.mean().item()),
        prob_l1_p90=_q(prob_l1_flat, 0.90),
        prob_kl_mean=float(kl_flat.mean().item()),
        prob_kl_p90=_q(kl_flat, 0.90),
    )


def compute_oracle_block_mass_scores(
    attention_probs: torch.Tensor,
    *,
    block_size: int,
    causal: bool = True,
) -> torch.Tensor:
    """Compute oracle block mass scores from dense attention probabilities.

    For each query block a and key block b:
        mass_{a,b} = (1/|Q_a|) * sum_{i in Q_a} sum_{j in K_b} A_dense[i,j]

    Only causal-valid key tokens are counted.

    Args:
        attention_probs: [B, H, L, L] dense attention probabilities.
        block_size: Tokens per block.
        causal: If True, only count causal-valid positions.

    Returns:
        block_mass_scores: [B, H, Q_blk, K_blk]
    """
    batch, heads, seq_len, _ = attention_probs.shape
    q_blk = (seq_len + block_size - 1) // block_size
    k_blk = q_blk

    # Pad to multiple of block_size if needed
    pad_q = q_blk * block_size - seq_len
    pad_k = k_blk * block_size - seq_len
    if pad_q > 0 or pad_k > 0:
        probs_padded = torch.zeros(
            batch, heads, q_blk * block_size, k_blk * block_size,
            device=attention_probs.device, dtype=attention_probs.dtype,
        )
        probs_padded[:, :, :seq_len, :seq_len] = attention_probs
    else:
        probs_padded = attention_probs

    # Reshape to [B, H, Q_blk, block_size, K_blk, block_size]
    probs_blocks = probs_padded.reshape(
        batch, heads, q_blk, block_size, k_blk, block_size,
    )

    # Sum over tokens within each block
    # mass[a,b] = sum_i sum_j probs[i,j] for i in block a, j in block b
    block_mass = probs_blocks.sum(dim=(3, 5))  # [B, H, Q_blk, K_blk]

    # Normalize by block size (average per query token)
    block_mass = block_mass / block_size

    return block_mass


def oracle_block_mass_mask(
    attention_probs: torch.Tensor,
    *,
    block_size: int,
    target_sparsity: float,
    valid_block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create oracle block mask based on dense attention mass.

    Keeps tiles with highest mass to achieve target sparsity.

    Args:
        attention_probs: [B, H, L, L] dense attention probabilities.
        block_size: Tokens per block.
        target_sparsity: Fraction of valid tiles to drop.
        valid_block_mask: Optional [B, H, Q_blk, K_blk] boolean mask.

    Returns:
        Boolean block mask [B, H, Q_blk, K_blk].
    """
    block_mass = compute_oracle_block_mass_scores(
        attention_probs, block_size=block_size, causal=True,
    )
    batch, heads, q_blk, k_blk = block_mass.shape

    if valid_block_mask is None:
        valid_block_mask = torch.ones_like(block_mass, dtype=torch.bool)

    # For each query block row, keep top-k mass tiles among valid ones
    # Number to keep per row
    valid_per_row = valid_block_mask.sum(dim=-1)  # [B, H, Q_blk]
    keep_per_row = (valid_per_row.float() * (1.0 - target_sparsity)).ceil().long()
    keep_per_row = torch.clamp(keep_per_row, min=1)

    mask = torch.zeros_like(block_mass, dtype=torch.bool)

    for b in range(batch):
        for h in range(heads):
            for q in range(q_blk):
                n_keep = keep_per_row[b, h, q].item()
                row_mass = block_mass[b, h, q].clone()
                # Set invalid tiles to -inf so they're never selected
                row_mass[~valid_block_mask[b, h, q]] = float("-inf")
                n_valid = int(valid_per_row[b, h, q].item())
                k_val = int(min(n_keep, n_valid))
                _, topk_idx = row_mass.topk(k_val)
                mask[b, h, q, topk_idx] = True

    return mask


def local_window_block_mask(
    num_query_blocks: int,
    num_key_blocks: int,
    *,
    window_blocks: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a local window block mask.

    For each query block, keeps the nearest `window_blocks` causal key blocks.

    Args:
        num_query_blocks: Number of query blocks.
        num_key_blocks: Number of key blocks.
        window_blocks: Number of key blocks to keep per query block.
        device: Device for the tensor.

    Returns:
        Boolean block mask [1, 1, Q_blk, K_blk].
    """
    mask = torch.zeros(1, 1, num_query_blocks, num_key_blocks, dtype=torch.bool, device=device)
    for q in range(num_query_blocks):
        start = max(0, q - window_blocks + 1)
        mask[0, 0, q, start:q + 1] = True
    return mask


def random_valid_block_mask(
    valid_block_mask: torch.Tensor,
    *,
    target_sparsity: float,
    seed: int = 42,
) -> torch.Tensor:
    """Create a random block mask within valid tiles.

    Args:
        valid_block_mask: [B, H, Q_blk, K_blk] boolean mask.
        target_sparsity: Fraction of valid tiles to drop.
        seed: Random seed.

    Returns:
        Boolean block mask [B, H, Q_blk, K_blk].
    """
    gen = torch.Generator(device=valid_block_mask.device).manual_seed(seed)
    rand = torch.rand(valid_block_mask.shape, generator=gen, device=valid_block_mask.device)
    keep = rand < (1.0 - target_sparsity)
    return keep & valid_block_mask


def local_plus_extra_mask(
    extra_scores: torch.Tensor,
    *,
    target_sparsity: float,
    local_blocks: int,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Create a mask with mandatory local blocks plus extra blocks by score.

    For each query block row:
    1. Keep the nearest `local_blocks` causal-valid key blocks.
    2. In the remaining budget, keep blocks with highest `extra_scores`.

    Args:
        extra_scores: [B, H, Q_blk, K_blk] scores for extra block selection.
        target_sparsity: Fraction of valid tiles to drop overall.
        local_blocks: Number of mandatory local blocks per query row.
        valid_mask: [B, H, Q_blk, K_blk] boolean causal valid mask.

    Returns:
        Tuple of (block_mask, local_budget_overflow).
        block_mask: [B, H, Q_blk, K_blk] boolean.
        local_budget_overflow: True if local blocks exceed the budget.
    """
    batch, heads, q_blk, k_blk = extra_scores.shape

    valid_per_row = valid_mask.sum(dim=-1)  # [B, H, Q_blk]
    keep_per_row = (valid_per_row.float() * (1.0 - target_sparsity)).ceil().long()
    keep_per_row = torch.clamp(keep_per_row, min=1)

    mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    overflow = False

    for b in range(batch):
        for h in range(heads):
            for q in range(q_blk):
                n_keep = int(keep_per_row[b, h, q].item())
                valid_k = valid_mask[b, h, q]  # [K_blk]

                # Step 1: mandatory local blocks (nearest causal blocks)
                local_start = max(0, q - local_blocks + 1)
                local_range = torch.arange(k_blk, device=extra_scores.device)
                is_local = (local_range >= local_start) & (local_range <= q) & valid_k

                n_local = int(is_local.sum().item())
                mask[b, h, q, is_local] = True

                if n_local >= n_keep:
                    overflow = True
                    continue

                # Step 2: fill remaining budget with highest extra_scores
                remaining = n_keep - n_local
                scores_row = extra_scores[b, h, q].clone()
                # Exclude already-selected local blocks and invalid blocks
                scores_row[is_local] = float("-inf")
                scores_row[~valid_k] = float("-inf")

                n_available = int((scores_row > float("-inf")).sum().item())
                n_extra = min(remaining, n_available)

                if n_extra > 0:
                    _, extra_idx = scores_row.topk(n_extra)
                    mask[b, h, q, extra_idx] = True

    return mask, overflow


def compute_benefit_proxy(
    layer_index: int,
    decision: str,
    selected_strategy: str,
    block_mask: torch.Tensor,
    refinement_rate: float,
) -> BenefitProxyMetrics:
    """Compute benefit proxy metrics for a layer.

    Args:
        layer_index: Layer index.
        decision: Layer decision string.
        selected_strategy: Selected strategy name.
        block_mask: [B, H, Q_blk, K_blk] boolean.
        refinement_rate: CertiMask refinement rate for this layer.

    Returns:
        BenefitProxyMetrics.
    """
    total_tiles = block_mask.numel()
    kept_tiles = block_mask.sum().item()
    tile_sparsity = 1.0 - kept_tiles / total_tiles
    kept_tile_fraction = kept_tiles / total_tiles

    score_work_fraction = 1.0 if decision == "FP16 fallback" else refinement_rate

    return BenefitProxyMetrics(
        layer_index=layer_index,
        decision=decision,
        selected_strategy=selected_strategy,
        tile_sparsity=tile_sparsity,
        kept_tile_fraction=kept_tile_fraction,
        score_work_fraction_proxy=score_work_fraction,
        attention_tile_work_fraction=kept_tile_fraction,
    )
