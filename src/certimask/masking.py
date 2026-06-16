"""Threshold-based sparse masking: reference, naive INT8, and CertiMask."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch

from certimask.bounds import ScoreBounds


class CertiMaskDecision(enum.IntEnum):
    """Three-way certified classification for each tile."""

    DROP = 0
    KEEP = 1
    AMBIGUOUS = 2
    INVALID = 3


@dataclass
class CertiMaskResult:
    """Result of certified threshold masking.

    Attributes:
        mask: Boolean mask, True = KEEP. Shape [B, H, Q, K].
        decisions: Integer decisions per tile (CertiMaskDecision values).
            Shape [B, H, Q, K], dtype int64.
        certain_keep: Boolean mask for CERTAIN KEEP tiles.
        certain_drop: Boolean mask for CERTAIN DROP tiles.
        ambiguous: Boolean mask for AMBIGUOUS tiles.
        refined_scores: Reference scores at ambiguous positions, NaN elsewhere.
            None if no ambiguous tiles exist.
    """

    mask: torch.Tensor
    decisions: torch.Tensor
    certain_keep: torch.Tensor
    certain_drop: torch.Tensor
    ambiguous: torch.Tensor
    refined_scores: torch.Tensor | None


def reference_mask(
    reference_scores: torch.Tensor,
    threshold: torch.Tensor | float,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the FP32 reference mask: M_ref = (score > threshold).

    Args:
        reference_scores: FP32 scores, shape [B, H, Q, K].
        threshold: Scalar or broadcastable threshold.
        valid_mask: Optional boolean mask, same broadcastable shape as scores.
            Invalid tiles are always False.

    Returns:
        Boolean mask, shape [B, H, Q, K]. True = KEEP.

    Raises:
        TypeError: If reference_scores is not floating-point.
        ValueError: If inputs contain NaN/Inf or valid_mask is not bool.
    """
    _validate_mask_inputs(reference_scores, valid_mask)

    scores = reference_scores.to(torch.float32)
    mask = scores > threshold

    if valid_mask is not None:
        mask = mask & valid_mask

    return mask


def naive_quantized_mask(
    quantized_scores: torch.Tensor,
    threshold: torch.Tensor | float,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the naive INT8 mask: M_INT8 = (quantized_score > threshold).

    Args:
        quantized_scores: Dequantized INT8 scores, shape [B, H, Q, K].
        threshold: Scalar or broadcastable threshold.
        valid_mask: Optional boolean mask, same broadcastable shape as scores.

    Returns:
        Boolean mask, shape [B, H, Q, K]. True = KEEP.

    Raises:
        TypeError: If quantized_scores is not floating-point.
        ValueError: If inputs contain NaN/Inf or valid_mask is not bool.
    """
    _validate_mask_inputs(quantized_scores, valid_mask)

    scores = quantized_scores.to(torch.float32)
    mask = scores > threshold

    if valid_mask is not None:
        mask = mask & valid_mask

    return mask


def certified_threshold_mask(
    bounds: ScoreBounds,
    reference_scores: torch.Tensor,
    threshold: torch.Tensor | float,
    *,
    valid_mask: torch.Tensor | None = None,
) -> CertiMaskResult:
    """Compute the CertiMask with three-way classification.

    Classification rules:
        - L > threshold  =>  CERTAIN KEEP
        - U < threshold  =>  CERTAIN DROP
        - L <= threshold <= U  =>  AMBIGUOUS  (use reference score to decide)

    For ambiguous tiles, the reference score is used as a fallback,
    guaranteeing M_CertiMask == M_ref.

    Args:
        bounds: ScoreBounds from compute_score_bounds.
        reference_scores: FP32 reference scores, shape [B, H, Q, K].
        threshold: Scalar or broadcastable threshold.
        valid_mask: Optional boolean mask. Invalid tiles get INVALID decision
            and mask=False.

    Returns:
        CertiMaskResult with mask, decisions, and classification masks.

    Raises:
        ValueError: If shapes don't match, inputs contain NaN/Inf, or
            valid_mask is not bool.
    """
    _validate_mask_inputs(reference_scores, valid_mask)

    if bounds.lower.shape != reference_scores.shape:
        raise ValueError(
            f"Shape mismatch: bounds {bounds.lower.shape} vs "
            f"reference_scores {reference_scores.shape}"
        )

    lower = bounds.lower.to(torch.float32)
    upper = bounds.upper.to(torch.float32)
    ref = reference_scores.to(torch.float32)

    # Three-way classification
    certain_keep = lower > threshold
    certain_drop = upper < threshold
    ambiguous = ~certain_keep & ~certain_drop  # includes L==tau, U==tau, and L<tau<U

    # For ambiguous tiles, use reference score
    fallback_keep = ref > threshold

    # Build final mask: certain_keep OR (ambiguous AND fallback)
    mask = certain_keep | (ambiguous & fallback_keep)

    # Build decisions tensor
    decisions = torch.full_like(ref, CertiMaskDecision.DROP, dtype=torch.int64)
    decisions[certain_keep] = CertiMaskDecision.KEEP
    decisions[ambiguous] = CertiMaskDecision.AMBIGUOUS

    # Handle valid_mask — expand to match scores shape for boolean indexing
    if valid_mask is not None:
        vm_expanded = valid_mask.expand_as(ref)
        invalid = ~vm_expanded
        mask = mask & vm_expanded
        decisions[invalid] = CertiMaskDecision.INVALID
        certain_keep = certain_keep & vm_expanded
        certain_drop = certain_drop & vm_expanded
        ambiguous = ambiguous & vm_expanded

    # refined_scores: reference scores at ambiguous positions, NaN elsewhere
    refined_scores: torch.Tensor | None = None
    if ambiguous.any():
        refined_scores = torch.full_like(ref, float("nan"))
        refined_scores[ambiguous] = ref[ambiguous]

    return CertiMaskResult(
        mask=mask,
        decisions=decisions,
        certain_keep=certain_keep,
        certain_drop=certain_drop,
        ambiguous=ambiguous,
        refined_scores=refined_scores,
    )


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


def thresholds_for_target_sparsity(
    reference_scores: torch.Tensor,
    target_sparsity: float,
    *,
    valid_mask: torch.Tensor | None = None,
    per_query: bool = True,
) -> torch.Tensor:
    """Compute thresholds to achieve a target sparsity rate.

    Uses quantile of reference scores over valid tiles. The actual sparsity
    may differ from the target due to strict score > threshold semantics
    and duplicate score values.

    Args:
        reference_scores: FP32 scores, shape [B, H, Q, K].
        target_sparsity: Desired fraction of dropped tiles in [0, 1).
        valid_mask: Optional boolean mask. Only valid tiles participate in
            quantile computation.
        per_query: If True, compute per [B, H, Q] threshold, output shape
            [B, H, Q, 1]. If False, compute per [B, H] threshold, shape
            [B, H, 1, 1].

    Returns:
        Threshold tensor, broadcastable to scores shape.

    Raises:
        ValueError: If target_sparsity is out of range, all tiles are invalid,
            or scores contain NaN/Inf.
    """
    if not (0.0 <= target_sparsity < 1.0):
        raise ValueError(
            f"target_sparsity must be in [0, 1), got {target_sparsity}"
        )

    scores = reference_scores.to(torch.float32)

    if torch.isnan(scores).any():
        raise ValueError("reference_scores contains NaN")
    if torch.isinf(scores).any():
        raise ValueError("reference_scores contains Inf")

    # We want: score > threshold keeps ~ (1 - target_sparsity) fraction.
    # threshold = quantile(scores, target_sparsity) = sorted_scores[floor(count * sparsity)]
    # Replace invalid scores with +inf so they sort to the end (ignored).

    if per_query:
        if valid_mask is not None:
            masked_scores = torch.where(
                valid_mask, scores, torch.tensor(float("inf"))
            )
            counts = valid_mask.sum(dim=-1).to(torch.float32)  # [B, H, Q]
        else:
            masked_scores = scores
            counts = torch.full(
                scores.shape[:3],
                float(scores.shape[-1]),
                dtype=torch.float32,
                device=scores.device,
            )

        # Check for all-invalid rows
        if valid_mask is not None and (counts == 0).any():
            raise ValueError(
                "Some query rows have no valid key tiles"
            )

        sorted_scores, _ = masked_scores.sort(dim=-1)

        rank = torch.clamp(
            (counts * target_sparsity).floor().to(torch.int64),
            min=0,
            max=scores.shape[-1] - 1,
        )

        idx = rank.unsqueeze(-1)  # [B, H, Q, 1]
        thresholds = sorted_scores.gather(-1, idx)  # [B, H, Q, 1]
    else:
        bh_shape = scores.shape[:2]
        flat = scores.reshape(*bh_shape, -1)  # [B, H, Q*K]

        if valid_mask is not None:
            flat_mask = valid_mask.reshape(*bh_shape, -1)
            flat = torch.where(flat_mask, flat, torch.tensor(float("inf")))
            counts = flat_mask.sum(dim=-1).to(torch.float32)  # [B, H]
        else:
            counts = torch.full(
                bh_shape,
                float(flat.shape[-1]),
                dtype=torch.float32,
                device=scores.device,
            )

        if valid_mask is not None and (counts == 0).any():
            raise ValueError(
                "Some batch-head groups have no valid tiles"
            )

        sorted_flat, _ = flat.sort(dim=-1)
        rank = torch.clamp(
            (counts * target_sparsity).floor().to(torch.int64),
            min=0,
            max=flat.shape[-1] - 1,
        )
        idx = rank.unsqueeze(-1)  # [B, H, 1]
        thresholds = sorted_flat.gather(-1, idx)  # [B, H, 1]
        thresholds = thresholds.unsqueeze(-1)  # [B, H, 1, 1]

    return thresholds


def _validate_mask_inputs(
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> None:
    """Validate inputs for mask computation.

    Raises:
        TypeError: If scores is not floating-point or valid_mask is not bool.
        ValueError: If scores contains NaN/Inf.
    """
    if not scores.is_floating_point():
        raise TypeError(f"scores must be a floating-point tensor, got {scores.dtype}")

    if torch.isnan(scores).any():
        raise ValueError("scores contains NaN")
    if torch.isinf(scores).any():
        raise ValueError("scores contains Inf")

    if valid_mask is not None and valid_mask.dtype != torch.bool:
        raise TypeError(
            f"valid_mask must be bool, got {valid_mask.dtype}"
        )
