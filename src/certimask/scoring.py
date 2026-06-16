"""Block score computation: FP32 reference and INT8 quantized."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from certimask.quantization import (
    GroupQuantizedTensor,
    QuantizedTensor,
    quantize_int8_per_group,
    quantize_int8_per_vector,
)


@dataclass
class QuantizedScoreResult:
    """Result of INT8 quantized block scoring.

    Attributes:
        scores: Dequantized scores in float32, shape [B, H, Q, K].
        integer_dot: Raw integer dot product, shape [B, H, Q, K], dtype int32.
        query_quantized: Quantized query blocks.
        key_quantized: Quantized key blocks.
    """

    scores: torch.Tensor
    integer_dot: torch.Tensor
    query_quantized: QuantizedTensor
    key_quantized: QuantizedTensor


def reference_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scale_by_sqrt_dim: bool = True,
) -> torch.Tensor:
    """Compute FP32 reference block scores s_{ab} = q_a^T k_b.

    Args:
        query: Query blocks, shape [B, H, Q, d].
        key: Key blocks, shape [B, H, K, d].
        scale_by_sqrt_dim: If True, divide scores by sqrt(d).

    Returns:
        Score tensor of shape [B, H, Q, K] in float32.

    Raises:
        TypeError: If inputs are not floating-point tensors.
        ValueError: If inputs are not 4-D, shapes incompatible, contain
            NaN/Inf, or are empty.
    """
    _validate_score_inputs(query, key)

    # Work in float32 for numerical stability
    q_f32 = query.to(torch.float32)
    k_f32 = key.to(torch.float32)

    # s_{ab} = q_a^T k_b
    scores = torch.einsum("bhqd,bhkd->bhqk", q_f32, k_f32)

    if scale_by_sqrt_dim:
        d = query.shape[-1]
        scores = scores / torch.sqrt(torch.tensor(float(d), dtype=torch.float32))

    return scores


def quantized_int8_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scale_by_sqrt_dim: bool = True,
) -> QuantizedScoreResult:
    """Compute INT8 quantized block scores.

    Quantizes query and key per-vector to INT8, computes integer dot product,
    then dequantizes the scores.

    Args:
        query: Query blocks, shape [B, H, Q, d].
        key: Key blocks, shape [B, H, K, d].
        scale_by_sqrt_dim: If True, divide scores by sqrt(d).

    Returns:
        QuantizedScoreResult with dequantized scores, integer dot product,
        and quantized query/key tensors.

    Raises:
        TypeError: If inputs are not floating-point tensors.
        ValueError: If inputs are not 4-D, shapes incompatible, contain
            NaN/Inf, or are empty.
    """
    _validate_score_inputs(query, key)

    # Quantize per-vector along head_dim
    q_quant = quantize_int8_per_vector(query, dim=-1)
    k_quant = quantize_int8_per_vector(key, dim=-1)

    # Integer dot product: q_int8^T k_int8
    # Convert int8 to int32 to avoid overflow, then use einsum
    # Note: This performs the dot product in float32 as int32 batched matmul
    # is not universally supported. The semantic is integer dot product since
    # the values are integer-valued float32 tensors.
    q_int = q_quant.values.to(torch.int32).to(torch.float32)
    k_int = k_quant.values.to(torch.int32).to(torch.float32)
    integer_dot = torch.einsum("bhqd,bhkd->bhqk", q_int, k_int).to(torch.int32)

    # Dequantized scores: z_{ab} * alpha_q * alpha_k
    # q_quant.scale: [B, H, Q, 1], k_quant.scale: [B, H, K, 1]
    # We need [B, H, Q, K]
    q_scale = q_quant.scale.squeeze(-1)  # [B, H, Q]
    k_scale = k_quant.scale.squeeze(-1)  # [B, H, K]

    # Broadcast: [B, H, Q, 1] * [B, H, 1, K] -> [B, H, Q, K]
    scores = integer_dot.to(torch.float32) * q_scale.unsqueeze(-1) * k_scale.unsqueeze(-2)

    if scale_by_sqrt_dim:
        d = query.shape[-1]
        scores = scores / torch.sqrt(torch.tensor(float(d), dtype=torch.float32))

    return QuantizedScoreResult(
        scores=scores,
        integer_dot=integer_dot,
        query_quantized=q_quant,
        key_quantized=k_quant,
    )


def _validate_score_inputs(query: torch.Tensor, key: torch.Tensor) -> None:
    """Validate inputs for score computation.

    Raises:
        TypeError: If inputs are not floating-point tensors.
        ValueError: If inputs are not 4-D, shapes incompatible, contain
            NaN/Inf, or are empty.
    """
    if not query.is_floating_point():
        raise TypeError(f"query must be a floating-point tensor, got {query.dtype}")
    if not key.is_floating_point():
        raise TypeError(f"key must be a floating-point tensor, got {key.dtype}")

    if query.dim() != 4:
        raise ValueError(f"query must be 4-D, got {query.dim()}-D")
    if key.dim() != 4:
        raise ValueError(f"key must be 4-D, got {key.dim()}-D")

    if query.shape[0] != key.shape[0]:
        raise ValueError(
            f"batch dimension mismatch: query {query.shape[0]} vs key {key.shape[0]}"
        )
    if query.shape[1] != key.shape[1]:
        raise ValueError(
            f"heads dimension mismatch: query {query.shape[1]} vs key {key.shape[1]}"
        )
    if query.shape[3] != key.shape[3]:
        raise ValueError(
            f"head_dim dimension mismatch: query {query.shape[3]} vs key {key.shape[3]}"
        )

    if query.numel() == 0 or key.numel() == 0:
        raise ValueError("Input tensors must not be empty")

    if torch.isnan(query).any() or torch.isnan(key).any():
        raise ValueError("Input tensors contain NaN")

    if torch.isinf(query).any() or torch.isinf(key).any():
        raise ValueError("Input tensors contain Inf")


@dataclass
class KOnlyScoreResult:
    """Result of K-only INT8 scoring (Q in float, K quantized).

    Attributes:
        scores: Dequantized scores in float32, shape [B, H, Q, K].
        query: Original float32 query, shape [B, H, Q, d].
        key_quantized: Quantized key (per-vector or per-group).
    """

    scores: torch.Tensor
    query: torch.Tensor
    key_quantized: QuantizedTensor


@dataclass
class KOnlyGroupScoreResult:
    """Result of K-only per-group INT8 scoring.

    Attributes:
        scores: Dequantized scores in float32, shape [B, H, Q, K].
        query: Original float32 query, shape [B, H, Q, d].
        key_quantized: GroupQuantizedTensor for the key.
    """

    scores: torch.Tensor
    query: torch.Tensor
    key_quantized: GroupQuantizedTensor


def k_only_per_vector_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scale_by_sqrt_dim: bool = True,
) -> KOnlyScoreResult:
    """Compute K-only scores: Q in float32, K quantized per-vector to INT8.

    score = q^T * k_tilde, optionally / sqrt(d).

    Args:
        query: Query blocks, shape [B, H, Q, d].
        key: Key blocks, shape [B, H, K, d].
        scale_by_sqrt_dim: If True, divide by sqrt(d).

    Returns:
        KOnlyScoreResult.
    """
    _validate_score_inputs(query, key)

    q_f32 = query.to(torch.float32)
    k_quant = quantize_int8_per_vector(key, dim=-1)
    k_deq = k_quant.dequantized.to(torch.float32)

    scores = torch.einsum("bhqd,bhkd->bhqk", q_f32, k_deq)

    if scale_by_sqrt_dim:
        d = query.shape[-1]
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=torch.float32))
        scores = scores / sqrt_d

    return KOnlyScoreResult(scores=scores, query=q_f32, key_quantized=k_quant)


def k_only_per_group_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    group_size: int,
    scale_by_sqrt_dim: bool = True,
) -> KOnlyGroupScoreResult:
    """Compute K-only scores: Q in float32, K quantized per-group to INT8.

    Args:
        query: Query blocks, shape [B, H, Q, d].
        key: Key blocks, shape [B, H, K, d].
        group_size: Group size for K quantization.
        scale_by_sqrt_dim: If True, divide by sqrt(d).

    Returns:
        KOnlyGroupScoreResult.
    """
    _validate_score_inputs(query, key)

    q_f32 = query.to(torch.float32)
    k_quant = quantize_int8_per_group(key, group_size=group_size, dim=-1)
    k_deq = k_quant.dequantized.to(torch.float32)

    scores = torch.einsum("bhqd,bhkd->bhqk", q_f32, k_deq)

    if scale_by_sqrt_dim:
        d = query.shape[-1]
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=torch.float32))
        scores = scores / sqrt_d

    return KOnlyGroupScoreResult(scores=scores, query=q_f32, key_quantized=k_quant)


@dataclass
class GroupQuantizedScoreResult:
    """Result of per-group INT8 quantized block scoring.

    Attributes:
        scores: Dequantized scores in float32, shape [B, H, Q, K].
        integer_dot: Raw integer dot product per group, shape
            [B, H, Q, K, num_groups], dtype int32.
        query_quantized: Per-group quantized query.
        key_quantized: Per-group quantized key.
    """

    scores: torch.Tensor
    integer_dot: torch.Tensor
    query_quantized: GroupQuantizedTensor
    key_quantized: GroupQuantizedTensor


def group_quantized_int8_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    group_size: int,
    scale_by_sqrt_dim: bool = True,
) -> GroupQuantizedScoreResult:
    """Compute INT8 quantized block scores using per-group quantization.

    For each group g: score_g = q_{q,g}^T k_{q,g} * alpha_q_g * alpha_k_g
    Total: score = sum_g score_g, optionally / sqrt(d).

    Args:
        query: Query blocks, shape [B, H, Q, d].
        key: Key blocks, shape [B, H, K, d].
        group_size: Number of coordinates per quantization group.
        scale_by_sqrt_dim: If True, divide by sqrt(d).

    Returns:
        GroupQuantizedScoreResult.

    Raises:
        TypeError: If inputs are not floating-point.
        ValueError: If inputs are invalid.
    """
    _validate_score_inputs(query, key)


    q_quant = quantize_int8_per_group(query, group_size=group_size, dim=-1)
    k_quant = quantize_int8_per_group(key, group_size=group_size, dim=-1)

    d = query.shape[-1]
    group_starts = list(range(0, d, group_size))
    num_groups = len(group_starts)

    # Accumulate per-group scores
    # scores shape: [B, H, Q, K]
    batch, heads, num_q, _ = query.shape
    _, _, num_k, _ = key.shape
    scores = torch.zeros(batch, heads, num_q, num_k, dtype=torch.float32, device=query.device)

    # integer_dot per group: [B, H, Q, K, num_groups]
    integer_dot = torch.zeros(
        batch, heads, num_q, num_k, num_groups,
        dtype=torch.int32, device=query.device,
    )

    q_scale = q_quant.scale  # [B, H, Q, num_groups]
    k_scale = k_quant.scale  # [B, H, K, num_groups]

    for g_idx, start in enumerate(group_starts):
        end = min(start + group_size, d)

        # Slice values for this group
        q_g = q_quant.values[..., start:end].to(torch.int32).to(torch.float32)
        k_g = k_quant.values[..., start:end].to(torch.int32).to(torch.float32)

        # Integer dot product for this group
        z_g = torch.einsum("bhqd,bhkd->bhqk", q_g, k_g).to(torch.int32)
        integer_dot[..., g_idx] = z_g

        # Get scales for this group
        # q_scale[..., g_idx]: [B, H, Q]
        # k_scale[..., g_idx]: [B, H, K]
        q_s = q_scale[..., g_idx].unsqueeze(-1)  # [B, H, Q, 1]
        k_s = k_scale[..., g_idx].unsqueeze(-2)  # [B, H, 1, K]

        # Dequantized score contribution from this group
        scores += z_g.to(torch.float32) * q_s * k_s

    if scale_by_sqrt_dim:
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=torch.float32))
        scores = scores / sqrt_d

    return GroupQuantizedScoreResult(
        scores=scores,
        integer_dot=integer_dot,
        query_quantized=q_quant,
        key_quantized=k_quant,
    )
