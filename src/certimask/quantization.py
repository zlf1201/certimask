"""Symmetric per-vector INT8 quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class QuantizedTensor:
    """Result of symmetric per-vector INT8 quantization.

    Attributes:
        values: INT8 quantized integers, shape matches input.
        scale: Per-vector scale factor, shape with quantized dim reduced to 1.
        dequantized: Dequantized floating-point tensor, same shape as input.
        actual_l2_error: L2 norm of quantization error per vector.
        analytic_l2_bound: Analytic upper bound on L2 error per vector.
        is_zero_vector: Boolean mask indicating all-zero input vectors.
            Shape matches scale (quantized dim reduced to 1). For zero
            vectors, the per-coordinate analytic error bound must be 0,
            not scale/2 = 0.5.
    """

    values: torch.Tensor
    scale: torch.Tensor
    dequantized: torch.Tensor
    actual_l2_error: torch.Tensor
    analytic_l2_bound: torch.Tensor
    is_zero_vector: torch.Tensor
    actual_coord_error: torch.Tensor


def quantize_int8_per_vector(
    x: torch.Tensor,
    dim: int = -1,
) -> QuantizedTensor:
    """Perform symmetric per-vector INT8 quantization along the given dimension.

    Args:
        x: Input floating-point tensor.
        dim: Dimension along which to quantize (default: -1, last dimension).

    Returns:
        QuantizedTensor with quantized values, scale, dequantized result,
        actual L2 error, and analytic L2 bound.

    Raises:
        TypeError: If input is not a floating-point tensor.
        ValueError: If input contains NaN, Inf, is empty, or dim is out of range.
    """
    _validate_input(x, dim)

    # Work in float32 for numerical stability
    x_f32 = x.to(torch.float32)
    d = x.shape[dim]

    # Compute scale: alpha_x = max(|x|) / 127 along dim
    max_abs = x_f32.abs().amax(dim=dim, keepdim=True)
    scale = max_abs / 127.0

    # Handle all-zero vectors: set scale to 1.0
    zero_mask = max_abs == 0.0
    scale = torch.where(zero_mask, torch.ones_like(scale), scale)

    # Quantize: clip(round(x / scale), -127, 127)
    x_scaled = x_f32 / scale
    x_rounded = torch.round(x_scaled)
    x_clipped = torch.clamp(x_rounded, -127, 127)
    values = x_clipped.to(torch.int8)

    # Dequantize
    dequantized = x_clipped * scale
    dequantized = dequantized.squeeze() if x_f32.dim() == 1 else dequantized

    # Actual L2 error
    error = x_f32 - dequantized
    actual_l2_error = torch.linalg.vector_norm(error, ord=2, dim=dim)

    # Analytic L2 bound: sqrt(d) * scale / 2
    # Squeeze the keepdim so shape matches actual_l2_error
    scale_squeezed = scale.squeeze(dim)
    sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=torch.float32))
    analytic_l2_bound = sqrt_d * scale_squeezed / 2.0

    # Identify zero vectors for correct per-coordinate analytic bound
    is_zero_vector = zero_mask.squeeze(dim) if zero_mask.dim() > 1 else zero_mask

    # Per-coordinate actual error: |x - dequantized|
    actual_coord_error = (x_f32 - dequantized).abs()

    return QuantizedTensor(
        values=values,
        scale=scale,
        dequantized=dequantized,
        actual_l2_error=actual_l2_error,
        analytic_l2_bound=analytic_l2_bound,
        is_zero_vector=is_zero_vector,
        actual_coord_error=actual_coord_error,
    )


@dataclass
class GroupQuantizedTensor:
    """Result of symmetric per-group INT8 quantization.

    Attributes:
        values: INT8 quantized integers, same shape as input.
        scale: Per-group scale factor. Shape is input with the quantized
            dim replaced by num_groups (may need unsqueeze for broadcasting).
        dequantized: Dequantized floating-point tensor, same shape as input.
        actual_l2_error: L2 error per group.
        analytic_l2_bound: Analytic L2 bound per group: sqrt(len) * scale / 2.
        group_size: Requested group size.
        group_lengths: Actual length of each group (last may be shorter).
        is_zero_group: Boolean mask for all-zero groups.
    """

    values: torch.Tensor
    scale: torch.Tensor
    dequantized: torch.Tensor
    actual_l2_error: torch.Tensor
    analytic_l2_bound: torch.Tensor
    group_size: int
    group_lengths: torch.Tensor
    is_zero_group: torch.Tensor


def quantize_int8_per_group(
    x: torch.Tensor,
    *,
    group_size: int,
    dim: int = -1,
) -> GroupQuantizedTensor:
    """Perform symmetric per-group INT8 quantization along the given dimension.

    Args:
        x: Input floating-point tensor.
        group_size: Number of coordinates per quantization group.
        dim: Dimension along which to quantize (default: -1).

    Returns:
        GroupQuantizedTensor with quantized values, scale, etc.

    Raises:
        TypeError: If input is not floating-point.
        ValueError: If input is empty, contains NaN/Inf, dim is out of
            range, or group_size is not positive.
    """
    _validate_input(x, dim)

    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    x_f32 = x.to(torch.float32)
    d = x.shape[dim]

    # Compute group boundaries
    group_starts = list(range(0, d, group_size))
    num_groups = len(group_starts)
    group_lengths = torch.tensor(
        [min(group_size, d - s) for s in group_starts],
        dtype=torch.long,
        device=x.device,
    )

    # Prepare output tensors
    # scale shape: input with dim replaced by num_groups
    scale_shape = list(x_f32.shape)
    scale_shape[dim] = num_groups
    scales = torch.zeros(scale_shape, dtype=torch.float32, device=x.device)

    # Quantize each group
    values = torch.zeros_like(x_f32).to(torch.int8)
    dequantized = torch.zeros_like(x_f32)
    actual_errors = torch.zeros_like(x_f32)
    analytic_bounds = torch.zeros_like(x_f32)
    is_zero_group = torch.zeros(scale_shape, dtype=torch.bool, device=x.device)

    for g_idx, start in enumerate(group_starts):
        end = min(start + group_size, d)

        # Slice along dim
        slc = [slice(None)] * x_f32.dim()
        slc[dim] = slice(start, end)
        x_g = x_f32[tuple(slc)]

        # Scale for this group
        max_abs = x_g.abs().amax(dim=dim, keepdim=True)
        zero_mask_g = max_abs == 0.0
        scale_g = torch.where(zero_mask_g, torch.ones_like(max_abs), max_abs / 127.0)

        # Quantize
        x_scaled = x_g / scale_g
        x_rounded = torch.round(x_scaled)
        x_clipped = torch.clamp(x_rounded, -127, 127)
        values_g = x_clipped.to(torch.int8)

        # Dequantize
        deq_g = x_clipped * scale_g

        # Store results
        values[tuple(slc)] = values_g
        dequantized[tuple(slc)] = deq_g
        actual_errors[tuple(slc)] = (x_f32[tuple(slc)] - deq_g).abs()

        # Analytic bound per coordinate: alpha_g / 2, but 0 for zero groups
        half_scale = scale_g / 2.0
        half_scale = torch.where(zero_mask_g, torch.zeros_like(half_scale), half_scale)
        analytic_bounds[tuple(slc)] = half_scale.expand_as(x_g)

        # Store scale for this group
        scale_slc: list[Any] = [slice(None)] * len(scale_shape)
        scale_slc[dim] = g_idx
        scales[tuple(scale_slc)] = scale_g.squeeze(dim)

        # Store zero group flag
        is_zero_group[tuple(scale_slc)] = zero_mask_g.squeeze(dim)

    # Compute per-group L2 errors
    # Reshape to [..., num_groups, group_size] for L2 computation
    # (handle last group being shorter)
    actual_l2_errors = torch.zeros(scale_shape, dtype=torch.float32, device=x.device)
    analytic_l2_bounds = torch.zeros(scale_shape, dtype=torch.float32, device=x.device)

    for g_idx, start in enumerate(group_starts):
        end = min(start + group_size, d)
        gl = end - start

        slc = [slice(None)] * x_f32.dim()
        slc[dim] = slice(start, end)

        err_g = actual_errors[tuple(slc)]
        actual_l2_errors_g = torch.linalg.vector_norm(err_g, ord=2, dim=dim)

        scale_slc2: list[Any] = [slice(None)] * len(scale_shape)
        scale_slc2[dim] = g_idx
        actual_l2_errors[tuple(scale_slc2)] = actual_l2_errors_g

        # Analytic L2 bound: sqrt(gl) * scale / 2
        sqrt_gl = torch.sqrt(torch.tensor(float(gl), dtype=torch.float32))
        scale_g = scales[tuple(scale_slc2)]
        is_zg = is_zero_group[tuple(scale_slc2)]
        analytic_l2_bounds[tuple(scale_slc2)] = torch.where(
            is_zg,
            torch.zeros_like(scale_g),
            sqrt_gl * scale_g / 2.0,
        )

    return GroupQuantizedTensor(
        values=values,
        scale=scales,
        dequantized=dequantized,
        actual_l2_error=actual_l2_errors,
        analytic_l2_bound=analytic_l2_bounds,
        group_size=group_size,
        group_lengths=group_lengths,
        is_zero_group=is_zero_group,
    )


def _validate_input(x: torch.Tensor, dim: int) -> None:
    """Validate input tensor and dimension.

    Raises:
        TypeError: If input is not a floating-point tensor.
        ValueError: If input contains NaN, Inf, is empty, or dim is out of range.
    """
    if not x.is_floating_point():
        raise TypeError(f"Input must be a floating-point tensor, got {x.dtype}")

    if x.numel() == 0:
        raise ValueError("Input tensor is empty")

    if torch.isnan(x).any():
        raise ValueError("Input tensor contains NaN")

    if torch.isinf(x).any():
        raise ValueError("Input tensor contains Inf")

    if dim < -x.dim() or dim >= x.dim():
        raise ValueError(
            f"dim {dim} is out of range for tensor with {x.dim()} dimensions"
        )
