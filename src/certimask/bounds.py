"""Score error bounds and interval validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from certimask.quantization import GroupQuantizedTensor, QuantizedTensor


def _get_per_coord_error_bound(
    qt: QuantizedTensor,
    certificate_type: str,
) -> torch.Tensor:
    """Get per-coordinate error bound for a quantized tensor.

    For actual: |x_i - x_tilde_i| (per-element absolute error).
    For analytic: alpha_x / 2 (half the scale), but 0 for zero vectors.

    Args:
        qt: QuantizedTensor.
        certificate_type: "actual" or "analytic".

    Returns:
        Per-coordinate error bound, same shape as qt.dequantized.
    """
    if certificate_type == "actual":
        # Per-element absolute error: |x - x_tilde|, stored during quantization
        return qt.actual_coord_error.to(torch.float32)

    # Analytic: alpha_x / 2 for each coordinate, but 0 for zero vectors
    # scale has shape [..., 1] (keepdim), broadcast to [..., d]
    half_scale = qt.scale.to(torch.float32) / 2.0
    half_scale_expanded = half_scale.expand_as(qt.dequantized)

    # Zero vectors: error is 0, not scale/2 = 0.5
    zero_expanded = qt.is_zero_vector.unsqueeze(-1).expand_as(qt.dequantized)
    return torch.where(zero_expanded, torch.zeros_like(half_scale_expanded), half_scale_expanded)


@dataclass
class ScoreBounds:
    """Certified score error bounds.

    Attributes:
        lower: Lower bound of the score interval, shape [B, H, Q, K].
        upper: Upper bound of the score interval, shape [B, H, Q, K].
        error_bound: The error bound E_{ab}, shape [B, H, Q, K].
    """

    lower: torch.Tensor
    upper: torch.Tensor
    error_bound: torch.Tensor


def compute_score_bounds(
    quantized_scores: torch.Tensor,
    query_quantized: QuantizedTensor,
    key_quantized: QuantizedTensor,
    *,
    certificate_type: Literal["actual", "analytic"] = "analytic",
    scale_by_sqrt_dim: bool = True,
) -> ScoreBounds:
    """Compute certified score error bounds.

    For q_a = tilde_q_a + Delta_q_a and k_b = tilde_k_b + Delta_k_b:

        E_{ab} = ||tilde_q_a||_2 * epsilon_b^k
               + ||tilde_k_b||_2 * epsilon_a^q
               + epsilon_a^q * epsilon_b^k

    where epsilon is either the actual L2 error or the analytic L2 bound.

    Args:
        quantized_scores: Dequantized scores, shape [B, H, Q, K].
        query_quantized: Quantized query from quantize_int8_per_vector.
        key_quantized: Quantized key from quantize_int8_per_vector.
        certificate_type: "actual" uses actual_l2_error, "analytic" uses
            analytic_l2_bound.
        scale_by_sqrt_dim: If True, the scores were divided by sqrt(d),
            so the error bound must also be divided by sqrt(d).

    Returns:
        ScoreBounds with lower, upper, and error_bound tensors.

    Raises:
        ValueError: If certificate_type is invalid or tensors have wrong shapes.
    """
    if certificate_type not in ("actual", "analytic"):
        raise ValueError(
            f"certificate_type must be 'actual' or 'analytic', got '{certificate_type}'"
        )

    # Get L2 norms of dequantized query/key vectors
    # query_quantized.dequantized: [B, H, Q, d]
    # key_quantized.dequantized: [B, H, K, d]
    query_norm = torch.linalg.vector_norm(
        query_quantized.dequantized.to(torch.float32), ord=2, dim=-1
    )  # [B, H, Q]
    key_norm = torch.linalg.vector_norm(
        key_quantized.dequantized.to(torch.float32), ord=2, dim=-1
    )  # [B, H, K]

    # Get error bounds (actual or analytic)
    if certificate_type == "actual":
        query_error = query_quantized.actual_l2_error.to(torch.float32)  # [B, H, Q]
        key_error = key_quantized.actual_l2_error.to(torch.float32)  # [B, H, K]
    else:
        query_error = query_quantized.analytic_l2_bound.to(torch.float32)  # [B, H, Q]
        key_error = key_quantized.analytic_l2_bound.to(torch.float32)  # [B, H, K]

    # Broadcast to [B, H, Q, K]:
    # query_norm[..., :, None] -> [B, H, Q, 1]
    # key_norm[..., None, :] -> [B, H, 1, K]
    # E_{ab} = ||tilde_q|| * eps_k + ||tilde_k|| * eps_q + eps_q * eps_k
    error_bound = (
        query_norm[..., :, None] * key_error[..., None, :]
        + key_norm[..., None, :] * query_error[..., :, None]
        + query_error[..., :, None] * key_error[..., None, :]
    )

    if scale_by_sqrt_dim:
        d = query_quantized.dequantized.shape[-1]
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=torch.float32))
        error_bound = error_bound / sqrt_d

    lower = quantized_scores - error_bound
    upper = quantized_scores + error_bound

    return ScoreBounds(
        lower=lower,
        upper=upper,
        error_bound=error_bound,
    )


def validate_score_bounds(
    reference: torch.Tensor,
    bounds: ScoreBounds,
    *,
    atol: float = 1e-5,
) -> torch.Tensor:
    """Validate that reference scores fall within certified bounds.

    Args:
        reference: FP32 reference scores, shape [B, H, Q, K].
        bounds: ScoreBounds from compute_score_bounds.
        atol: Absolute tolerance for boundary checking.

    Returns:
        Boolean violation mask, shape [B, H, Q, K].
        True means the reference score is outside the certified interval.

    Raises:
        ValueError: If shapes don't match or tensors contain NaN/Inf.
    """
    if reference.shape != bounds.lower.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs bounds {bounds.lower.shape}"
        )

    ref_f32 = reference.to(torch.float32)

    if torch.isnan(ref_f32).any():
        raise ValueError("reference tensor contains NaN")
    if torch.isinf(ref_f32).any():
        raise ValueError("reference tensor contains Inf")

    if torch.isnan(bounds.lower).any() or torch.isnan(bounds.upper).any():
        raise ValueError("bounds tensor contains NaN")
    if torch.isinf(bounds.lower).any() or torch.isinf(bounds.upper).any():
        raise ValueError("bounds tensor contains Inf")

    violations = (ref_f32 < bounds.lower - atol) | (ref_f32 > bounds.upper + atol)
    return violations


def compute_groupwise_score_bounds(
    quantized_scores: torch.Tensor,
    query_quantized: QuantizedTensor,
    key_quantized: QuantizedTensor,
    *,
    group_size: int,
    certificate_type: Literal["actual", "analytic"],
    scale_by_sqrt_dim: bool = True,
) -> ScoreBounds:
    """Compute score bounds using groupwise L2 error propagation.

    Splits head_dim into groups of consecutive coordinates. For each group g:

        E_g = ||tilde_q_g||_2 * eps_g^k + ||tilde_k_g||_2 * eps_g^q + eps_g^q * eps_g^k

    Total bound: E = sum_g E_g.

    Args:
        quantized_scores: Dequantized scores, shape [B, H, Q, K].
        query_quantized: Quantized query.
        key_quantized: Quantized key.
        group_size: Number of coordinates per group.
        certificate_type: "actual" or "analytic".
        scale_by_sqrt_dim: If True, divide by sqrt(d).

    Returns:
        ScoreBounds with groupwise error bounds.

    Raises:
        ValueError: If group_size is invalid.
    """
    if certificate_type not in ("actual", "analytic"):
        raise ValueError(
            f"certificate_type must be 'actual' or 'analytic', got '{certificate_type}'"
        )
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    q_deq = query_quantized.dequantized.to(torch.float32)
    k_deq = key_quantized.dequantized.to(torch.float32)
    d = q_deq.shape[-1]

    q_err = _get_per_coord_error_bound(query_quantized, certificate_type)
    k_err = _get_per_coord_error_bound(key_quantized, certificate_type)

    # Accumulate groupwise error bounds
    error_bound = torch.zeros(
        q_deq.shape[0], q_deq.shape[1], q_deq.shape[2], k_deq.shape[2],
        dtype=torch.float32, device=q_deq.device,
    )

    for start in range(0, d, group_size):
        end = min(start + group_size, d)

        q_g = q_deq[..., start:end]  # [B, H, Q, g]
        k_g = k_deq[..., start:end]  # [B, H, K, g]
        q_err_g = q_err[..., start:end]  # [B, H, Q, g]
        k_err_g = k_err[..., start:end]  # [B, H, K, g]

        # L2 norms per group
        q_norm_g = torch.linalg.vector_norm(q_g, ord=2, dim=-1)  # [B, H, Q]
        k_norm_g = torch.linalg.vector_norm(k_g, ord=2, dim=-1)  # [B, H, K]

        # Per-vector group error: L2 norm of per-coordinate errors
        q_eps_g = torch.linalg.vector_norm(q_err_g, ord=2, dim=-1)  # [B, H, Q]
        k_eps_g = torch.linalg.vector_norm(k_err_g, ord=2, dim=-1)  # [B, H, K]

        # E_g = ||q_g|| * eps_k_g + ||k_g|| * eps_q_g + eps_q_g * eps_k_g
        error_bound += (
            q_norm_g[..., :, None] * k_eps_g[..., None, :]
            + k_norm_g[..., None, :] * q_eps_g[..., :, None]
            + q_eps_g[..., :, None] * k_eps_g[..., None, :]
        )

    if scale_by_sqrt_dim:
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=torch.float32))
        error_bound = error_bound / sqrt_d

    lower = quantized_scores - error_bound
    upper = quantized_scores + error_bound

    return ScoreBounds(lower=lower, upper=upper, error_bound=error_bound)


def _get_group_per_coord_error(
    qt: GroupQuantizedTensor,
    certificate_type: str,
) -> torch.Tensor:
    """Get per-coordinate error bound for a GroupQuantizedTensor.

    For actual: stored per-coordinate |x - dequantized|.
    For analytic: alpha_g / 2 per coordinate, 0 for zero groups.
    """
    if certificate_type == "actual":
        # We don't store per-coordinate actual error for GroupQuantizedTensor.
        # Compute from values and dequantized: error = |x - deq| is not available.
        # Use the L2 error as a fallback (less tight but correct).
        # For coordinate actual, we need the original tensor.
        # Since we can't recover it, raise an error.
        raise ValueError(
            "Per-coordinate actual error not available for GroupQuantizedTensor. "
            "Use certificate_type='analytic' or compute from original tensor."
        )

    # Analytic: alpha_g / 2 per coordinate, 0 for zero groups
    # scale shape: [..., num_groups]
    # Need to expand to [..., d] per coordinate
    d = qt.values.shape[-1]
    num_groups = qt.scale.shape[-1]
    gs = qt.group_size

    half_scale_expanded = torch.zeros_like(qt.dequantized)
    zero_expanded = torch.zeros_like(qt.dequantized, dtype=torch.bool)

    for g_idx in range(num_groups):
        start = g_idx * gs
        end = min(start + gs, d)

        slc = [slice(None)] * qt.dequantized.dim()
        slc[-1] = slice(start, end)

        scale_slc: list[Any] = [slice(None)] * qt.scale.dim()
        scale_slc[-1] = g_idx

        half_s = qt.scale[tuple(scale_slc)] / 2.0
        is_zg = qt.is_zero_group[tuple(scale_slc)]

        # Expand to the group slice shape
        half_s_exp = half_s.unsqueeze(-1).expand_as(qt.dequantized[tuple(slc)])
        is_zg_exp = is_zg.unsqueeze(-1).expand_as(qt.dequantized[tuple(slc)])

        half_scale_expanded[tuple(slc)] = torch.where(
            is_zg_exp, torch.zeros_like(half_s_exp), half_s_exp,
        )
        zero_expanded[tuple(slc)] = is_zg_exp

    return half_scale_expanded


def compute_group_quantized_coordinate_bounds(
    quantized_scores: torch.Tensor,
    query_quantized: GroupQuantizedTensor,
    key_quantized: GroupQuantizedTensor,
    *,
    certificate_type: Literal["actual", "analytic"] = "analytic",
    scale_by_sqrt_dim: bool = True,
) -> ScoreBounds:
    """Compute coordinate-wise bounds using per-group quantization.

    E = sum_i (|tilde_q_i| * b_i^k + |tilde_k_i| * b_i^q + b_i^q * b_i^k)

    Args:
        quantized_scores: Dequantized scores, shape [B, H, Q, K].
        query_quantized: Per-group quantized query.
        key_quantized: Per-group quantized key.
        certificate_type: "actual" or "analytic".
        scale_by_sqrt_dim: If True, divide by sqrt(d).

    Returns:
        ScoreBounds.
    """
    if certificate_type not in ("actual", "analytic"):
        raise ValueError(
            f"certificate_type must be 'actual' or 'analytic', got '{certificate_type}'"
        )

    q_deq = query_quantized.dequantized.to(torch.float32)
    k_deq = key_quantized.dequantized.to(torch.float32)
    d = q_deq.shape[-1]

    q_err = _get_group_per_coord_error(query_quantized, certificate_type)
    k_err = _get_group_per_coord_error(key_quantized, certificate_type)

    q_abs = q_deq.abs()
    k_abs = k_deq.abs()

    coord_error = (
        q_abs[..., :, None, :] * k_err[..., None, :, :]
        + k_abs[..., None, :, :] * q_err[..., :, None, :]
        + q_err[..., :, None, :] * k_err[..., None, :, :]
    )

    error_bound = coord_error.sum(dim=-1)

    if scale_by_sqrt_dim:
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=torch.float32))
        error_bound = error_bound / sqrt_d

    lower = quantized_scores - error_bound
    upper = quantized_scores + error_bound

    return ScoreBounds(lower=lower, upper=upper, error_bound=error_bound)


def compute_k_only_per_vector_bounds(
    quantized_scores: torch.Tensor,
    query: torch.Tensor,
    key_quantized: QuantizedTensor,
    *,
    certificate_type: Literal["actual", "analytic"] = "analytic",
    scale_by_sqrt_dim: bool = True,
) -> ScoreBounds:
    """Compute K-only per-vector bounds.

    Q is exact (float32), only K is quantized.

    E = ||q||_2 * eps_k

    Args:
        quantized_scores: K-only dequantized scores, shape [B, H, Q, K].
        query: Original float32 query, shape [B, H, Q, d].
        key_quantized: Quantized key from quantize_int8_per_vector.
        certificate_type: "actual" or "analytic".
        scale_by_sqrt_dim: If True, divide by sqrt(d).

    Returns:
        ScoreBounds.
    """
    if certificate_type not in ("actual", "analytic"):
        raise ValueError(
            f"certificate_type must be 'actual' or 'analytic', got '{certificate_type}'"
        )

    q_f32 = query.to(torch.float32)
    d = q_f32.shape[-1]

    # ||q||_2 per query vector
    q_norm = torch.linalg.vector_norm(q_f32, ord=2, dim=-1)  # [B, H, Q]

    # K error per key vector
    if certificate_type == "actual":
        k_error = key_quantized.actual_l2_error.to(torch.float32)  # [B, H, K]
    else:
        k_error = key_quantized.analytic_l2_bound.to(torch.float32)  # [B, H, K]

    # E = ||q|| * eps_k
    error_bound = q_norm[..., :, None] * k_error[..., None, :]  # [B, H, Q, K]

    if scale_by_sqrt_dim:
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=torch.float32))
        error_bound = error_bound / sqrt_d

    lower = quantized_scores - error_bound
    upper = quantized_scores + error_bound

    return ScoreBounds(lower=lower, upper=upper, error_bound=error_bound)


def compute_k_only_per_group_bounds(
    quantized_scores: torch.Tensor,
    query: torch.Tensor,
    key_quantized: GroupQuantizedTensor,
    *,
    certificate_type: Literal["actual", "analytic"] = "analytic",
    scale_by_sqrt_dim: bool = True,
) -> ScoreBounds:
    """Compute K-only per-group coordinate bounds.

    Q is exact (float32), only K is quantized per-group.

    E = sum_i |q_i| * b_i^k

    Args:
        quantized_scores: K-only dequantized scores, shape [B, H, Q, K].
        query: Original float32 query, shape [B, H, Q, d].
        key_quantized: GroupQuantizedTensor for the key.
        certificate_type: "actual" or "analytic".
        scale_by_sqrt_dim: If True, divide by sqrt(d).

    Returns:
        ScoreBounds.
    """
    if certificate_type not in ("actual", "analytic"):
        raise ValueError(
            f"certificate_type must be 'actual' or 'analytic', got '{certificate_type}'"
        )

    q_f32 = query.to(torch.float32)
    d = q_f32.shape[-1]

    # Per-coordinate K error bound
    k_err = _get_group_per_coord_error(key_quantized, certificate_type)  # [B, H, K, d]

    # E = sum_i |q_i| * b_i^k
    # q_f32: [B, H, Q, d] -> [B, H, Q, 1, d]
    # k_err: [B, H, K, d] -> [B, H, 1, K, d]
    coord_error = q_f32.abs()[..., :, None, :] * k_err[..., None, :, :]  # [B, H, Q, K, d]
    error_bound = coord_error.sum(dim=-1)  # [B, H, Q, K]

    if scale_by_sqrt_dim:
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=torch.float32))
        error_bound = error_bound / sqrt_d

    lower = quantized_scores - error_bound
    upper = quantized_scores + error_bound

    return ScoreBounds(lower=lower, upper=upper, error_bound=error_bound)


def compute_coordinate_score_bounds(
    quantized_scores: torch.Tensor,
    query_quantized: QuantizedTensor,
    key_quantized: QuantizedTensor,
    *,
    certificate_type: Literal["actual", "analytic"],
    scale_by_sqrt_dim: bool = True,
) -> ScoreBounds:
    """Compute score bounds using coordinate-wise error propagation.

    For each coordinate i:
        |q_i * k_i - tilde_q_i * tilde_k_i|
            <= |tilde_q_i| * b_i^k + |tilde_k_i| * b_i^q + b_i^q * b_i^k

    Total: E = sum_i (...).

    Args:
        quantized_scores: Dequantized scores, shape [B, H, Q, K].
        query_quantized: Quantized query.
        key_quantized: Quantized key.
        certificate_type: "actual" or "analytic".
        scale_by_sqrt_dim: If True, divide by sqrt(d).

    Returns:
        ScoreBounds with coordinate-wise error bounds.
    """
    if certificate_type not in ("actual", "analytic"):
        raise ValueError(
            f"certificate_type must be 'actual' or 'analytic', got '{certificate_type}'"
        )

    q_deq = query_quantized.dequantized.to(torch.float32)
    k_deq = key_quantized.dequantized.to(torch.float32)
    d = q_deq.shape[-1]

    q_err = _get_per_coord_error_bound(query_quantized, certificate_type)  # [B, H, Q, d]
    k_err = _get_per_coord_error_bound(key_quantized, certificate_type)    # [B, H, K, d]

    # Per-coordinate: |tilde_q_i| * b_i^k + |tilde_k_i| * b_i^q + b_i^q * b_i^k
    # Shape: [B, H, Q, d] and [B, H, K, d]
    q_abs = q_deq.abs()  # [B, H, Q, d]
    k_abs = k_deq.abs()  # [B, H, K, d]

    # Per-coordinate error for query-key pairs
    # q_abs[..., :, None, :] -> [B, H, Q, 1, d]
    # k_err[..., None, :, :] -> [B, H, 1, K, d]
    coord_error = (
        q_abs[..., :, None, :] * k_err[..., None, :, :]
        + k_abs[..., None, :, :] * q_err[..., :, None, :]
        + q_err[..., :, None, :] * k_err[..., None, :, :]
    )  # [B, H, Q, K, d]

    # Sum over coordinates
    error_bound = coord_error.sum(dim=-1)  # [B, H, Q, K]

    if scale_by_sqrt_dim:
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=torch.float32))
        error_bound = error_bound / sqrt_d

    lower = quantized_scores - error_bound
    upper = quantized_scores + error_bound

    return ScoreBounds(lower=lower, upper=upper, error_bound=error_bound)
