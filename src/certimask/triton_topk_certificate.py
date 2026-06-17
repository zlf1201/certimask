# ruff: noqa: N803, N806
"""Triton fused top-k partition certificate kernel.

Computes partition-aware certification decisions on GPU:
  For each query row:
    U_R_max = max upper over rejected valid candidates
    L_T_min = min lower over selected valid candidates
    Selected tile: KEEP if L_t > U_R_max else AMBIGUOUS
    Rejected tile: DROP if U_r < L_T_min else AMBIGUOUS
    Invalid tile: INVALID

Supports:
    - block_size = 8, group_size = 4, D = 64
    - partition ambiguity mode only
"""

from __future__ import annotations

import torch

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# Decision codes (must match topk_certificate.py)
_DROP = 0
_KEEP = 1
_AMBIGUOUS = 2
_INVALID = 3


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _topk_certificate_kernel(  # type: ignore[no-untyped-def]
        Lower_ptr,
        Upper_ptr,
        Selected_ptr,
        Valid_ptr,
        Out_dec_ptr,
        Out_amb_ptr,
        # Strides
        stride_bhqk_b,
        stride_bhqk_h,
        stride_bhqk_q,
        # Shapes
        B,
        H,
        Q_blk,
        K_blk,
        # Constexpr decision codes
        DEC_DROP: tl.constexpr,
        DEC_KEEP: tl.constexpr,
        DEC_AMBIGUOUS: tl.constexpr,
        DEC_INVALID: tl.constexpr,
    ):
        """Partition certificate kernel: one program per query row.

        First pass: compute U_R_max (max upper over rejected valid) and
                    L_T_min (min lower over selected valid).
        Second pass: classify each tile.
        """
        pid = tl.program_id(0)

        # Decompose pid -> (b, h, q)
        q_idx = pid % Q_blk
        h_idx = (pid // Q_blk) % H
        b_idx = pid // (Q_blk * H)

        row_base = (
            b_idx * stride_bhqk_b
            + h_idx * stride_bhqk_h
            + q_idx * stride_bhqk_q
        )

        # Pass 1: compute U_R_max and L_T_min
        POS_INF: tl.constexpr = 3.4e38  # noqa: N806
        NEG_INF: tl.constexpr = -3.4e38  # noqa: N806

        u_r_max = NEG_INF  # max upper over rejected valid tiles
        l_t_min = POS_INF  # min lower over selected valid tiles

        for k_off in range(0, K_blk, 1):
            off = row_base + k_off
            is_valid = tl.load(Valid_ptr + off).to(tl.int8) != 0
            is_selected = tl.load(Selected_ptr + off).to(tl.int8) != 0

            if is_valid:
                lo = tl.load(Lower_ptr + off)
                hi = tl.load(Upper_ptr + off)

                if is_selected:
                    # Update L_T_min
                    l_t_min = tl.minimum(l_t_min, lo)
                else:
                    # Update U_R_max
                    u_r_max = tl.maximum(u_r_max, hi)

        # Pass 2: classify each tile
        for k_off in range(0, K_blk, 1):
            off = row_base + k_off
            is_valid = tl.load(Valid_ptr + off).to(tl.int8) != 0
            is_selected = tl.load(Selected_ptr + off).to(tl.int8) != 0

            if is_valid:
                lo = tl.load(Lower_ptr + off)
                hi = tl.load(Upper_ptr + off)

                if is_selected:
                    # Selected tile: KEEP if L_t > U_R_max
                    if lo > u_r_max:
                        tl.store(Out_dec_ptr + off, DEC_KEEP)
                        tl.store(Out_amb_ptr + off, 0)
                    else:
                        tl.store(Out_dec_ptr + off, DEC_AMBIGUOUS)
                        tl.store(Out_amb_ptr + off, 1)
                else:
                    # Rejected tile: DROP if U_r < L_T_min
                    if hi < l_t_min:
                        tl.store(Out_dec_ptr + off, DEC_DROP)
                        tl.store(Out_amb_ptr + off, 0)
                    else:
                        tl.store(Out_dec_ptr + off, DEC_AMBIGUOUS)
                        tl.store(Out_amb_ptr + off, 1)
            else:
                # Invalid tile
                tl.store(Out_dec_ptr + off, DEC_INVALID)
                tl.store(Out_amb_ptr + off, 0)


def _check_triton_available() -> None:
    """Raise if CUDA or Triton is not available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")


def triton_certified_topk_mask_partition(
    lower_scores: torch.Tensor,
    upper_scores: torch.Tensor,
    selected_reference_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated partition certificate.

    Computes per-tile decisions using the partition mode:
      - Selected tile: KEEP if L_t > U_R_max, else AMBIGUOUS
      - Rejected tile: DROP if U_r < L_T_min, else AMBIGUOUS
      - Invalid tile: INVALID

    Args:
        lower_scores: Lower bound scores [B, H, Q_blk, K_blk].
        upper_scores: Upper bound scores [B, H, Q_blk, K_blk].
        selected_reference_mask: Boolean FP reference selection [B, H, Q_blk, K_blk].
        valid_mask: Boolean causal valid mask [B, H, Q_blk, K_blk].

    Returns:
        decisions: Per-tile decision codes [B, H, Q_blk, K_blk] (int64).
        ambiguous: Boolean ambiguous mask [B, H, Q_blk, K_blk].

    Raises:
        RuntimeError: If CUDA or Triton is not available.
    """
    _check_triton_available()

    assert lower_scores.is_cuda, "lower_scores must be on CUDA"
    assert upper_scores.is_cuda, "upper_scores must be on CUDA"
    assert selected_reference_mask.is_cuda, "selected_reference_mask must be on CUDA"
    assert valid_mask.is_cuda, "valid_mask must be on CUDA"
    assert (
        lower_scores.shape == upper_scores.shape
        == selected_reference_mask.shape == valid_mask.shape
    )

    batch, heads, q_blk, k_blk = lower_scores.shape
    device = lower_scores.device

    # Ensure contiguous
    lower_scores = lower_scores.contiguous()
    upper_scores = upper_scores.contiguous()
    selected_i8 = selected_reference_mask.to(torch.int8).contiguous()
    valid_i8 = valid_mask.to(torch.int8).contiguous()

    # Output tensors
    out_dec = torch.empty(batch, heads, q_blk, k_blk, dtype=torch.int64, device=device)
    out_amb = torch.empty(batch, heads, q_blk, k_blk, dtype=torch.bool, device=device)

    # Strides (contiguous layout)
    stride_b = heads * q_blk * k_blk
    stride_h = q_blk * k_blk
    stride_q = k_blk

    # Grid: one program per query row
    grid = (batch * heads * q_blk,)

    _topk_certificate_kernel[grid](
        lower_scores,
        upper_scores,
        selected_i8,
        valid_i8,
        out_dec,
        out_amb,
        stride_b,
        stride_h,
        stride_q,
        batch,
        heads,
        q_blk,
        k_blk,
        DEC_DROP=_DROP,
        DEC_KEEP=_KEEP,
        DEC_AMBIGUOUS=_AMBIGUOUS,
        DEC_INVALID=_INVALID,
    )

    return out_dec, out_amb
