# ruff: noqa: N803, N806
"""Triton kernels for AGLR-C CertiMask sampled scoring and interval computation.

Supports:
    - block_size = 8
    - group_size = 4
    - sample_pattern = both_diagonals (16 samples)
    - aggregation = logsumexp
    - K-only per-group INT8 quantization
    - Q in FP16 or FP32
"""

from __future__ import annotations

import torch

# Triton is optional — import at module level so the JIT kernel can see `tl`.
try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BLOCK_SIZE = 8
_NUM_SAMPLES = 16
_D_FIXED = 64
_GROUP_SIZE = 4


# ---------------------------------------------------------------------------
# Triton kernel (defined at module level so `tl` is in scope)
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _aglr_certimask_logsumexp_kernel(  # type: ignore[no-untyped-def]
        # Pointers
        Q_ptr,
        K_int8_ptr,
        K_scale_ptr,
        K_zero_ptr,
        Valid_ptr,
        Out_quant_ptr,
        Out_lower_ptr,
        Out_upper_ptr,
        # Strides for [B, H, L, D] tensors
        stride_bhld_b,
        stride_bhld_h,
        stride_bhld_l,
        # Strides for [B, H, Q_blk, K_blk] tensors
        stride_bhqk_b,
        stride_bhqk_h,
        stride_bhqk_q,
        # Shapes
        B,
        H,
        Q_blk,
        K_blk,
        # Constexprs
        BLOCK_SIZE: tl.constexpr,
        D_SIZE: tl.constexpr,
        NUM_SAMPLES: tl.constexpr,
        SQRT_D: tl.constexpr,
    ):
        """AGLR-C CertiMask logsumexp scoring kernel.

        Each program computes one output tile (b, h, q_blk, k_blk).
        For each of 16 sample positions, computes:
            z_tilde = q^T k_dequant / sqrt(d)
            err     = sum |q_r| * half_scale_r / sqrt(d)   (zero-group safe)
            lower   = z_tilde - err
            upper   = z_tilde + err
        Then aggregates via running logsumexp.
        """
        pid = tl.program_id(0)

        # Decompose program id -> (b, h, q, k)
        k_idx = pid % K_blk
        q_idx = (pid // K_blk) % Q_blk
        h_idx = (pid // (K_blk * Q_blk)) % H
        b_idx = pid // (K_blk * Q_blk * H)

        # Valid mask check
        valid_off = (
            b_idx * stride_bhqk_b
            + h_idx * stride_bhqk_h
            + q_idx * stride_bhqk_q
            + k_idx
        )
        is_valid = tl.load(Valid_ptr + valid_off).to(tl.int8) != 0

        out_off = valid_off  # same layout

        if not is_valid:
            NEG_INF: tl.constexpr = -3.4e38  # noqa: N806
            tl.store(Out_quant_ptr + out_off, NEG_INF)
            tl.store(Out_lower_ptr + out_off, NEG_INF)
            tl.store(Out_upper_ptr + out_off, NEG_INF)
            return

        # Base offsets for the Q and K blocks
        q_base = (
            b_idx * stride_bhld_b
            + h_idx * stride_bhld_h
            + q_idx * BLOCK_SIZE * stride_bhld_l
        )
        k_base = (
            b_idx * stride_bhld_b
            + h_idx * stride_bhld_h
            + k_idx * BLOCK_SIZE * stride_bhld_l
        )

        d_offs = tl.arange(0, D_SIZE)

        # Running logsumexp accumulators
        NEG_INF_VAL: tl.constexpr = -3.4e38  # noqa: N806
        max_quant = NEG_INF_VAL
        sum_exp_quant = 0.0
        max_lower = NEG_INF_VAL
        sum_exp_lower = 0.0
        max_upper = NEG_INF_VAL
        sum_exp_upper = 0.0

        # Unrolled sample loop for both_diagonals with block_size=8
        # 16 samples: (qi, ki) pairs
        # (0,0), (0,7), (1,1), (1,6), (2,2), (2,5), (3,3), (3,4),
        # (4,4), (4,3), (5,5), (5,2), (6,6), (6,1), (7,7), (7,0)

        # ---- sample 0: qi=0, ki=0 ----
        q_vec = tl.load(Q_ptr + q_base + 0 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 0 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 0 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 0 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 1: qi=0, ki=7 ----
        q_vec = tl.load(Q_ptr + q_base + 0 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 7 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 7 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 7 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 2: qi=1, ki=1 ----
        q_vec = tl.load(Q_ptr + q_base + 1 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 1 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 1 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 1 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 3: qi=1, ki=6 ----
        q_vec = tl.load(Q_ptr + q_base + 1 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 6 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 6 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 6 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 4: qi=2, ki=2 ----
        q_vec = tl.load(Q_ptr + q_base + 2 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 2 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 2 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 2 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 5: qi=2, ki=5 ----
        q_vec = tl.load(Q_ptr + q_base + 2 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 5 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 5 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 5 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 6: qi=3, ki=3 ----
        q_vec = tl.load(Q_ptr + q_base + 3 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 3 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 3 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 3 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 7: qi=3, ki=4 ----
        q_vec = tl.load(Q_ptr + q_base + 3 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 4 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 4 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 4 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 8: qi=4, ki=4 ----
        q_vec = tl.load(Q_ptr + q_base + 4 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 4 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 4 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 4 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 9: qi=4, ki=3 ----
        q_vec = tl.load(Q_ptr + q_base + 4 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 3 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 3 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 3 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 10: qi=5, ki=5 ----
        q_vec = tl.load(Q_ptr + q_base + 5 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 5 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 5 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 5 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 11: qi=5, ki=2 ----
        q_vec = tl.load(Q_ptr + q_base + 5 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 2 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 2 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 2 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 12: qi=6, ki=6 ----
        q_vec = tl.load(Q_ptr + q_base + 6 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 6 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 6 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 6 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 13: qi=6, ki=1 ----
        q_vec = tl.load(Q_ptr + q_base + 6 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 1 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 1 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 1 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 14: qi=7, ki=7 ----
        q_vec = tl.load(Q_ptr + q_base + 7 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 7 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 7 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 7 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # ---- sample 15: qi=7, ki=0 ----
        q_vec = tl.load(Q_ptr + q_base + 7 * stride_bhld_l + d_offs).to(tl.float32)
        k_int8 = tl.load(K_int8_ptr + k_base + 0 * stride_bhld_l + d_offs).to(
            tl.float32,
        )
        scale_vec = tl.load(K_scale_ptr + k_base + 0 * stride_bhld_l + d_offs)
        zero_vec = tl.load(K_zero_ptr + k_base + 0 * stride_bhld_l + d_offs).to(tl.int8) != 0
        k_deq = k_int8 * scale_vec
        z = tl.sum(q_vec * k_deq) / SQRT_D
        hs = tl.where(zero_vec, 0.0, scale_vec * 0.5)
        e = tl.sum(tl.abs(q_vec) * hs) / SQRT_D
        lo = z - e
        hi = z + e
        nq = tl.maximum(max_quant, z)
        sum_exp_quant = sum_exp_quant * tl.exp(max_quant - nq) + tl.exp(z - nq)
        max_quant = nq
        nl = tl.maximum(max_lower, lo)
        sum_exp_lower = sum_exp_lower * tl.exp(max_lower - nl) + tl.exp(lo - nl)
        max_lower = nl
        nu = tl.maximum(max_upper, hi)
        sum_exp_upper = sum_exp_upper * tl.exp(max_upper - nu) + tl.exp(hi - nu)
        max_upper = nu

        # Final logsumexp
        quantized_score = tl.log(sum_exp_quant) + max_quant
        lower_score = tl.log(sum_exp_lower) + max_lower
        upper_score = tl.log(sum_exp_upper) + max_upper

        # Ensure lower <= upper
        upper_score = tl.maximum(lower_score, upper_score)

        tl.store(Out_quant_ptr + out_off, quantized_score)
        tl.store(Out_lower_ptr + out_off, lower_score)
        tl.store(Out_upper_ptr + out_off, upper_score)


# ---------------------------------------------------------------------------
# Public launcher
# ---------------------------------------------------------------------------

def _check_triton_available() -> None:
    """Raise if CUDA or Triton is not available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — Triton kernel requires a GPU")
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not installed. Install with: pip install triton"
        )


def triton_aglr_logsumexp_scoring(
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scales_expanded: torch.Tensor,
    key_is_zero_expanded: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the Triton AGLR-C logsumexp scoring kernel.

    Args:
        query: [B, H, L, D] FP16 or FP32 query tensor (on CUDA).
        key_int8: [B, H, L, D] INT8 key values (on CUDA).
        key_scales_expanded: [B, H, L, D] FP32 per-coordinate scales (on CUDA).
        key_is_zero_expanded: [B, H, L, D] bool zero-group flags (on CUDA).
        valid_mask: [B, H, Q_blk, K_blk] bool causal mask (on CUDA).

    Returns:
        quantized_scores: [B, H, Q_blk, K_blk] FP32.
        lower_scores: [B, H, Q_blk, K_blk] FP32.
        upper_scores: [B, H, Q_blk, K_blk] FP32.
    """
    _check_triton_available()

    batch, heads, seq_len, dim = query.shape
    block_size = _BLOCK_SIZE
    num_blocks = seq_len // block_size
    q_blk, k_blk = num_blocks, num_blocks

    assert dim == _D_FIXED, f"Expected D={_D_FIXED}, got {dim}"
    assert valid_mask.shape == (batch, heads, q_blk, k_blk), (
        f"valid_mask shape {valid_mask.shape} != expected {(batch, heads, q_blk, k_blk)}"
    )

    # Ensure contiguous and convert bools to int8 for Triton compatibility
    query = query.contiguous()
    key_int8 = key_int8.contiguous()
    key_scales_expanded = key_scales_expanded.contiguous()
    key_is_zero_expanded = key_is_zero_expanded.to(torch.int8).contiguous()
    valid_mask_i8 = valid_mask.to(torch.int8).contiguous()

    # Create output tensors
    out_quant = torch.zeros(
        batch, heads, q_blk, k_blk, dtype=torch.float32, device=query.device,
    )
    out_lower = torch.zeros(
        batch, heads, q_blk, k_blk, dtype=torch.float32, device=query.device,
    )
    out_upper = torch.zeros(
        batch, heads, q_blk, k_blk, dtype=torch.float32, device=query.device,
    )

    # Strides for [B, H, L, D] tensors (contiguous)
    stride_b = heads * seq_len * dim
    stride_h = seq_len * dim
    stride_l = dim

    # Strides for [B, H, Q_blk, K_blk] tensors (contiguous)
    stride_vb = heads * q_blk * k_blk
    stride_vh = q_blk * k_blk
    stride_vq = k_blk

    grid = (batch * heads * q_blk * k_blk,)

    _aglr_certimask_logsumexp_kernel[grid](
        query,
        key_int8,
        key_scales_expanded,
        key_is_zero_expanded,
        valid_mask_i8,
        out_quant,
        out_lower,
        out_upper,
        stride_b,
        stride_h,
        stride_l,
        stride_vb,
        stride_vh,
        stride_vq,
        batch,
        heads,
        q_blk,
        k_blk,
        BLOCK_SIZE=_BLOCK_SIZE,
        D_SIZE=_D_FIXED,
        NUM_SAMPLES=_NUM_SAMPLES,
        SQRT_D=8.0,  # sqrt(64) = 8.0
    )

    return out_quant, out_lower, out_upper
