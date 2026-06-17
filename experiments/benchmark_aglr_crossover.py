#!/usr/bin/env python3
"""Phase 9E: Long-context Crossover and Benchmark Semantics Audit.

Scans L=1024, 2048, 4096, 8192 to find the crossover point where
AGLR-C + CertiMask becomes competitive with dense SDPA.

All certificate paths use the fused Triton partition certificate
(triton_certified_topk_mask_partition) instead of the slow PyTorch
loop-based certified_topk_mask.

Modes:
  1. Dense SDPA baseline (FP16 and FP32)
  2A. Online full with quantization (reference + quant + kernel + vectorized topk + fused cert)
  2B. Online full with cached quantization (quant outside, reference/topk/kernel/fused cert inside)
  3. Optimistic cached indexer (reference/mask/quant outside, kernel + fused cert inside)

Usage:
    python experiments/benchmark_aglr_crossover.py \
        --batch-size 1 --num-heads 14 --head-dim 64 \
        --block-size 8 --group-size 4 \
        --dtype float16 --device cuda --warmup 20 --iters 100 \
        --seq-lens 1024 2048 4096 8192 \
        --output-dir outputs/phase9e_long_context_crossover
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from certimask.masking import make_block_causal_valid_mask
from certimask.quantization import quantize_int8_per_group
from certimask.triton_aglr_ops import _expand_group_tensor


def _check_env() -> tuple[bool, bool]:
    cuda_ok = torch.cuda.is_available()
    triton_ok = False
    if cuda_ok:
        try:
            import triton  # noqa: F401
            triton_ok = True
        except ImportError:
            pass
    return cuda_ok, triton_ok


def _make_synthetic(
    batch: int, heads: int, seq_len: int, dim: int,
    dtype: torch.dtype, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(42)
    q = torch.randn(batch, heads, seq_len, dim, dtype=dtype, device=device, generator=gen)
    k = torch.randn(batch, heads, seq_len, dim, dtype=dtype, device=device, generator=gen)
    return q, k


def _cuda_timer(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _stats(times: list[float]) -> dict[str, float]:
    s = sorted(times)
    n = len(s)
    p10 = s[max(0, n // 10 - 1)]
    p90 = s[min(n - 1, int(n * 0.9))]
    return {
        "median": statistics.median(s),
        "p10": p10,
        "p90": p90,
        "mean": statistics.mean(s),
        "std": statistics.stdev(s) if n > 1 else 0.0,
    }


def _dense_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Dense attention via torch SDPA."""
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=causal,
    )


def _estimate_alpha(lengths: list[int], times: list[float]) -> float:
    """Estimate scaling exponent alpha where T ~ L^alpha via log-log fit."""
    valid = [
        (length, t) for length, t in zip(lengths, times, strict=True) if t > 0
    ]
    if len(valid) < 2:
        return float("nan")

    log_lengths = [math.log(length) for length, _ in valid]
    log_times = [math.log(t) for _, t in valid]

    n = len(valid)
    sum_x = sum(log_lengths)
    sum_y = sum(log_times)
    sum_xy = sum(x * y for x, y in zip(log_lengths, log_times, strict=True))
    sum_x2 = sum(x * x for x in log_lengths)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        return float("nan")

    alpha = (n * sum_xy - sum_x * sum_y) / denom
    return alpha


def _find_crossover_length(
    lengths: list[int],
    indexer_times: list[float],
    dense_times: list[float],
) -> int | None:
    """Find the first L where indexer_time + ideal_sparse <= dense_time."""
    work_fraction = 0.3765
    for length, idx_t, dense_t in zip(
        lengths, indexer_times, dense_times, strict=True,
    ):
        if dense_t <= 0:
            continue
        ideal_sparse = dense_t * work_fraction
        total = idx_t + ideal_sparse
        if total <= dense_t:
            return length
    return None


def _build_vectorized_topk_mask(
    fp_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    target_sparsity: float,
) -> torch.Tensor:
    """Build reference mask via vectorized top-k."""
    from certimask.vectorized_topk import vectorized_topk_mask

    valid_per_row = valid_mask.sum(dim=-1)
    keep_per_row = (valid_per_row.float() * (1.0 - target_sparsity)).ceil().long()
    keep_per_row = torch.clamp(keep_per_row, min=1)
    result = vectorized_topk_mask(
        fp_scores, k_per_row=keep_per_row, valid_mask=valid_mask,
    )
    mask = result.mask
    if mask.shape != valid_mask.shape:
        padded = torch.zeros_like(valid_mask)
        n2, k2 = mask.shape[2], mask.shape[3]
        padded[:, :, :n2, :k2] = mask
        mask = padded
    return mask


def _fused_certificate(
    triton_lower: torch.Tensor,
    triton_upper: torch.Tensor,
    reference_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused Triton partition certificate."""
    from certimask.triton_topk_certificate import (
        triton_certified_topk_mask_partition,
    )

    return triton_certified_topk_mask_partition(
        triton_lower, triton_upper, reference_mask, valid_mask,
    )


def _resolve_mask(
    reference_mask: torch.Tensor,
    decisions: torch.Tensor,
    ambiguous: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Resolve final mask: certified tiles keep their decision,
    ambiguous tiles fall back to reference."""
    from certimask.topk_certificate import KEEP

    mask = torch.zeros_like(valid_mask)
    # KEEP tiles stay KEEP
    mask[(decisions == KEEP) & valid_mask] = True
    # DROP tiles stay DROP (already False)
    # AMBIGUOUS tiles fall back to reference
    mask[ambiguous & valid_mask] = reference_mask[ambiguous & valid_mask]
    return mask


def _run_online_full(
    q: torch.Tensor,
    key: torch.Tensor,
    valid_mask: torch.Tensor,
    block_size: int,
    target_sparsity: float,
    key_int8: torch.Tensor,
    key_scales: torch.Tensor,
    key_is_zero: torch.Tensor,
    *,
    quantize_inside: bool,
    group_size: int,
    dim: int,
) -> torch.Tensor:
    """Online full pipeline with fused Triton certificate.

    If quantize_inside=True, K quantization happens inside timing.
    If quantize_inside=False, pre-quantized key_int8/scales/is_zero are used.
    """
    from certimask.aglr_indexer import compute_antidiagonal_block_scores
    from certimask.triton_aglr_kernels import triton_aglr_logsumexp_scoring

    if quantize_inside:
        k_q = quantize_int8_per_group(key, group_size=group_size)
        ki8 = k_q.values
        ks = _expand_group_tensor(k_q.scale, group_size, dim).contiguous()
        kz = _expand_group_tensor(
            k_q.is_zero_group.to(torch.int8), group_size, dim,
        ).to(torch.bool).contiguous()
    else:
        ki8 = key_int8
        ks = key_scales
        kz = key_is_zero

    # FP32 reference scores (online)
    fp_scores = compute_antidiagonal_block_scores(
        q, key, block_size=block_size,
        sample_pattern="both_diagonals", aggregation="logsumexp",
        valid_mask=valid_mask, scale_by_sqrt_dim=True,
    )

    # Vectorized top-k mask (online)
    ref_mask = _build_vectorized_topk_mask(fp_scores, valid_mask, target_sparsity)

    # Triton scoring kernel
    triton_q, triton_l, triton_u = triton_aglr_logsumexp_scoring(
        q, ki8, ks, kz, valid_mask,
    )

    # Fused Triton certificate
    decisions, ambiguous = _fused_certificate(
        triton_l, triton_u, ref_mask, valid_mask,
    )

    # Resolve final mask
    mask = _resolve_mask(ref_mask, decisions, ambiguous, valid_mask)
    return mask


def _precompute_optimistic(
    q: torch.Tensor,
    key: torch.Tensor,
    valid_mask: torch.Tensor,
    block_size: int,
    target_sparsity: float,
    group_size: int,
    dim: int,
) -> tuple:
    """Pre-compute reference scores, mask, and K quantization for Mode 3."""
    from certimask.aglr_indexer import compute_antidiagonal_block_scores

    fp_scores = compute_antidiagonal_block_scores(
        q, key, block_size=block_size,
        sample_pattern="both_diagonals", aggregation="logsumexp",
        valid_mask=valid_mask, scale_by_sqrt_dim=True,
    )
    ref_mask = _build_vectorized_topk_mask(fp_scores, valid_mask, target_sparsity)

    k_q = quantize_int8_per_group(key, group_size=group_size)
    ks = _expand_group_tensor(k_q.scale, group_size, dim).contiguous()
    kz = _expand_group_tensor(
        k_q.is_zero_group.to(torch.int8), group_size, dim,
    ).to(torch.bool).contiguous()

    return k_q.values, ks, kz, ref_mask


def _run_optimistic(
    q: torch.Tensor,
    valid_mask: torch.Tensor,
    key_int8: torch.Tensor,
    key_scales: torch.Tensor,
    key_is_zero: torch.Tensor,
    ref_mask: torch.Tensor,
) -> torch.Tensor:
    """Mode 3: optimistic cached indexer with fused certificate."""
    from certimask.triton_aglr_kernels import triton_aglr_logsumexp_scoring

    # Triton scoring kernel
    triton_q, triton_l, triton_u = triton_aglr_logsumexp_scoring(
        q, key_int8, key_scales, key_is_zero, valid_mask,
    )

    # Fused Triton certificate
    decisions, ambiguous = _fused_certificate(
        triton_l, triton_u, ref_mask, valid_mask,
    )

    # Resolve final mask
    mask = _resolve_mask(ref_mask, decisions, ambiguous, valid_mask)
    return mask


def run_benchmark(args: argparse.Namespace) -> dict:
    cuda_ok, triton_ok = _check_env()
    if not cuda_ok:
        raise RuntimeError("CUDA not available")
    if not triton_ok:
        raise RuntimeError("Triton not installed")

    device = args.device
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    block_size = args.block_size
    group_size = args.group_size
    warmup = args.warmup
    iters = args.iters
    target_sparsity = 0.5
    work_fraction = 0.3765

    seq_lens = args.seq_lens

    print("=" * 60)
    print("Phase 9E: Long-context Crossover Benchmark")
    print("=" * 60)
    print(f"Sequence lengths: {seq_lens}")
    print(f"Block size: {block_size}, Group size: {group_size}")
    print(f"Warmup: {warmup}, Iterations: {iters}")
    print("Certificate: fused Triton (triton_certified_topk_mask_partition)")

    per_length: list[dict] = []

    for seq_len in seq_lens:
        print(f"\n{'='*60}")
        print(f"L = {seq_len}")
        print(f"{'='*60}")

        try:
            q, k = _make_synthetic(
                args.batch_size, args.num_heads, seq_len,
                args.head_dim, dtype, device,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"  OOM at L={seq_len}, skipping")
                per_length.append({
                    "seq_len": seq_len,
                    "num_blocks": seq_len // block_size,
                    "oom": True,
                })
                torch.cuda.empty_cache()
                continue
            raise

        batch, heads, length, dim = q.shape
        num_blocks = length // block_size

        valid_mask = make_block_causal_valid_mask(
            num_blocks, num_blocks, device=device,
        ).expand(batch, heads, num_blocks, num_blocks)

        valid_tiles = int(valid_mask.sum().item())
        total_tiles = num_blocks * num_blocks

        print(f"  Shape: B={batch}, H={heads}, L={length}, D={dim}")
        print(f"  Blocks: {num_blocks}, Valid tiles: {valid_tiles}/{total_tiles}")

        # Mode 1: Dense SDPA (FP16 and FP32)
        v = torch.randn_like(k)
        q_f32 = q.float()
        k_f32 = k.float()
        v_f32 = v.float()

        def mode1_fp16(_q=q, _k=k, _v=v):
            return _dense_sdpa(_q, _k, _v, causal=True)

        def mode1_fp32(_q=q_f32, _k=k_f32, _v=v_f32):
            return _dense_sdpa(_q, _k, _v, causal=True)

        t1_fp16 = _cuda_timer(mode1_fp16, warmup, iters)
        t1_fp32 = _cuda_timer(mode1_fp32, warmup, iters)

        # Pre-quantize K for Modes 2B and 3
        k_q_cached = quantize_int8_per_group(k, group_size=group_size)
        ks_cached = _expand_group_tensor(
            k_q_cached.scale, group_size, dim,
        ).contiguous()
        kz_cached = _expand_group_tensor(
            k_q_cached.is_zero_group.to(torch.int8), group_size, dim,
        ).to(torch.bool).contiguous()

        # Mode 2A: Online full with quantization + fused cert
        def mode2a(
            _q=q, _k=k, _vm=valid_mask,
            _ki8=k_q_cached.values, _ks=ks_cached, _kz=kz_cached,
            _dim=dim,
        ):
            return _run_online_full(
                _q, _k, _vm, block_size, target_sparsity,
                _ki8, _ks, _kz,
                quantize_inside=True, group_size=group_size, dim=_dim,
            )

        t2a = _cuda_timer(mode2a, warmup, iters)

        # Mode 2B: Cached quant + online reference/topk + fused cert
        def mode2b(
            _q=q, _k=k, _vm=valid_mask,
            _ki8=k_q_cached.values, _ks=ks_cached, _kz=kz_cached,
            _dim=dim,
        ):
            return _run_online_full(
                _q, _k, _vm, block_size, target_sparsity,
                _ki8, _ks, _kz,
                quantize_inside=False, group_size=group_size, dim=_dim,
            )

        t2b = _cuda_timer(mode2b, warmup, iters)

        # Mode 3: Optimistic cached indexer + fused cert
        ki8_pre, ks_pre, kz_pre, ref_mask_pre = _precompute_optimistic(
            q, k, valid_mask, block_size, target_sparsity, group_size, dim,
        )

        def mode3(
            _q=q, _vm=valid_mask,
            _ki8=ki8_pre, _ks=ks_pre, _kz=kz_pre, _rm=ref_mask_pre,
        ):
            return _run_optimistic(_q, _vm, _ki8, _ks, _kz, _rm)

        t3 = _cuda_timer(mode3, warmup, iters)

        # Collect stats
        s1_fp16 = _stats(t1_fp16)
        s1_fp32 = _stats(t1_fp32)
        s2a = _stats(t2a)
        s2b = _stats(t2b)
        s3 = _stats(t3)

        dense_fp16_ms = s1_fp16["median"]
        dense_fp32_ms = s1_fp32["median"]
        ideal_sparse_ms = dense_fp16_ms * work_fraction

        online_total = s2a["median"] + ideal_sparse_ms
        cached_total = s2b["median"] + ideal_sparse_ms
        optimistic_total = s3["median"] + ideal_sparse_ms

        online_speedup = dense_fp16_ms / online_total if online_total > 0 else 0.0
        cached_speedup = dense_fp16_ms / cached_total if cached_total > 0 else 0.0
        optimistic_speedup = (
            dense_fp16_ms / optimistic_total if optimistic_total > 0 else 0.0
        )

        entry = {
            "seq_len": seq_len,
            "num_blocks": num_blocks,
            "valid_tiles": valid_tiles,
            "total_tiles": total_tiles,
            "oom": False,
            "dense_sdpa_fp16_ms": s1_fp16,
            "dense_sdpa_fp32_ms": s1_fp32,
            "online_full_with_quant_ms": s2a,
            "online_full_cached_quant_ms": s2b,
            "optimistic_cached_indexer_ms": s3,
            "ideal_sparse_attention_ms": ideal_sparse_ms,
            "online_total_proxy_ms": online_total,
            "cached_quant_total_proxy_ms": cached_total,
            "optimistic_total_proxy_ms": optimistic_total,
            "online_speedup_proxy": online_speedup,
            "cached_quant_speedup_proxy": cached_speedup,
            "optimistic_speedup_proxy": optimistic_speedup,
        }
        per_length.append(entry)

        print(f"  Dense SDPA (FP16):       {dense_fp16_ms:10.4f} ms")
        print(f"  Dense SDPA (FP32):       {dense_fp32_ms:10.4f} ms")
        print(f"  Online full (quant):     {s2a['median']:10.4f} ms")
        print(f"  Online full (cached Q):  {s2b['median']:10.4f} ms")
        print(f"  Optimistic cached:       {s3['median']:10.4f} ms")
        print(f"  Ideal sparse:            {ideal_sparse_ms:10.4f} ms")
        print(f"  Online speedup proxy:    {online_speedup:10.4f}x")
        print(f"  Cached Q speedup proxy:  {cached_speedup:10.4f}x")
        print(f"  Optimistic speedup proxy:{optimistic_speedup:10.4f}x")

        del q, k, v, valid_mask, q_f32, k_f32, v_f32
        torch.cuda.empty_cache()

    # Scaling exponents
    valid_entries = [e for e in per_length if not e.get("oom")]
    lengths = [e["seq_len"] for e in valid_entries]
    dense_fp16_times = [e["dense_sdpa_fp16_ms"]["median"] for e in valid_entries]
    dense_fp32_times = [e["dense_sdpa_fp32_ms"]["median"] for e in valid_entries]
    online_times = [e["online_full_with_quant_ms"]["median"] for e in valid_entries]
    triton_times = [e["optimistic_cached_indexer_ms"]["median"] for e in valid_entries]

    dense_fp16_alpha = _estimate_alpha(lengths, dense_fp16_times)
    dense_fp32_alpha = _estimate_alpha(lengths, dense_fp32_times)
    indexer_alpha = _estimate_alpha(lengths, online_times)
    triton_alpha = _estimate_alpha(lengths, triton_times)

    scaling = {
        "dense_sdpa_fp16_alpha": dense_fp16_alpha,
        "dense_sdpa_fp32_alpha": dense_fp32_alpha,
        "online_indexer_alpha": indexer_alpha,
        "triton_scoring_alpha": triton_alpha,
    }

    # Crossover detection
    online_speedups = [e.get("online_speedup_proxy", 0.0) for e in per_length]
    cached_speedups = [e.get("cached_quant_speedup_proxy", 0.0) for e in per_length]

    def _crossover_helper(metric_key: str) -> int | None:
        return _find_crossover_length(
            [e["seq_len"] for e in per_length],
            [
                e.get(metric_key, {}).get("median", 1e18)
                if not e.get("oom") else 1e18
                for e in per_length
            ],
            [
                e.get("dense_sdpa_fp16_ms", {}).get("median", 0.0)
                if not e.get("oom") else 0.0
                for e in per_length
            ],
        )

    online_crossover = _crossover_helper("online_full_with_quant_ms")
    cached_crossover = _crossover_helper("online_full_cached_quant_ms")
    optimistic_crossover = _crossover_helper("optimistic_cached_indexer_ms")

    # Readiness decision
    any_online_viable = any(s >= 1.1 for s in online_speedups if s > 0)
    any_cached_viable = any(s >= 1.1 for s in cached_speedups if s > 0)

    if any_online_viable:
        readiness = "ready_for_sparse_attention_kernel"
        next_step = "Proceed to sparse attention kernel implementation."
    elif any_cached_viable:
        readiness = "ready_under_cached_quantization"
        next_step = (
            "Sparse attention kernel justified only if K quantization "
            "can be cached or fused. Proceed with caution."
        )
    else:
        readiness = "not_viable_at_tested_lengths"
        next_step = (
            "No crossover found at tested lengths. "
            "Consider even longer contexts or fundamental redesign."
        )

    results = {
        "config": vars(args),
        "per_length": per_length,
        "scaling_exponents": scaling,
        "crossover": {
            "first_online_crossover_length": online_crossover,
            "first_cached_quant_crossover_length": cached_crossover,
            "first_optimistic_crossover_length": optimistic_crossover,
        },
        "readiness": {
            "readiness": readiness,
            "ready_for_sparse_attention_kernel": any_online_viable,
            "recommended_next_step": next_step,
        },
    }

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 9E long-context crossover benchmark",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=14)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "float32", "bfloat16"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seq-lens", type=int, nargs="+",
                    default=[1024, 2048, 4096, 8192])
    p.add_argument("--output-dir", type=str,
                    default="outputs/phase9e_long_context_crossover")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(args)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(results["config"], f, indent=2)

    # Save full results
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save crossover results CSV
    csv_lines = [
        "seq_len,num_blocks,valid_tiles,"
        "dense_sdpa_fp16_ms,dense_sdpa_fp32_ms,"
        "online_full_with_quant_ms,online_full_cached_quant_ms,"
        "optimistic_cached_indexer_ms,ideal_sparse_attention_ms,"
        "online_total_proxy_ms,cached_quant_total_proxy_ms,"
        "optimistic_total_proxy_ms,online_speedup_proxy,"
        "cached_quant_speedup_proxy,optimistic_speedup_proxy,oom",
    ]
    for entry in results["per_length"]:
        if entry.get("oom"):
            csv_lines.append(
                f"{entry['seq_len']},,,,,0,,,,,,,,,,,,true"
            )
        else:
            csv_lines.append(
                f"{entry['seq_len']},{entry['num_blocks']},{entry['valid_tiles']},"
                f"{entry['dense_sdpa_fp16_ms']['median']:.4f},"
                f"{entry['dense_sdpa_fp32_ms']['median']:.4f},"
                f"{entry['online_full_with_quant_ms']['median']:.4f},"
                f"{entry['online_full_cached_quant_ms']['median']:.4f},"
                f"{entry['optimistic_cached_indexer_ms']['median']:.4f},"
                f"{entry['ideal_sparse_attention_ms']:.4f},"
                f"{entry['online_total_proxy_ms']:.4f},"
                f"{entry['cached_quant_total_proxy_ms']:.4f},"
                f"{entry['optimistic_total_proxy_ms']:.4f},"
                f"{entry['online_speedup_proxy']:.4f},"
                f"{entry['cached_quant_speedup_proxy']:.4f},"
                f"{entry['optimistic_speedup_proxy']:.4f},false",
            )
    with open(output_dir / "crossover_results.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    # Save scaling exponents
    with open(output_dir / "scaling_exponents.json", "w") as f:
        json.dump(results["scaling_exponents"], f, indent=2)

    # Save summary
    crossover = results["crossover"]
    scaling = results["scaling_exponents"]
    readiness = results["readiness"]
    summary = {
        "tested_lengths": [e["seq_len"] for e in results["per_length"]],
        "dense_sdpa_fp16_ms_by_length": {
            str(e["seq_len"]): e.get("dense_sdpa_fp16_ms", {}).get("median")
            for e in results["per_length"] if not e.get("oom")
        },
        "dense_sdpa_fp32_ms_by_length": {
            str(e["seq_len"]): e.get("dense_sdpa_fp32_ms", {}).get("median")
            for e in results["per_length"] if not e.get("oom")
        },
        "online_indexer_ms_by_length": {
            str(e["seq_len"]): e.get("online_full_with_quant_ms", {}).get("median")
            for e in results["per_length"] if not e.get("oom")
        },
        "cached_quant_indexer_ms_by_length": {
            str(e["seq_len"]): e.get("online_full_cached_quant_ms", {}).get("median")
            for e in results["per_length"] if not e.get("oom")
        },
        "optimistic_indexer_ms_by_length": {
            str(e["seq_len"]): e.get("optimistic_cached_indexer_ms", {}).get("median")
            for e in results["per_length"] if not e.get("oom")
        },
        "online_speedup_proxy_by_length": {
            str(e["seq_len"]): e.get("online_speedup_proxy")
            for e in results["per_length"] if not e.get("oom")
        },
        "cached_quant_speedup_proxy_by_length": {
            str(e["seq_len"]): e.get("cached_quant_speedup_proxy")
            for e in results["per_length"] if not e.get("oom")
        },
        "optimistic_speedup_proxy_by_length": {
            str(e["seq_len"]): e.get("optimistic_speedup_proxy")
            for e in results["per_length"] if not e.get("oom")
        },
        "first_online_crossover_length": crossover["first_online_crossover_length"],
        "first_cached_quant_crossover_length": crossover["first_cached_quant_crossover_length"],
        "first_optimistic_crossover_length": crossover["first_optimistic_crossover_length"],
        "dense_sdpa_fp16_scaling_exponent": scaling["dense_sdpa_fp16_alpha"],
        "dense_sdpa_fp32_scaling_exponent": scaling["dense_sdpa_fp32_alpha"],
        "indexer_scaling_exponent": scaling["online_indexer_alpha"],
        "ready_for_sparse_attention_kernel": readiness["ready_for_sparse_attention_kernel"],
        "recommended_next_step": readiness["recommended_next_step"],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save pipeline mode definitions
    mode_defs = """# Pipeline Mode Definitions

## Certificate
All modes use fused Triton partition certificate
(`triton_certified_topk_mask_partition`) instead of the slow PyTorch
loop-based `certified_topk_mask`. The fused certificate is ~73,000x faster.

## Mode 1: Dense SDPA Baseline
- `torch.nn.functional.scaled_dot_product_attention`
- Causal=True
- Reports both FP16 and FP32 timings

## Mode 2A: Online Full with Quantization
- FP32 reference AGLR scores (online)
- K quantization (online)
- Vectorized top-k mask (online)
- Triton scoring kernel
- Fused Triton certificate
- **Online valid**: all computation happens inside timing loop

## Mode 2B: Online Full with Cached Quantization
- K quantization pre-computed outside timing loop
- FP32 reference AGLR scores (online)
- Vectorized top-k mask (online)
- Triton scoring kernel (with cached K)
- Fused Triton certificate
- **Online valid**: reference scores and top-k still computed per-call

## Mode 3: Optimistic Cached Indexer
- Pre-computed outside: FP32 reference scores, vectorized top-k mask, K quantization
- Inside timing: Triton scoring kernel + fused certificate + fallback
- **NOT online valid**: assumes reference scores and mask are cached
- Measures: lower-bound certificate overhead

## Ideal Sparse Attention Proxy
- `ideal_sparse_ms = dense_sdpa_fp16_ms * work_fraction`
- `work_fraction = 0.3765` (from Phase 8C mean attention tile work fraction)
- Assumes: sparse attention compute scales linearly with kept tiles
"""
    with open(output_dir / "pipeline_mode_definitions.md", "w") as f:
        f.write(mode_defs)

    # Save README
    readme_lines = [
        "# Phase 9E: Long-context Crossover Analysis",
        "",
        "## Configuration",
        f"- Batch: {args.batch_size}, Heads: {args.num_heads}, D: {args.head_dim}",
        f"- Block size: {args.block_size}, Group size: {args.group_size}",
        f"- Sequence lengths: {args.seq_lens}",
        f"- Warmup: {args.warmup}, Iterations: {args.iters}",
        "- Certificate: fused Triton (triton_certified_topk_mask_partition)",
        "",
        "## Results Summary",
        "",
        "| L | Dense FP16 | Dense FP32 | Online | Cached Q | Optimistic |",
        "|---|---|---|---|---|---|",
    ]
    for entry in results["per_length"]:
        if entry.get("oom"):
            readme_lines.append(f"| {entry['seq_len']} | OOM | - | - | - | - |")
        else:
            d16 = entry["dense_sdpa_fp16_ms"]["median"]
            d32 = entry["dense_sdpa_fp32_ms"]["median"]
            o = entry["online_full_with_quant_ms"]["median"]
            c = entry["online_full_cached_quant_ms"]["median"]
            op = entry["optimistic_cached_indexer_ms"]["median"]
            readme_lines.append(
                f"| {entry['seq_len']} | {d16:.2f} | {d32:.2f} | {o:.2f} | {c:.2f} | {op:.2f} |"
            )

    readme_lines += [
        "",
        "| L | Online Speedup | Cached Speedup | Optimistic Speedup |",
        "|---|---|---|---|",
    ]
    for entry in results["per_length"]:
        if entry.get("oom"):
            readme_lines.append(f"| {entry['seq_len']} | - | - | - |")
        else:
            os_ = entry["online_speedup_proxy"]
            cs = entry["cached_quant_speedup_proxy"]
            ops = entry["optimistic_speedup_proxy"]
            readme_lines.append(
                f"| {entry['seq_len']} | {os_:.4f}x | {cs:.4f}x | {ops:.4f}x |"
            )

    readme_lines += [
        "",
        "## Scaling Exponents",
        f"- Dense SDPA (FP16): alpha = {scaling['dense_sdpa_fp16_alpha']:.2f}",
        f"- Dense SDPA (FP32): alpha = {scaling['dense_sdpa_fp32_alpha']:.2f}",
        f"- Online indexer: alpha = {scaling['online_indexer_alpha']:.2f}",
        f"- Triton scoring: alpha = {scaling['triton_scoring_alpha']:.2f}",
        "",
        "## Crossover Points",
        f"- Online: {crossover['first_online_crossover_length']}",
        f"- Cached quantization: {crossover['first_cached_quant_crossover_length']}",
        f"- Optimistic: {crossover['first_optimistic_crossover_length']}",
        "",
        f"## Readiness: {readiness['readiness']}",
        readiness["recommended_next_step"],
    ]
    with open(output_dir / "README_results.md", "w") as f:
        f.write("\n".join(readme_lines) + "\n")

    # Print summary
    print("\n" + "=" * 60)
    print("CROSSOVER ANALYSIS SUMMARY")
    print("=" * 60)
    for entry in results["per_length"]:
        if entry.get("oom"):
            print(f"  L={entry['seq_len']:5d}: OOM")
        else:
            d16 = entry["dense_sdpa_fp16_ms"]["median"]
            o = entry["online_full_with_quant_ms"]["median"]
            c = entry["online_full_cached_quant_ms"]["median"]
            op = entry["optimistic_cached_indexer_ms"]["median"]
            os_ = entry["online_speedup_proxy"]
            print(
                f"  L={entry['seq_len']:5d}: "
                f"dense_fp16={d16:.2f}ms  online={o:.2f}ms  "
                f"cachedQ={c:.2f}ms  optimistic={op:.2f}ms  "
                f"speedup_online={os_:.4f}x"
            )

    print("\nScaling exponents:")
    print(f"  Dense SDPA (FP16): {scaling['dense_sdpa_fp16_alpha']:.2f}")
    print(f"  Dense SDPA (FP32): {scaling['dense_sdpa_fp32_alpha']:.2f}")
    print(f"  Online indexer:    {scaling['online_indexer_alpha']:.2f}")
    print(f"  Triton scoring:   {scaling['triton_scoring_alpha']:.2f}")

    print("\nCrossover points:")
    print(f"  Online:    {crossover['first_online_crossover_length']}")
    print(f"  Cached Q:  {crossover['first_cached_quant_crossover_length']}")
    print(f"  Optimistic:{crossover['first_optimistic_crossover_length']}")

    print(f"\nReadiness: {readiness['readiness']}")
    print(f"Next step: {readiness['recommended_next_step']}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
