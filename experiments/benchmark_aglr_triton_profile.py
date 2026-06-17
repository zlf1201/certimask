#!/usr/bin/env python3
"""Phase 9B: Triton CertiMask latency decomposition and profiling.

Breaks down the Triton full wrapper into individual stages:
  1. key_quantization_ms
  2. triton_score_interval_kernel_ms
  3. reference_fp32_aglr_score_ms
  4. topk_reference_mask_ms
  5. partition_certificate_ms
  6. fallback_resolution_ms

Also compares scoring-only vs full wrapper, and measures allocation overhead.

Usage:
    python experiments/benchmark_aglr_triton_profile.py \
        --batch-size 1 --num-heads 14 --seq-len 1024 \
        --head-dim 64 --block-size 8 --group-size 4 \
        --dtype float16 --device cuda --warmup 20 --iters 100 \
        --output-dir outputs/phase9b_triton_profile
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from certimask.aglr_indexer import aglr_local_plus_landmark_mask, compute_antidiagonal_block_scores
from certimask.masking import make_block_causal_valid_mask
from certimask.quantization import quantize_int8_per_group
from certimask.topk_certificate import certified_topk_mask
from certimask.triton_aglr_kernels import triton_aglr_logsumexp_scoring
from certimask.triton_aglr_ops import _expand_group_tensor
from certimask.triton_topk_certificate import triton_certified_topk_mask_partition
from certimask.vectorized_topk import vectorized_topk_mask


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


def run_profile(args: argparse.Namespace) -> dict:
    cuda_ok, triton_ok = _check_env()
    if not cuda_ok:
        raise RuntimeError("CUDA not available")
    if not triton_ok:
        raise RuntimeError("Triton not installed")

    device = args.device
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    block_size = args.block_size
    group_size = args.group_size

    q, k = _make_synthetic(
        args.batch_size, args.num_heads, args.seq_len, args.head_dim, dtype, device,
    )
    batch, heads, seq_len, dim = q.shape
    num_blocks = seq_len // block_size

    valid_mask = make_block_causal_valid_mask(
        num_blocks, num_blocks, device=device,
    ).expand(batch, heads, num_blocks, num_blocks)

    target_sparsity = 0.5
    local_blocks = 0
    warmup = args.warmup
    iters = args.iters

    print(f"Shape: B={batch}, H={heads}, L={seq_len}, D={dim}")
    print(f"Block size: {block_size}, Group size: {group_size}")
    print(f"Warmup: {warmup}, Iters: {iters}")

    # -----------------------------------------------------------------------
    # Stage 1: Key quantization
    # -----------------------------------------------------------------------
    def stage_key_quant():
        k_q = quantize_int8_per_group(k, group_size=group_size)
        ks = _expand_group_tensor(
            k_q.scale, group_size, dim,
        ).contiguous()
        kz = _expand_group_tensor(
            k_q.is_zero_group.to(torch.int8), group_size, dim,
        ).to(torch.bool).contiguous()
        return k_q, ks, kz

    t_key_quant = _cuda_timer(stage_key_quant, warmup, iters)

    # Pre-quantize for subsequent stages
    k_q = quantize_int8_per_group(k, group_size=group_size)
    key_scales_expanded = _expand_group_tensor(k_q.scale, group_size, dim).contiguous()
    key_is_zero_expanded = _expand_group_tensor(
        k_q.is_zero_group.to(torch.int8), group_size, dim,
    ).to(torch.bool).contiguous()

    # -----------------------------------------------------------------------
    # Stage 2: Triton score interval kernel
    # -----------------------------------------------------------------------
    def stage_triton_kernel():
        return triton_aglr_logsumexp_scoring(
            q, k_q.values, key_scales_expanded, key_is_zero_expanded, valid_mask,
        )

    t_triton_kernel = _cuda_timer(stage_triton_kernel, warmup, iters)

    # -----------------------------------------------------------------------
    # Stage 3: FP32 reference AGLR scores
    # -----------------------------------------------------------------------
    def stage_fp32_ref():
        return compute_antidiagonal_block_scores(
            q, k, block_size=block_size,
            sample_pattern="both_diagonals", aggregation="logsumexp",
            valid_mask=valid_mask, scale_by_sqrt_dim=True,
        )

    t_fp32_ref = _cuda_timer(stage_fp32_ref, warmup, iters)

    # Pre-compute for subsequent stages
    fp_scores = stage_fp32_ref()

    # -----------------------------------------------------------------------
    # Stage 4a: Top-k reference mask (loop - original)
    # -----------------------------------------------------------------------
    def stage_topk_mask_loop():
        return aglr_local_plus_landmark_mask(
            fp_scores, target_sparsity=target_sparsity,
            local_blocks=local_blocks, valid_mask=valid_mask,
        )

    t_topk_mask_loop = _cuda_timer(stage_topk_mask_loop, warmup, iters)

    # -----------------------------------------------------------------------
    # Stage 4b: Top-k reference mask (vectorized)
    # -----------------------------------------------------------------------
    valid_per_row = valid_mask.sum(dim=-1)
    keep_per_row = (valid_per_row.float() * (1.0 - target_sparsity)).ceil().long()
    keep_per_row = torch.clamp(keep_per_row, min=1)

    def stage_topk_mask_vectorized():
        return vectorized_topk_mask(
            fp_scores, k_per_row=keep_per_row, valid_mask=valid_mask,
        )

    t_topk_mask_vec = _cuda_timer(stage_topk_mask_vectorized, warmup, iters)

    # Verify masks match
    loop_result = stage_topk_mask_loop()
    vec_result = stage_topk_mask_vectorized()
    topk_mask_match = torch.equal(loop_result.mask, vec_result.mask)

    # Pre-compute for subsequent stages (use vectorized result)
    reference_mask = vec_result.mask
    if reference_mask.shape != valid_mask.shape:
        padded = torch.zeros_like(valid_mask)
        n2, k2 = reference_mask.shape[2], reference_mask.shape[3]
        padded[:, :, :n2, :k2] = reference_mask
        reference_mask = padded
    k_per_row = (reference_mask & valid_mask).sum(dim=-1).long()

    # Pre-compute Triton scores for certificate
    triton_quant, triton_lower, triton_upper = stage_triton_kernel()

    # -----------------------------------------------------------------------
    # Stage 5: Partition certificate
    # -----------------------------------------------------------------------
    def stage_certificate():
        return certified_topk_mask(
            fp_scores, triton_lower, triton_upper,
            k_per_row=k_per_row, valid_mask=valid_mask,
            ambiguity_mode="partition",
        )

    t_certificate = _cuda_timer(stage_certificate, warmup, iters)

    # Pre-compute for fallback stage
    topk_result = stage_certificate()

    # -----------------------------------------------------------------------
    # Stage 5b: Fused Triton certificate
    # -----------------------------------------------------------------------
    def stage_fused_certificate():
        return triton_certified_topk_mask_partition(
            triton_lower, triton_upper, reference_mask, valid_mask,
        )

    t_fused_certificate = _cuda_timer(stage_fused_certificate, warmup, iters)

    # Verify fused certificate matches PyTorch
    fused_dec, fused_amb = stage_fused_certificate()
    dec_match = torch.equal(topk_result.decisions, fused_dec)
    amb_match = torch.equal(topk_result.ambiguous, fused_amb)

    # -----------------------------------------------------------------------
    # Stage 6: Fallback resolution
    # -----------------------------------------------------------------------
    def stage_fallback():
        mask = topk_result.certified_mask.clone()
        # Fallback: ambiguous tiles use FP reference
        mask[topk_result.ambiguous] = reference_mask[topk_result.ambiguous]
        mismatch = (mask != reference_mask) & valid_mask
        return mismatch.sum().item()

    t_fallback = _cuda_timer(stage_fallback, warmup, iters)

    # -----------------------------------------------------------------------
    # Full wrapper timing
    # -----------------------------------------------------------------------
    def stage_full_wrapper():
        from certimask.triton_aglr_ops import triton_aglr_certimask_logsumexp_g4
        return triton_aglr_certimask_logsumexp_g4(
            q, k, target_sparsity=target_sparsity, local_blocks=local_blocks,
        )

    t_full_wrapper = _cuda_timer(stage_full_wrapper, warmup, iters)

    # -----------------------------------------------------------------------
    # Scoring-only timing (just Triton kernel)
    # -----------------------------------------------------------------------
    def stage_scoring_only():
        return triton_aglr_logsumexp_scoring(
            q, k_q.values, key_scales_expanded, key_is_zero_expanded, valid_mask,
        )

    t_scoring_only = _cuda_timer(stage_scoring_only, warmup, iters)

    # -----------------------------------------------------------------------
    # Allocation overhead check
    # -----------------------------------------------------------------------
    # With allocation: create everything inside the timed loop
    def stage_with_alloc():
        k_q2 = quantize_int8_per_group(k, group_size=group_size)
        ks2 = _expand_group_tensor(k_q2.scale, group_size, dim).contiguous()
        kz2 = _expand_group_tensor(
            k_q2.is_zero_group.to(torch.int8), group_size, dim,
        ).to(torch.bool).contiguous()
        return triton_aglr_logsumexp_scoring(
            q, k_q2.values, ks2, kz2, valid_mask,
        )

    t_with_alloc = _cuda_timer(stage_with_alloc, warmup, iters)

    # Reuse buffers: everything pre-allocated
    def stage_reuse():
        return triton_aglr_logsumexp_scoring(
            q, k_q.values, key_scales_expanded, key_is_zero_expanded, valid_mask,
        )

    t_reuse = _cuda_timer(stage_reuse, warmup, iters)

    # -----------------------------------------------------------------------
    # Collect results
    # -----------------------------------------------------------------------
    stats_key_quant = _stats(t_key_quant)
    stats_triton_kernel = _stats(t_triton_kernel)
    stats_fp32_ref = _stats(t_fp32_ref)
    stats_topk_mask_loop = _stats(t_topk_mask_loop)
    stats_topk_mask_vec = _stats(t_topk_mask_vec)
    stats_certificate = _stats(t_certificate)
    stats_fused_certificate = _stats(t_fused_certificate)
    stats_fallback = _stats(t_fallback)
    stats_full_wrapper = _stats(t_full_wrapper)
    stats_scoring_only = _stats(t_scoring_only)
    stats_with_alloc = _stats(t_with_alloc)
    stats_reuse = _stats(t_reuse)

    wrapper_overhead_ms = stats_full_wrapper["median"] - stats_scoring_only["median"]
    wrapper_overhead_fraction = (
        wrapper_overhead_ms / stats_full_wrapper["median"]
        if stats_full_wrapper["median"] > 0 else 0.0
    )
    allocation_overhead_ms = stats_with_alloc["median"] - stats_reuse["median"]

    # Top-k mask speedup
    topk_speedup = (
        stats_topk_mask_loop["median"] / stats_topk_mask_vec["median"]
        if stats_topk_mask_vec["median"] > 0 else 0.0
    )

    # Find largest bottleneck (excluding full_wrapper and scoring_only)
    stage_medians = {
        "key_quantization": stats_key_quant["median"],
        "triton_score_interval_kernel": stats_triton_kernel["median"],
        "reference_fp32_aglr_score": stats_fp32_ref["median"],
        "topk_reference_mask_loop": stats_topk_mask_loop["median"],
        "topk_reference_mask_vectorized": stats_topk_mask_vec["median"],
        "partition_certificate": stats_certificate["median"],
        "fallback_resolution": stats_fallback["median"],
    }
    largest_bottleneck = max(stage_medians, key=stage_medians.get)  # type: ignore[arg-type]

    # Fused certificate speedup
    cert_speedup = (
        stats_certificate["median"] / stats_fused_certificate["median"]
        if stats_fused_certificate["median"] > 0 else 0.0
    )

    # Optimized total: replace PyTorch certificate with fused Triton + vectorized topk
    optimized_total_ms = (
        stats_key_quant["median"]
        + stats_triton_kernel["median"]
        + stats_fp32_ref["median"]
        + stats_topk_mask_vec["median"]
        + stats_fused_certificate["median"]
        + stats_fallback["median"]
    )

    results = {
        "config": vars(args),
        "shape": {"B": batch, "H": heads, "L": seq_len, "D": dim},
        "latency_breakdown": {
            "key_quantization": stats_key_quant,
            "triton_score_interval_kernel": stats_triton_kernel,
            "reference_fp32_aglr_score": stats_fp32_ref,
            "topk_reference_mask_loop": stats_topk_mask_loop,
            "topk_reference_mask_vectorized": stats_topk_mask_vec,
            "partition_certificate_pytorch": stats_certificate,
            "partition_certificate_fused_triton": stats_fused_certificate,
            "fallback_resolution": stats_fallback,
            "total_triton_certimask": stats_full_wrapper,
        },
        "topk_mask_comparison": {
            "loop_ms": stats_topk_mask_loop["median"],
            "vectorized_ms": stats_topk_mask_vec["median"],
            "speedup": topk_speedup,
            "mask_exact_match": topk_mask_match,
        },
        "certificate_fusion": {
            "pytorch_certificate_ms": stats_certificate["median"],
            "fused_triton_certificate_ms": stats_fused_certificate["median"],
            "certificate_speedup": cert_speedup,
            "decisions_match": dec_match,
            "ambiguous_match": amb_match,
        },
        "scoring_only_vs_full": {
            "triton_scoring_only": stats_scoring_only,
            "triton_full_wrapper": stats_full_wrapper,
            "wrapper_overhead_ms": wrapper_overhead_ms,
            "wrapper_overhead_fraction": wrapper_overhead_fraction,
        },
        "allocation_overhead": {
            "with_allocation": stats_with_alloc,
            "reuse_inputs": stats_reuse,
            "allocation_overhead_ms": allocation_overhead_ms,
        },
        "latency_summary": {
            "total_triton_certimask_ms": stats_full_wrapper["median"],
            "optimized_total_with_fused_cert_ms": optimized_total_ms,
            "triton_scoring_only_ms": stats_scoring_only["median"],
            "wrapper_overhead_ms": wrapper_overhead_ms,
            "wrapper_overhead_fraction": wrapper_overhead_fraction,
            "key_quantization_ms": stats_key_quant["median"],
            "triton_score_interval_kernel_ms": stats_triton_kernel["median"],
            "reference_fp32_aglr_score_ms": stats_fp32_ref["median"],
            "topk_reference_mask_loop_ms": stats_topk_mask_loop["median"],
            "topk_reference_mask_vectorized_ms": stats_topk_mask_vec["median"],
            "topk_mask_speedup": topk_speedup,
            "topk_mask_exact_match": topk_mask_match,
            "partition_certificate_pytorch_ms": stats_certificate["median"],
            "partition_certificate_fused_triton_ms": stats_fused_certificate["median"],
            "certificate_speedup": cert_speedup,
            "fallback_resolution_ms": stats_fallback["median"],
            "largest_bottleneck": largest_bottleneck,
            "recommended_next_step": (
                f"Use fused Triton certificate ({cert_speedup:.0f}x speedup) + "
                f"vectorized topk ({topk_speedup:.0f}x speedup). "
                f"Next bottleneck: key_quantization at {stats_key_quant['median']:.3f} ms"
            ),
        },
    }

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 9B Triton CertiMask profiling")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=14)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "float32", "bfloat16"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--output-dir", type=str,
                    default="outputs/phase9b_triton_profile")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 9B: Triton CertiMask Latency Decomposition")
    print("=" * 60)

    results = run_profile(args)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(results["config"], f, indent=2)

    # Save latency summary
    with open(output_dir / "latency_summary.json", "w") as f:
        json.dump(results["latency_summary"], f, indent=2)

    # Save full results
    with open(output_dir / "benchmark_profile.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # CSV: latency breakdown
    breakdown = results["latency_breakdown"]
    csv_lines = ["stage,median_ms,p10_ms,p90_ms,mean_ms,std_ms"]
    for stage_name, stats in breakdown.items():
        csv_lines.append(
            f"{stage_name},{stats['median']:.4f},{stats['p10']:.4f},"
            f"{stats['p90']:.4f},{stats['mean']:.4f},{stats['std']:.4f}",
        )
    with open(output_dir / "latency_breakdown.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    # CSV: scoring-only vs full
    so_vs = results["scoring_only_vs_full"]
    csv_so = [
        "path,median_ms,p10_ms,p90_ms,mean_ms,std_ms",
        f"scoring_only,{so_vs['triton_scoring_only']['median']:.4f},"
        f"{so_vs['triton_scoring_only']['p10']:.4f},"
        f"{so_vs['triton_scoring_only']['p90']:.4f},"
        f"{so_vs['triton_scoring_only']['mean']:.4f},"
        f"{so_vs['triton_scoring_only']['std']:.4f}",
        f"full_wrapper,{so_vs['triton_full_wrapper']['median']:.4f},"
        f"{so_vs['triton_full_wrapper']['p10']:.4f},"
        f"{so_vs['triton_full_wrapper']['p90']:.4f},"
        f"{so_vs['triton_full_wrapper']['mean']:.4f},"
        f"{so_vs['triton_full_wrapper']['std']:.4f}",
    ]
    with open(output_dir / "scoring_only_vs_full.csv", "w") as f:
        f.write("\n".join(csv_so) + "\n")

    # CSV: allocation overhead
    alloc = results["allocation_overhead"]
    csv_alloc = [
        "path,median_ms,p10_ms,p90_ms,mean_ms,std_ms",
        f"with_allocation,{alloc['with_allocation']['median']:.4f},"
        f"{alloc['with_allocation']['p10']:.4f},"
        f"{alloc['with_allocation']['p90']:.4f},"
        f"{alloc['with_allocation']['mean']:.4f},"
        f"{alloc['with_allocation']['std']:.4f}",
        f"reuse_inputs,{alloc['reuse_inputs']['median']:.4f},"
        f"{alloc['reuse_inputs']['p10']:.4f},"
        f"{alloc['reuse_inputs']['p90']:.4f},"
        f"{alloc['reuse_inputs']['mean']:.4f},"
        f"{alloc['reuse_inputs']['std']:.4f}",
    ]
    with open(output_dir / "allocation_overhead.csv", "w") as f:
        f.write("\n".join(csv_alloc) + "\n")

    # Print summary
    summary = results["latency_summary"]
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN (median ms)")
    print("=" * 60)
    for stage_name, stats in breakdown.items():
        total_ms = summary["total_triton_certimask_ms"]
        pct = stats["median"] / total_ms * 100 if total_ms > 0 else 0
        print(f"  {stage_name:40s} {stats['median']:8.4f} ms  ({pct:5.1f}%)")

    print(f"\n  {'TOTAL (full wrapper)':40s} {summary['total_triton_certimask_ms']:8.4f} ms")
    print(f"  {'Scoring-only (Triton kernel)':40s} {summary['triton_scoring_only_ms']:8.4f} ms")
    print(f"  {'Wrapper overhead':40s} {summary['wrapper_overhead_ms']:8.4f} ms  "
          f"({summary['wrapper_overhead_fraction']*100:.1f}%)")
    print(f"  {'Allocation overhead':40s} {alloc['allocation_overhead_ms']:8.4f} ms")

    # Top-k mask comparison
    topk_comp = results["topk_mask_comparison"]
    print(f"\n  {'='*60}")
    print("  TOP-K MASK COMPARISON")
    print(f"  {'='*60}")
    print(f"  {'Loop (original)':40s} {topk_comp['loop_ms']:8.4f} ms")
    print(f"  {'Vectorized':40s} {topk_comp['vectorized_ms']:8.4f} ms")
    print(f"  {'Speedup':40s} {topk_comp['speedup']:8.0f}x")
    print(f"  {'Mask exact match':40s} {topk_comp['mask_exact_match']}")

    # Fused certificate comparison
    cert_fusion = results["certificate_fusion"]
    print(f"\n  {'='*60}")
    print("  CERTIFICATE FUSION COMPARISON")
    print(f"  {'='*60}")
    print(f"  {'PyTorch certificate':40s} {cert_fusion['pytorch_certificate_ms']:8.4f} ms")
    fused_ms = cert_fusion['fused_triton_certificate_ms']
    print(f"  {'Fused Triton certificate':40s} {fused_ms:8.4f} ms")
    print(f"  {'Certificate speedup':40s} {cert_fusion['certificate_speedup']:8.0f}x")
    print(f"  {'Decisions match':40s} {cert_fusion['decisions_match']}")
    print(f"  {'Ambiguous match':40s} {cert_fusion['ambiguous_match']}")
    opt_ms = summary['optimized_total_with_fused_cert_ms']
    print(f"  {'Optimized total (fused+vec)':40s} {opt_ms:8.4f} ms")

    print(f"\n  Largest bottleneck: {summary['largest_bottleneck']}")
    print(f"  Recommendation: {summary['recommended_next_step']}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
