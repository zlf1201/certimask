#!/usr/bin/env python3
"""Phase 9C: Top-k mask construction benchmark.

Compares loop-based vs vectorized top-k mask implementations.

Usage:
    python experiments/benchmark_topk_mask.py \
        --batch-size 1 --num-heads 14 --seq-len 1024 \
        --block-size 8 --device cuda --warmup 20 --iters 100 \
        --output-dir outputs/phase9c_topk_mask
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from certimask.masking import make_block_causal_valid_mask
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


def _make_synthetic_scores(
    batch: int, heads: int, q_blk: int, k_blk: int,
    dtype: torch.dtype, device: str,
) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(42)
    return torch.randn(batch, heads, q_blk, k_blk, dtype=dtype, device=device, generator=gen)


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


def _loop_topk_mask(
    scores: torch.Tensor,
    *,
    target_sparsity: float,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Original loop-based top-k mask (from aglr_local_plus_landmark_mask logic)."""
    batch, heads, q_blk, k_blk = scores.shape

    valid_per_row = valid_mask.sum(dim=-1)
    keep_per_row = (valid_per_row.float() * (1.0 - target_sparsity)).ceil().long()
    keep_per_row = torch.clamp(keep_per_row, min=1)

    mask = torch.zeros_like(valid_mask, dtype=torch.bool)

    for b in range(batch):
        for h in range(heads):
            for q in range(q_blk):
                n_keep = int(keep_per_row[b, h, q].item())
                valid_k = valid_mask[b, h, q]

                scores_row = scores[b, h, q].clone()
                scores_row[~valid_k] = torch.finfo(torch.float32).min

                n_available = int((scores_row > torch.finfo(torch.float32).min).sum().item())
                n_extra = min(n_keep, n_available)

                if n_extra > 0:
                    _, extra_idx = scores_row.topk(n_extra)
                    mask[b, h, q, extra_idx] = True

    return mask


def run_benchmark(args: argparse.Namespace) -> dict:
    cuda_ok, triton_ok = _check_env()
    if not cuda_ok:
        raise RuntimeError("CUDA not available")

    device = args.device
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    block_size = args.block_size

    batch = args.batch_size
    heads = args.num_heads
    seq_len = args.seq_len
    num_blocks = seq_len // block_size

    valid_mask = make_block_causal_valid_mask(
        num_blocks, num_blocks, device=device,
    ).expand(batch, heads, num_blocks, num_blocks)

    scores = _make_synthetic_scores(
        batch, heads, num_blocks, num_blocks, dtype, device,
    )
    scores_f32 = scores.float()

    target_sparsity = 0.5
    warmup = args.warmup
    iters = args.iters

    print(f"Shape: B={batch}, H={heads}, Q_blk={num_blocks}, K_blk={num_blocks}")
    print(f"Target sparsity: {target_sparsity}")
    print(f"Warmup: {warmup}, Iters: {iters}")

    # --- Loop-based ---
    def stage_loop():
        return _loop_topk_mask(
            scores_f32, target_sparsity=target_sparsity, valid_mask=valid_mask,
        )

    t_loop = _cuda_timer(stage_loop, warmup, iters)
    loop_mask = stage_loop()

    # --- Vectorized ---
    valid_per_row = valid_mask.sum(dim=-1)
    k_per_row = (valid_per_row.float() * (1.0 - target_sparsity)).ceil().long()
    k_per_row = torch.clamp(k_per_row, min=1)

    def stage_vectorized():
        return vectorized_topk_mask(
            scores_f32, k_per_row=k_per_row, valid_mask=valid_mask,
        )

    t_vectorized = _cuda_timer(stage_vectorized, warmup, iters)
    vec_result = stage_vectorized()
    vec_mask = vec_result.mask

    # Verify correctness
    mask_match = torch.equal(loop_mask, vec_mask)
    mismatch_count = int((loop_mask != vec_mask).sum().item())

    stats_loop = _stats(t_loop)
    stats_vec = _stats(t_vectorized)

    speedup = stats_loop["median"] / stats_vec["median"] if stats_vec["median"] > 0 else 0.0

    results = {
        "config": vars(args),
        "shape": {"B": batch, "H": heads, "Q_blk": num_blocks, "K_blk": num_blocks},
        "target_sparsity": target_sparsity,
        "loop": stats_loop,
        "vectorized": stats_vec,
        "speedup_vectorized_vs_loop": speedup,
        "mask_exact_match": mask_match,
        "mismatch_count": mismatch_count,
        "selected_mode": "vectorized",
    }

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 9C top-k mask benchmark")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=14)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "float32", "bfloat16"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--output-dir", type=str,
                    default="outputs/phase9c_topk_mask")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 9C: Top-k Mask Construction Benchmark")
    print("=" * 60)

    results = run_benchmark(args)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(results["config"], f, indent=2)

    # Save full results
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # CSV
    csv_lines = [
        "mode,median_ms,p10_ms,p90_ms,mean_ms,std_ms",
        f"loop,{results['loop']['median']:.4f},{results['loop']['p10']:.4f},"
        f"{results['loop']['p90']:.4f},{results['loop']['mean']:.4f},{results['loop']['std']:.4f}",
        f"vectorized,{results['vectorized']['median']:.4f},{results['vectorized']['p10']:.4f},"
        f"{results['vectorized']['p90']:.4f},{results['vectorized']['mean']:.4f},{results['vectorized']['std']:.4f}",
    ]
    with open(output_dir / "topk_mask_benchmark.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    # Correctness CSV
    csv_corr = [
        "mode,mask_exact_match,mismatch_count",
        f"vectorized_vs_loop,{results['mask_exact_match']},{results['mismatch_count']}",
    ]
    with open(output_dir / "topk_mask_correctness.csv", "w") as f:
        f.write("\n".join(csv_corr) + "\n")

    # Print summary
    print("\n" + "=" * 60)
    print("TOP-K MASK BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  {'Loop (original)':40s} {results['loop']['median']:8.4f} ms")
    print(f"  {'Vectorized':40s} {results['vectorized']['median']:8.4f} ms")
    print(f"  {'Speedup':40s} {results['speedup_vectorized_vs_loop']:8.1f}x")
    print(f"  {'Mask exact match':40s} {results['mask_exact_match']}")
    print(f"  {'Mismatch count':40s} {results['mismatch_count']}")
    print(f"\n  Selected mode: {results['selected_mode']}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
