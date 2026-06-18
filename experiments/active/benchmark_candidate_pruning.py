"""Benchmark candidate-pruned AGLR-C v2 indexer modes.

Measures candidate fraction, teacher coverage, and latency for each
candidate generation mode at various sequence lengths.

Usage:
    python experiments/active/benchmark_candidate_pruning.py \
        --batch-size 1 --num-heads 14 --head-dim 64 \
        --seq-lens 1024 2048 4096 \
        --block-size 8 --device cuda --dtype float16 \
        --modes local_stride block_norm coarse_to_fine head_pattern \
        --output-dir outputs/phase10a_candidate_pruning
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from certimask.aglr_indexer import compute_antidiagonal_block_scores
from certimask.candidate_pruning import (
    compute_candidate_antidiagonal_scores,
    compute_teacher_mask_overlap,
    compute_teacher_selected_coverage,
    generate_candidate_mask,
)
from certimask.masking import make_block_causal_valid_mask
from certimask.vectorized_topk import vectorized_topk_mask


def _ms(t: torch.cuda.Event, t2: torch.cuda.Event) -> float:
    """Elapsed time between two CUDA events in ms."""
    return t.elapsed_time(t2)


def _time_fn(fn, *args, warmup: int = 2, repeats: int = 5, **kwargs):
    """Time a function with CUDA events. Returns (result, median_ms)."""
    for _ in range(warmup):
        result = fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(_ms(start, end))

    return result, sorted(times)[len(times) // 2]


def _dense_sdpa_latency(
    query: torch.Tensor,
    key: torch.Tensor,
    warmup: int = 2,
    repeats: int = 5,
) -> float:
    """Measure dense FP16 SDPA latency in ms."""
    value = key  # dummy value
    for _ in range(warmup):
        torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True,
        )
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True,
        )
        end.record()
        torch.cuda.synchronize()
        times.append(_ms(start, end))

    return sorted(times)[len(times) // 2]


def run_benchmark(
    *,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    block_size: int,
    device: str,
    dtype: torch.dtype,
    modes: list[str],
) -> list[dict]:
    """Run benchmark for one sequence length across all modes."""
    results = []

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    num_blocks = seq_len // block_size
    valid_mask = make_block_causal_valid_mask(
        num_blocks, num_blocks, device=device,
    ).expand(batch_size, num_heads, num_blocks, num_blocks)

    # Dense SDPA baseline
    dense_sdpa_ms = _dense_sdpa_latency(query, key)

    # Teacher: full-pair AGLR-C v1 scoring + top-k
    def teacher_fn():
        fp_scores = compute_antidiagonal_block_scores(
            query, key, block_size=block_size,
            sample_pattern="both_diagonals", aggregation="logsumexp",
            valid_mask=valid_mask, scale_by_sqrt_dim=True,
        )
        # Top-k with 25% keep fraction
        valid_per_row = valid_mask.sum(dim=-1)
        k_per_row = (valid_per_row.float() * 0.25).ceil().long()
        k_per_row = torch.clamp(k_per_row, min=1)
        result = vectorized_topk_mask(
            fp_scores, k_per_row=k_per_row, valid_mask=valid_mask,
        )
        return fp_scores, result.mask

    (teacher_scores, teacher_mask), teacher_ms = _time_fn(teacher_fn)

    for mode in modes:
        row: dict = {
            "seq_len": seq_len,
            "mode": mode,
            "batch_size": batch_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "dense_sdpa_fp16_ms": round(dense_sdpa_ms, 4),
            "teacher_full_pair_ms": round(teacher_ms, 4),
        }

        # Step 1: Candidate generation
        def candidate_gen_fn(_mode=mode):
            return generate_candidate_mask(
                query, key,
                mode=_mode, block_size=block_size, valid_mask=valid_mask,
                target_candidate_fraction=0.25,
                local_blocks=4, stride=16,
                coarse_block_size=64, topk_coarse=8,
            )

        candidate_result, cand_gen_ms = _time_fn(candidate_gen_fn)
        cand_mask_tensor = candidate_result.candidate_mask
        row["candidate_generation_ms"] = round(cand_gen_ms, 4)
        row["candidate_fraction"] = round(candidate_result.candidate_fraction, 4)
        c_tiles = int(candidate_result.metadata.get("candidate_tiles", 0))
        v_tiles = int(candidate_result.metadata.get("valid_tiles", 0))
        row["candidate_tiles"] = c_tiles
        row["valid_tiles"] = v_tiles
        row["uses_full_pair_scoring"] = candidate_result.metadata.get(
            "uses_full_pair_scoring", False,
        )
        row["uses_full_pair_proxy"] = candidate_result.metadata.get(
            "uses_full_pair_proxy", False,
        )

        # Step 2: Candidate-only AGLR scoring
        def scoring_fn(_cm=cand_mask_tensor):
            return compute_candidate_antidiagonal_scores(
                query, key, _cm,
                block_size=block_size,
                sample_pattern="both_diagonals", aggregation="logsumexp",
                scale_by_sqrt_dim=True,
            )

        (cand_scores, scoring_meta), cand_scoring_ms = _time_fn(scoring_fn)
        row["candidate_scoring_ms"] = round(cand_scoring_ms, 4)
        row["computed_tile_count"] = int(
            scoring_meta.get("computed_tile_count", 0),
        )

        # Step 3: Top-k on candidate scores
        def topk_fn(_cs=cand_scores, _cm=cand_mask_tensor):
            valid_per_row = valid_mask.sum(dim=-1)
            k_per_row = (valid_per_row.float() * 0.25).ceil().long()
            k_per_row = torch.clamp(k_per_row, min=1)
            masked_scores = _cs.clone()
            masked_scores[~_cm] = float("-inf")
            result = vectorized_topk_mask(
                masked_scores, k_per_row=k_per_row, valid_mask=valid_mask,
            )
            return result.mask

        cand_mask, topk_ms = _time_fn(topk_fn)
        row["topk_mask_ms"] = round(topk_ms, 4)

        # Total indexer latency
        total_ms = cand_gen_ms + cand_scoring_ms + topk_ms
        row["total_indexer_ms"] = round(total_ms, 4)
        row["speedup_proxy_vs_dense"] = round(dense_sdpa_ms / total_ms, 4) if total_ms > 0 else 0.0

        # Teacher coverage
        coverage = compute_teacher_selected_coverage(
            candidate_result.candidate_mask, teacher_mask, valid_mask,
        )
        row["teacher_selected_coverage"] = round(coverage, 4)

        overlap = compute_teacher_mask_overlap(cand_mask, teacher_mask, valid_mask)
        row["teacher_mask_overlap"] = round(overlap, 4)

        results.append(row)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark candidate pruning modes")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=14)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument(
        "--modes", type=str, nargs="+",
        default=["local_stride", "block_norm", "coarse_to_fine", "head_pattern"],
    )
    parser.add_argument("--output-dir", type=str, default="outputs/phase10a_candidate_pruning")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seq_len in args.seq_lens:
        print(f"\n=== seq_len={seq_len} ===")
        results = run_benchmark(
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            seq_len=seq_len,
            block_size=args.block_size,
            device=args.device,
            dtype=dtype,
            modes=args.modes,
        )
        for r in results:
            print(
                f"  {r['mode']:20s} | "
                f"cand_frac={r['candidate_fraction']:.3f} | "
                f"coverage={r['teacher_selected_coverage']:.3f} | "
                f"overlap={r['teacher_mask_overlap']:.3f} | "
                f"gen={r['candidate_generation_ms']:.2f}ms | "
                f"score={r['candidate_scoring_ms']:.2f}ms | "
                f"topk={r['topk_mask_ms']:.2f}ms | "
                f"total={r['total_indexer_ms']:.2f}ms | "
                f"dense={r['dense_sdpa_fp16_ms']:.2f}ms | "
                f"speedup={r['speedup_proxy_vs_dense']:.3f}"
            )
        all_results.extend(results)

    # Save results
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'mode':20s} | {'L':>5s} | {'cand%':>6s} | {'cov':>5s} | {'overlap':>7s} | "
          f"{'total_ms':>9s} | {'dense_ms':>9s} | {'speedup':>7s}")
    print("-" * 95)
    for r in all_results:
        print(
            f"{r['mode']:20s} | {r['seq_len']:>5d} | "
            f"{r['candidate_fraction'] * 100:>5.1f}% | "
            f"{r['teacher_selected_coverage']:>.3f} | "
            f"{r['teacher_mask_overlap']:>.3f}  | "
            f"{r['total_indexer_ms']:>8.2f} | "
            f"{r['dense_sdpa_fp16_ms']:>8.2f} | "
            f"{r['speedup_proxy_vs_dense']:>6.3f}"
        )


if __name__ == "__main__":
    main()
