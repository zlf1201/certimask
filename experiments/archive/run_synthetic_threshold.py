#!/usr/bin/env python3
"""Synthetic threshold experiment for CertiMask Phase 3.

Generates synthetic Q/K blocks, computes reference and quantized scores,
applies threshold masking, and evaluates mask quality and bound tightness.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from certimask.bounds import compute_score_bounds, validate_score_bounds
from certimask.masking import (
    certified_threshold_mask,
    make_block_causal_valid_mask,
    naive_quantized_mask,
    reference_mask,
    thresholds_for_target_sparsity,
)
from certimask.metrics import compute_bound_metrics, compute_mask_metrics
from certimask.scoring import quantized_int8_scores, reference_scores
from certimask.synthetic import generate_synthetic_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CertiMask synthetic threshold experiment"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-query-blocks", type=int, default=128)
    parser.add_argument("--num-key-blocks", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--distribution", type=str, default="normal")
    parser.add_argument(
        "--target-sparsities",
        type=float,
        nargs="+",
        default=[0.70, 0.80, 0.85, 0.90, 0.95],
    )
    parser.add_argument(
        "--certificate-types",
        type=str,
        nargs="+",
        default=["actual", "analytic"],
    )
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/phase3")
    return parser.parse_args()


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def main() -> None:
    args = parse_args()

    dtype = DTYPE_MAP.get(args.dtype)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "batch_size": args.batch_size,
        "num_heads": args.num_heads,
        "num_query_blocks": args.num_query_blocks,
        "num_key_blocks": args.num_key_blocks,
        "head_dim": args.head_dim,
        "distribution": args.distribution,
        "target_sparsities": args.target_sparsities,
        "certificate_types": args.certificate_types,
        "causal": args.causal,
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate synthetic Q/K
    query, key = generate_synthetic_summaries(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        num_query_blocks=args.num_query_blocks,
        num_key_blocks=args.num_key_blocks,
        head_dim=args.head_dim,
        distribution=args.distribution,
        seed=args.seed,
        device=args.device,
        dtype=dtype,
    )

    # Compute scores
    ref_scores = reference_scores(query, key, scale_by_sqrt_dim=True)
    q_result = quantized_int8_scores(query, key, scale_by_sqrt_dim=True)

    # Valid mask — expand [1,1,Q,K] to [B,H,Q,K] for boolean indexing
    valid_mask = None
    if args.causal:
        valid_mask = make_block_causal_valid_mask(
            args.num_query_blocks,
            args.num_key_blocks,
            device=args.device,
        ).expand_as(ref_scores)

    # Collect results
    results = []

    for cert_type in args.certificate_types:
        bounds = compute_score_bounds(
            q_result.scores,
            q_result.query_quantized,
            q_result.key_quantized,
            certificate_type=cert_type,  # type: ignore[arg-type]
            scale_by_sqrt_dim=True,
        )

        # Validate score bounds
        validate_score_bounds(ref_scores, bounds)

        for target_sp in args.target_sparsities:
            # Compute threshold from reference scores
            thresholds = thresholds_for_target_sparsity(
                ref_scores,
                target_sp,
                valid_mask=valid_mask,
                per_query=True,
            )

            # Reference mask
            ref_mask = reference_mask(ref_scores, thresholds, valid_mask=valid_mask)

            # Naive INT8 mask
            naive_mask = naive_quantized_mask(
                q_result.scores, thresholds, valid_mask=valid_mask
            )

            # CertiMask
            cert_result = certified_threshold_mask(
                bounds,
                ref_scores,
                thresholds,
                valid_mask=valid_mask,
            )

            # Mask metrics
            mask_metrics = compute_mask_metrics(
                ref_mask, naive_mask, cert_result, valid_mask=valid_mask
            )

            # Bound metrics
            bound_metrics = compute_bound_metrics(
                ref_scores,
                q_result.scores,
                bounds,
                thresholds,
                valid_mask=valid_mask,
            )

            row = {
                "seed": args.seed,
                "distribution": args.distribution,
                "certificate_type": cert_type,
                "target_sparsity": target_sp,
                "actual_sparsity": mask_metrics.actual_sparsity,
                "valid_tiles": mask_metrics.valid_tiles,
                "certificate_violations": bound_metrics.violations,
                "naive_mismatch_count": mask_metrics.naive_mismatch_count,
                "naive_mismatch_rate": mask_metrics.naive_mismatch_rate,
                "false_drop_count": mask_metrics.false_drop_count,
                "false_drop_rate": mask_metrics.false_drop_rate,
                "false_keep_count": mask_metrics.false_keep_count,
                "false_keep_rate": mask_metrics.false_keep_rate,
                "certimask_mismatch_count": mask_metrics.certimask_mismatch_count,
                "certimask_match_rate": mask_metrics.certimask_match_rate,
                "certain_keep_count": mask_metrics.certain_keep_count,
                "certain_drop_count": mask_metrics.certain_drop_count,
                "ambiguous_count": mask_metrics.ambiguous_count,
                "refinement_rate": mask_metrics.refinement_rate,
                "rho_mean": bound_metrics.rho_mean,
                "rho_p50": bound_metrics.rho_p50,
                "rho_p90": bound_metrics.rho_p90,
                "rho_p99": bound_metrics.rho_p99,
                "rho_max": bound_metrics.rho_max,
                "margin_over_bound_mean": bound_metrics.margin_over_bound_mean,
                "margin_over_bound_p50": bound_metrics.margin_over_bound_p50,
                "margin_over_bound_p90": bound_metrics.margin_over_bound_p90,
            }
            results.append(row)

            print(
                f"  cert={cert_type:8s} sparsity={target_sp:.2f} "
                f"actual_sp={mask_metrics.actual_sparsity:.4f} "
                f"naive_mismatch={mask_metrics.naive_mismatch_count} "
                f"false_drop={mask_metrics.false_drop_count} "
                f"false_keep={mask_metrics.false_keep_count} "
                f"certimask_match={mask_metrics.certimask_match_rate:.4f} "
                f"refine={mask_metrics.refinement_rate:.4f} "
                f"rho_p90={bound_metrics.rho_p90:.4f} "
                f"rho_max={bound_metrics.rho_max:.4f}"
            )

    # Save CSV
    csv_path = output_dir / "results.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Save summary
    summary = {
        "total_configs": len(results),
        "all_violations_zero": all(
            r["certificate_violations"] == 0 for r in results
        ),
        "all_exact_match": all(
            r["certimask_match_rate"] == 1.0 for r in results
        ),
        "configs": results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"Total configs: {len(results)}")
    print(
        f"All violations zero: {summary['all_violations_zero']}"
    )
    print(f"All exact match: {summary['all_exact_match']}")


if __name__ == "__main__":
    main()
