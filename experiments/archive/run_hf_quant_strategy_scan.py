#!/usr/bin/env python3
"""Phase 5.5: Per-group and K-only quantization strategy scan.

Compares multiple quantization strategies on conditional/fallback layers
to determine which can push refinement rate below 20%.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from certimask.block_summary import expand_kv_heads, mean_pool_qk_blocks
from certimask.bounds import (
    compute_coordinate_score_bounds,
    compute_group_quantized_coordinate_bounds,
    compute_k_only_per_group_bounds,
    compute_k_only_per_vector_bounds,
    validate_score_bounds,
)
from certimask.diagnostics import (
    compute_per_tile_diagnostics,
    compute_refinement_decomposition,
)
from certimask.masking import (
    certified_threshold_mask,
    make_block_causal_valid_mask,
    naive_quantized_mask,
    reference_mask,
    thresholds_for_target_sparsity,
)
from certimask.metrics import compute_mask_metrics
from certimask.scoring import (
    group_quantized_int8_scores,
    k_only_per_group_scores,
    k_only_per_vector_scores,
    quantized_int8_scores,
    reference_scores,
)

DEFAULT_TEXT = (
    "The transformer architecture has revolutionized natural language processing "
    "by introducing self-attention mechanisms that allow models to weigh the "
    "importance of different parts of the input sequence. This approach has led "
    "to significant improvements in various tasks including translation, "
    "summarization, and question answering. Modern large language models build "
    "upon this foundation with increasingly sophisticated techniques for "
    "handling long contexts and improving computational efficiency. "
    "Recent advances in quantization have shown that it is possible to reduce "
    "the precision of model weights and activations while maintaining acceptable "
    "quality. INT8 quantization in particular has become widely adopted for "
    "inference acceleration, as it offers a good balance between memory savings "
    "and computational speedup on modern hardware architectures."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CertiMask Phase 5.5 quant strategy scan")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--layers", type=int, nargs="+",
                    default=[2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--target-sparsity", type=float, default=0.85)
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str, default="outputs/phase55_quant_strategy")
    return p.parse_args()


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def prepare_text(tokenizer: object, text: str | None, ctx: int) -> torch.Tensor:
    raw = text if text is not None else DEFAULT_TEXT
    enc = tokenizer(raw, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    while ids.shape[0] < ctx:
        ids = torch.cat([ids, ids])
    return ids[:ctx].unsqueeze(0)


def classify(ref_rate: float) -> str:
    if ref_rate < 0.20:
        return "Go"
    if ref_rate <= 0.30:
        return "Conditional Go"
    return "FP16 fallback"


def evaluate_strategy(
    ref_scores: torch.Tensor,
    scores: torch.Tensor,
    bounds_obj: object,
    valid_mask: torch.Tensor,
    thresholds: torch.Tensor,
    ref_mask: torch.Tensor,
) -> dict[str, float | int]:
    """Evaluate a single strategy and return metrics."""
    from certimask.bounds import ScoreBounds
    assert isinstance(bounds_obj, ScoreBounds)

    violations = validate_score_bounds(ref_scores, bounds_obj)
    cert = certified_threshold_mask(bounds_obj, ref_scores, thresholds, valid_mask=valid_mask)
    naive = naive_quantized_mask(scores, thresholds, valid_mask=valid_mask)
    mm = compute_mask_metrics(ref_mask, naive, cert, valid_mask=valid_mask)

    diag = compute_per_tile_diagnostics(
        ref_scores, scores, bounds_obj, bounds_obj, thresholds, valid_mask=valid_mask,
    )
    decomp = compute_refinement_decomposition(diag)

    vm = diag.valid_mask
    err_vals = diag.score_error[vm]
    bound_vals = bounds_obj.error_bound[vm]
    margin_vals = diag.margin[vm]

    rho_vals = bound_vals / (err_vals + 1e-12)

    return {
        "valid_tiles": int(vm.sum().item()),
        "actual_sparsity": mm.actual_sparsity,
        "naive_mismatch_rate": mm.naive_mismatch_rate,
        "false_drop_rate": mm.false_drop_rate,
        "false_keep_rate": mm.false_keep_rate,
        "refinement_rate": decomp.analytic_refinement_rate,
        "certimask_exact_match_rate": mm.certimask_match_rate,
        "certificate_violations": int(violations.sum().item()),
        "bound_p50": float(bound_vals.quantile(0.50).item()),
        "bound_p90": float(bound_vals.quantile(0.90).item()),
        "bound_p99": float(bound_vals.quantile(0.99).item()),
        "margin_over_bound_p50": float((margin_vals / (bound_vals + 1e-12)).quantile(0.50).item()),
        "margin_over_bound_p90": float((margin_vals / (bound_vals + 1e-12)).quantile(0.90).item()),
        "score_error_p50": float(err_vals.quantile(0.50).item()),
        "score_error_p90": float(err_vals.quantile(0.90).item()),
        "score_error_p99": float(err_vals.quantile(0.99).item()),
        "rho_p50": float(rho_vals.quantile(0.50).item()),
        "rho_p90": float(rho_vals.quantile(0.90).item()),
        "rho_p99": float(rho_vals.quantile(0.99).item()),
        "rho_max": float(rho_vals.max().item()),
    }


def main() -> None:
    args = parse_args()
    dtype = DTYPE_MAP.get(args.dtype)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as err:
        raise ImportError("transformers required") from err

    import transformers

    print(f"transformers: {transformers.__version__}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype,
        attn_implementation="eager", trust_remote_code=True,
    )
    model.to(args.device).eval()

    input_ids = prepare_text(tokenizer, args.text, args.context_length).to(args.device)
    print(f"Sequence length: {input_ids.shape[1]}")

    from certimask.hf_extraction import extract_qk_from_qwen2

    # Define strategies
    strategies = [
        ("baseline_per_vector_qk_coordinate_analytic", "qk", "per_vector", 0),
        ("qk_per_group_g32_coordinate_analytic", "qk", "per_group", 32),
        ("qk_per_group_g16_coordinate_analytic", "qk", "per_group", 16),
        ("qk_per_group_g8_coordinate_analytic", "qk", "per_group", 8),
        ("qk_per_group_g4_coordinate_analytic", "qk", "per_group", 4),
        ("k_only_per_vector_coordinate_analytic", "k_only", "per_vector", 0),
        ("k_only_per_group_g16_coordinate_analytic", "k_only", "per_group", 16),
        ("k_only_per_group_g8_coordinate_analytic", "k_only", "per_group", 8),
        ("k_only_per_group_g4_coordinate_analytic", "k_only", "per_group", 4),
    ]

    all_results = []
    baseline_ref: dict[int, float] = {}

    for layer_idx in args.layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        extracted = extract_qk_from_qwen2(model, input_ids, layer_index=layer_idx)
        q_full = extracted.query
        k_full = expand_kv_heads(extracted.key, extracted.num_query_heads)

        summaries = mean_pool_qk_blocks(q_full, k_full, block_size=args.block_size)
        num_blocks = summaries.num_blocks

        ref_scores = reference_scores(summaries.query, summaries.key, scale_by_sqrt_dim=True)
        valid_mask = make_block_causal_valid_mask(
            num_blocks, num_blocks, device=args.device,
        ).expand_as(ref_scores)
        thresholds = thresholds_for_target_sparsity(
            ref_scores, args.target_sparsity, valid_mask=valid_mask, per_query=True,
        )
        ref_mask = reference_mask(ref_scores, thresholds, valid_mask=valid_mask)

        for strat_name, quant_side, quant_mode, group_size in strategies:
            if quant_side == "qk" and quant_mode == "per_vector":
                result = quantized_int8_scores(
                    summaries.query, summaries.key, scale_by_sqrt_dim=True,
                )
                bounds = compute_coordinate_score_bounds(
                    result.scores, result.query_quantized, result.key_quantized,
                    certificate_type="analytic", scale_by_sqrt_dim=True,
                )
                metrics = evaluate_strategy(
                    ref_scores, result.scores, bounds, valid_mask, thresholds, ref_mask,
                )

            elif quant_side == "qk" and quant_mode == "per_group":
                g_result = group_quantized_int8_scores(
                    summaries.query, summaries.key, group_size=group_size, scale_by_sqrt_dim=True,
                )
                bounds = compute_group_quantized_coordinate_bounds(
                    g_result.scores, g_result.query_quantized, g_result.key_quantized,
                    certificate_type="analytic", scale_by_sqrt_dim=True,
                )
                metrics = evaluate_strategy(
                    ref_scores, g_result.scores, bounds, valid_mask, thresholds, ref_mask,
                )

            elif quant_side == "k_only" and quant_mode == "per_vector":
                ko_result = k_only_per_vector_scores(
                    summaries.query, summaries.key, scale_by_sqrt_dim=True,
                )
                bounds = compute_k_only_per_vector_bounds(
                    ko_result.scores, ko_result.query, ko_result.key_quantized,
                    certificate_type="analytic", scale_by_sqrt_dim=True,
                )
                metrics = evaluate_strategy(
                    ref_scores, ko_result.scores, bounds, valid_mask, thresholds, ref_mask,
                )

            elif quant_side == "k_only" and quant_mode == "per_group":
                ko_result = k_only_per_group_scores(
                    summaries.query, summaries.key, group_size=group_size, scale_by_sqrt_dim=True,
                )
                bounds = compute_k_only_per_group_bounds(
                    ko_result.scores, ko_result.query, ko_result.key_quantized,
                    certificate_type="analytic", scale_by_sqrt_dim=True,
                )
                metrics = evaluate_strategy(
                    ref_scores, ko_result.scores, bounds, valid_mask, thresholds, ref_mask,
                )
            else:
                continue

            row = {
                "layer": layer_idx,
                "strategy": strat_name,
                "group_size": group_size,
                "quantized_side": quant_side,
                "certificate_type": "analytic",
                **metrics,
            }
            all_results.append(row)

            if strat_name == "baseline_per_vector_qk_coordinate_analytic":
                baseline_ref[layer_idx] = metrics["refinement_rate"]

            print(
                f"  {strat_name:45s} ref={metrics['refinement_rate']:.4f}"
                f"  naive_mm={metrics['naive_mismatch_rate']:.4f}"
                f"  viol={metrics['certificate_violations']}"
                f"  match={metrics['certimask_exact_match_rate']:.4f}"
            )

    # Add reduction vs baseline
    for row in all_results:
        layer = row["layer"]
        if layer in baseline_ref:
            row["refinement_reduction_vs_baseline"] = baseline_ref[layer] - row["refinement_rate"]
            row["naive_mismatch_reduction_vs_baseline"] = 0.0  # computed below

    # Compute naive mismatch reduction
    for row in all_results:
        layer = row["layer"]
        baseline_rows = [
            r for r in all_results
            if r["layer"] == layer
            and r["strategy"] == "baseline_per_vector_qk_coordinate_analytic"
        ]
        if baseline_rows:
            base_mm = baseline_rows[0]["naive_mismatch_rate"]
            row["naive_mismatch_reduction_vs_baseline"] = base_mm - row["naive_mismatch_rate"]

    # Find best strategy per layer
    best_per_layer: dict[int, dict[str, object]] = {}
    for layer_idx in args.layers:
        layer_rows = [r for r in all_results if r["layer"] == layer_idx]
        if not layer_rows:
            continue
        # Filter rows with 0 violations
        valid_rows = [r for r in layer_rows if r["certificate_violations"] == 0]
        if not valid_rows:
            best_per_layer[layer_idx] = {
                "best_strategy": "none",
                "best_refinement_rate": 1.0,
                "best_naive_mismatch_rate": 1.0,
                "decision": "FP16 fallback",
                "improved": False,
                "score_error_regression": True,
            }
            continue
        best = min(valid_rows, key=lambda r: r["refinement_rate"])
        baseline_r = baseline_ref.get(layer_idx, 1.0)
        best_per_layer[layer_idx] = {
            "best_strategy": best["strategy"],
            "best_refinement_rate": best["refinement_rate"],
            "best_naive_mismatch_rate": best["naive_mismatch_rate"],
            "decision": classify(best["refinement_rate"]),
            "improved": best["refinement_rate"] < baseline_r - 0.001,
            "score_error_regression": (
                best["naive_mismatch_rate"] > baseline_r * 1.5
                if layer_idx in baseline_ref else False
            ),
        }

    # Classify layers
    go_layers = [
        layer for layer, info in best_per_layer.items() if info["decision"] == "Go"
    ]
    conditional_layers = [
        layer for layer, info in best_per_layer.items()
        if info["decision"] == "Conditional Go"
    ]
    fallback_layers = [
        layer for layer, info in best_per_layer.items()
        if info["decision"] == "FP16 fallback"
    ]
    layers_improved = [
        layer for layer, info in best_per_layer.items() if info["improved"]
    ]
    layers_not_improved = [
        layer for layer, info in best_per_layer.items() if not info["improved"]
    ]

    # Count best strategies
    strategy_counts: dict[str, int] = {}
    for info in best_per_layer.values():
        s = str(info["best_strategy"])
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    # Estimate FP16 fraction
    total = len(best_per_layer)
    fp16_work = 0.0
    for info in best_per_layer.values():
        if info["decision"] == "FP16 fallback":
            fp16_work += 1.0
        else:
            fp16_work += info["best_refinement_rate"]
    est_fp16_frac = fp16_work / total if total > 0 else 0.0

    # Save outputs
    config = {
        "model_name": args.model_name,
        "layers": args.layers,
        "context_length": args.context_length,
        "block_size": args.block_size,
        "target_sparsity": args.target_sparsity,
        "device": args.device,
        "dtype": args.dtype,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if all_results:
        fields = list(all_results[0].keys())
        with open(output_dir / "strategy_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_results)

    # best_layer_strategy.csv
    best_rows = []
    for layer_idx in sorted(best_per_layer.keys()):
        info = best_per_layer[layer_idx]
        best_rows.append({
            "layer": layer_idx,
            "baseline_refinement_rate": baseline_ref.get(layer_idx),
            "best_strategy": info["best_strategy"],
            "best_refinement_rate": info["best_refinement_rate"],
            "best_naive_mismatch_rate": info["best_naive_mismatch_rate"],
            "decision": info["decision"],
            "improved": info["improved"],
        })
    if best_rows:
        with open(output_dir / "best_layer_strategy.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
            w.writeheader()
            w.writerows(best_rows)

    strategy_summary = {
        "go_layers": sorted(go_layers),
        "conditional_layers": sorted(conditional_layers),
        "fallback_layers": sorted(fallback_layers),
        "estimated_fp16_refinement_fraction": est_fp16_frac,
        "best_strategy_counts": strategy_counts,
        "layers_improved": sorted(layers_improved),
        "layers_not_improved": sorted(layers_not_improved),
        "note": (
            "This is not a latency estimate. "
            "estimated_fp16_refinement_fraction approximates FP16 scoring work fraction."
        ),
    }
    with open(output_dir / "strategy_summary.json", "w") as f:
        json.dump(strategy_summary, f, indent=2)

    # Print summary
    print()
    print("=" * 70)
    print("STRATEGY SCAN SUMMARY")
    print("=" * 70)
    print(f"  Go layers:            {sorted(go_layers)}")
    print(f"  Conditional layers:   {sorted(conditional_layers)}")
    print(f"  Fallback layers:      {sorted(fallback_layers)}")
    print(f"  Layers improved:      {sorted(layers_improved)}")
    print(f"  Est. FP16 fraction:   {est_fp16_frac:.4f}")
    print(f"  Best strategy counts: {strategy_counts}")
    print()
    for row in best_rows:
        bl = row["baseline_refinement_rate"]
        br = row["best_refinement_rate"]
        bs = row["best_strategy"]
        dec = row["decision"]
        imp = "  IMPROVED" if row["improved"] else ""
        print(f"  L{row['layer']:2d}  baseline={bl:.4f}  best={br:.4f}  {bs:45s} [{dec}]{imp}")
    print()
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
