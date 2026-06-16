#!/usr/bin/env python3
"""Phase 5.6: K-only Per-Group Full Layer Scan.

Scans all 24 layers of Qwen2.5-0.5B-Instruct with K-only per-group
quantization strategies to determine the final layer-adaptive policy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from certimask.block_summary import expand_kv_heads, mean_pool_qk_blocks
from certimask.bounds import (
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
    p = argparse.ArgumentParser(
        description="CertiMask Phase 5.6 K-only full layer scan"
    )
    p.add_argument(
        "--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--target-sparsity", type=float, default=0.85)
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument(
        "--output-dir", type=str, default="outputs/phase56_konly_full_scan"
    )
    return p.parse_args()


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def prepare_text(
    tokenizer: object, text: str | None, ctx: int
) -> torch.Tensor:
    raw = text if text is not None else DEFAULT_TEXT
    enc = tokenizer(raw, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    while ids.shape[0] < ctx:
        ids = torch.cat([ids, ids])
    return ids[:ctx].unsqueeze(0)


def classify(ref_rate: float) -> str:
    """Classify layer based on refinement rate."""
    if ref_rate < 0.20:
        return "Go"
    if ref_rate <= 0.30:
        return "Conditional Go"
    return "FP16 fallback"


def select_group_size(results: dict[str, float]) -> tuple[str, int]:
    """Select the largest group size that achieves <20% refinement.

    Priority: g16 > g8 > g4 (prefer larger groups for lower overhead).
    Returns (strategy_name, group_size).
    """
    # Check in order of preference (largest group first)
    for gs in [16, 8, 4]:
        key = f"k_only_per_group_g{gs}_coordinate_analytic"
        if key in results and results[key] < 0.20:
            return key, gs
    # If none achieve <20%, return the best one
    best_key = min(results, key=lambda k: results[k])
    gs = _extract_group_size(best_key)
    return best_key, gs


def _extract_group_size(strategy_name: str) -> int:
    """Extract group size from strategy name."""
    # Handle patterns like "k_only_per_group_g4_coordinate_analytic"
    # or "qk_per_group_g16_coordinate_analytic"
    # or "k_only_per_vector_coordinate_analytic" (group_size=0)
    if "per_vector" in strategy_name:
        return 0
    parts = strategy_name.split("_g")
    if len(parts) >= 2:
        # After "_g", the next characters are the number
        num_str = ""
        for ch in parts[1]:
            if ch.isdigit():
                num_str += ch
            else:
                break
        if num_str:
            return int(num_str)
    return 0


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
    cert = certified_threshold_mask(
        bounds_obj, ref_scores, thresholds, valid_mask=valid_mask
    )
    naive = naive_quantized_mask(scores, thresholds, valid_mask=valid_mask)
    mm = compute_mask_metrics(
        ref_mask, naive, cert, valid_mask=valid_mask
    )

    diag = compute_per_tile_diagnostics(
        ref_scores,
        scores,
        bounds_obj,
        bounds_obj,
        thresholds,
        valid_mask=valid_mask,
    )
    decomp = compute_refinement_decomposition(diag)

    vm = diag.valid_mask
    err_vals = diag.score_error[vm]
    bound_vals = bounds_obj.error_bound[vm]
    rho_vals = bound_vals / (err_vals + 1e-12)

    # Score statistics
    ref_vals = ref_scores[vm]
    score_std = float(ref_vals.std().item())

    return {
        "valid_tiles": int(vm.sum().item()),
        "actual_sparsity": mm.actual_sparsity,
        "naive_mismatch_rate": mm.naive_mismatch_rate,
        "false_drop_rate": mm.false_drop_rate,
        "false_keep_rate": mm.false_keep_rate,
        "certificate_violations": int(violations.sum().item()),
        "certimask_exact_match_rate": mm.certimask_match_rate,
        "analytic_refinement_rate": decomp.analytic_refinement_rate,
        "actual_refinement_rate": decomp.actual_refinement_rate,
        "score_error_p50": float(err_vals.quantile(0.50).item()),
        "score_error_p90": float(err_vals.quantile(0.90).item()),
        "score_error_p99": float(err_vals.quantile(0.99).item()),
        "bound_p50": float(bound_vals.quantile(0.50).item()),
        "bound_p90": float(bound_vals.quantile(0.90).item()),
        "bound_p99": float(bound_vals.quantile(0.99).item()),
        "rho_p50": float(rho_vals.quantile(0.50).item()),
        "rho_p90": float(rho_vals.quantile(0.90).item()),
        "rho_p99": float(rho_vals.quantile(0.99).item()),
        "rho_max": float(rho_vals.max().item()),
        "score_std": score_std,
        "abs_score_p50": float(ref_vals.abs().quantile(0.50).item()),
        "abs_score_p90": float(ref_vals.abs().quantile(0.90).item()),
        "abs_score_p99": float(ref_vals.abs().quantile(0.99).item()),
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
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.to(args.device).eval()

    num_layers = len(model.model.layers)
    print(f"Model: {args.model_name}")
    print(f"Layers: {num_layers}")

    input_ids = prepare_text(
        tokenizer, args.text, args.context_length
    ).to(args.device)
    print(f"Sequence length: {input_ids.shape[1]}")

    from certimask.hf_extraction import extract_qk_from_qwen2

    # Define strategies
    strategies = [
        ("k_only_per_group_g16_coordinate_analytic", "k_only", 16),
        ("k_only_per_group_g8_coordinate_analytic", "k_only", 8),
        ("k_only_per_group_g4_coordinate_analytic", "k_only", 4),
        ("k_only_per_vector_coordinate_analytic", "k_only", 0),
        ("qk_per_group_g4_coordinate_analytic", "qk", 4),
    ]

    all_results: list[dict[str, float | int | str]] = []
    layer_refinement: dict[int, dict[str, float]] = {}

    for layer_idx in range(num_layers):
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        extracted = extract_qk_from_qwen2(
            model, input_ids, layer_index=layer_idx
        )
        q_full = extracted.query
        k_full = expand_kv_heads(extracted.key, extracted.num_query_heads)

        summaries = mean_pool_qk_blocks(
            q_full, k_full, block_size=args.block_size
        )
        num_blocks = summaries.num_blocks

        ref_scores = reference_scores(
            summaries.query, summaries.key, scale_by_sqrt_dim=True
        )
        valid_mask = make_block_causal_valid_mask(
            num_blocks, num_blocks, device=args.device
        ).expand_as(ref_scores)

        thresholds = thresholds_for_target_sparsity(
            ref_scores,
            args.target_sparsity,
            valid_mask=valid_mask,
            per_query=True,
        )
        ref_mask = reference_mask(
            ref_scores, thresholds, valid_mask=valid_mask
        )

        layer_refinement[layer_idx] = {}

        for strat_name, quant_side, group_size in strategies:
            if quant_side == "k_only" and group_size > 0:
                ko_result = k_only_per_group_scores(
                    summaries.query,
                    summaries.key,
                    group_size=group_size,
                    scale_by_sqrt_dim=True,
                )
                bounds = compute_k_only_per_group_bounds(
                    ko_result.scores,
                    ko_result.query,
                    ko_result.key_quantized,
                    certificate_type="analytic",
                    scale_by_sqrt_dim=True,
                )
                scores = ko_result.scores

            elif quant_side == "k_only" and group_size == 0:
                ko_result = k_only_per_vector_scores(
                    summaries.query,
                    summaries.key,
                    scale_by_sqrt_dim=True,
                )
                bounds = compute_k_only_per_vector_bounds(
                    ko_result.scores,
                    ko_result.query,
                    ko_result.key_quantized,
                    certificate_type="analytic",
                    scale_by_sqrt_dim=True,
                )
                scores = ko_result.scores

            elif quant_side == "qk" and group_size > 0:
                g_result = group_quantized_int8_scores(
                    summaries.query,
                    summaries.key,
                    group_size=group_size,
                    scale_by_sqrt_dim=True,
                )
                bounds = compute_group_quantized_coordinate_bounds(
                    g_result.scores,
                    g_result.query_quantized,
                    g_result.key_quantized,
                    certificate_type="analytic",
                    scale_by_sqrt_dim=True,
                )
                scores = g_result.scores
            else:
                continue

            metrics = evaluate_strategy(
                ref_scores,
                scores,
                bounds,
                valid_mask,
                thresholds,
                ref_mask,
            )

            layer_refinement[layer_idx][strat_name] = float(
                metrics["analytic_refinement_rate"]
            )

            row: dict[str, float | int | str] = {
                "layer": layer_idx,
                "strategy_name": strat_name,
                "group_size": group_size,
                **metrics,
            }
            all_results.append(row)

            ref_rate = metrics["analytic_refinement_rate"]
            viol = metrics["certificate_violations"]
            match = metrics["certimask_exact_match_rate"]
            naive_mm = metrics["naive_mismatch_rate"]
            print(
                f"  {strat_name:45s} ref={ref_rate:.4f}"
                f"  naive_mm={naive_mm:.4f}"
                f"  viol={viol}"
                f"  match={match:.4f}"
            )

    # Determine selected policy per layer
    stage5_proxy = 0.4572  # From Phase 5
    selected_policy: list[dict[str, float | int | str | bool]] = []
    fp16_work_sum = 0.0

    for layer_idx in range(num_layers):
        ref_map = layer_refinement.get(layer_idx, {})
        if not ref_map:
            continue

        # Find best refinement strategy (lowest refinement)
        best_strat = min(ref_map, key=lambda k: ref_map[k])
        best_ref = ref_map[best_strat]

        # Select system strategy (largest group achieving <20%)
        selected_strat, selected_gs = select_group_size(ref_map)
        selected_ref = ref_map.get(selected_strat, best_ref)

        # Decision based on selected strategy
        decision = classify(selected_ref)

        # Score stability
        score_std = 0.0
        naive_mm = 0.0
        actual_sp = 0.0
        for row in all_results:
            if (
                row["layer"] == layer_idx
                and row["strategy_name"] == selected_strat
            ):
                score_std = float(row["score_std"])
                naive_mm = float(row["naive_mismatch_rate"])
                actual_sp = float(row["actual_sparsity"])
                break

        score_unstable = naive_mm > 0.10

        # FP16 work contribution
        if decision == "FP16 fallback":
            fp16_work_sum += 1.0
        else:
            fp16_work_sum += selected_ref

        selected_policy.append({
            "layer": layer_idx,
            "decision": decision,
            "selected_system_strategy": selected_strat,
            "selected_group_size": selected_gs,
            "selected_refinement_rate": selected_ref,
            "best_refinement_strategy": best_strat,
            "best_refinement_rate": best_ref,
            "naive_mismatch_rate": naive_mm,
            "score_quantization_unstable": score_unstable,
            "score_std": score_std,
            "actual_sparsity": actual_sp,
        })

    # Compute full-model FP16 proxy
    full_model_fp16_proxy = fp16_work_sum / num_layers
    relative_reduction = 1.0 - full_model_fp16_proxy / stage5_proxy

    # Classify layers
    go_layers = [
        p["layer"]
        for p in selected_policy
        if p["decision"] == "Go"
    ]
    conditional_layers = [
        p["layer"]
        for p in selected_policy
        if p["decision"] == "Conditional Go"
    ]
    fallback_layers = [
        p["layer"]
        for p in selected_policy
        if p["decision"] == "FP16 fallback"
    ]
    score_unstable_layers = [
        p["layer"]
        for p in selected_policy
        if p["score_quantization_unstable"]
    ]

    # Strategy counts
    selected_strategy_counts: dict[str, int] = {}
    selected_gs_counts: dict[int, int] = {}
    for p in selected_policy:
        s = str(p["selected_system_strategy"])
        selected_strategy_counts[s] = selected_strategy_counts.get(s, 0) + 1
        gs = int(p["selected_group_size"])
        selected_gs_counts[gs] = selected_gs_counts.get(gs, 0) + 1

    # Best strategy distribution
    best_strategy_counts: dict[str, int] = {}
    for p in selected_policy:
        s = str(p["best_refinement_strategy"])
        best_strategy_counts[s] = best_strategy_counts.get(s, 0) + 1

    # Save config
    config = {
        "model_name": args.model_name,
        "context_length": args.context_length,
        "block_size": args.block_size,
        "target_sparsity": args.target_sparsity,
        "device": args.device,
        "dtype": args.dtype,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save model metadata
    model_metadata = {
        "model_name": args.model_name,
        "transformers_version": transformers.__version__,
        "num_layers": num_layers,
        "context_length": args.context_length,
        "block_size": args.block_size,
        "threshold_source": "reference_oracle",
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)

    # Save all strategy results
    if all_results:
        fields = list(all_results[0].keys())
        with open(output_dir / "all_strategy_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)

    # Save selected layer policy
    if selected_policy:
        fields = list(selected_policy[0].keys())
        with open(output_dir / "selected_layer_policy.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected_policy)

    # Save strategy summary
    strategy_summary = {
        "total_layers": num_layers,
        "certimask_layers": len(go_layers),
        "fallback_layers": len(fallback_layers),
        "conditional_layers": len(conditional_layers),
        "go_layers": sorted(go_layers),
        "go_layer_indices": sorted(go_layers),
        "conditional_layer_indices": sorted(conditional_layers),
        "fallback_layer_indices": sorted(fallback_layers),
        "score_unstable_layers": sorted(score_unstable_layers),
        "full_model_fp16_proxy": full_model_fp16_proxy,
        "stage5_full_model_proxy": stage5_proxy,
        "relative_reduction_vs_stage5": relative_reduction,
        "selected_strategy_counts": selected_strategy_counts,
        "selected_group_size_counts": {
            str(k): v for k, v in selected_gs_counts.items()
        },
        "best_refinement_strategy_counts": best_strategy_counts,
        "threshold_source": "reference_oracle",
        "note": (
            "full_model_fp16_proxy is NOT a latency estimate. "
            "It approximates the fraction of FP16 scoring work."
        ),
    }
    with open(output_dir / "strategy_summary.json", "w") as f:
        json.dump(strategy_summary, f, indent=2)

    # Save summary
    summary = {
        "all_violations_zero": all(
            r["certificate_violations"] == 0 for r in all_results
        ),
        "all_exact_match": all(
            r["certimask_exact_match_rate"] == 1.0 for r in all_results
        ),
        "strategy_summary": strategy_summary,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print()
    print("=" * 70)
    print("FULL LAYER SCAN SUMMARY")
    print("=" * 70)
    print(f"  Total layers:    {num_layers}")
    print(f"  Go layers:       {len(go_layers)} — {sorted(go_layers)}")
    print(f"  Conditional:     {len(conditional_layers)} — {sorted(conditional_layers)}")
    print(f"  Fallback:        {len(fallback_layers)} — {sorted(fallback_layers)}")
    print(f"  Score unstable:  {len(score_unstable_layers)} — {sorted(score_unstable_layers)}")
    print()
    print("  Selected strategy counts:")
    for s, c in sorted(selected_strategy_counts.items()):
        print(f"    {s}: {c}")
    print()
    print("  Selected group size counts:")
    for gs, c in sorted(selected_gs_counts.items()):
        print(f"    g{gs}: {c}")
    print()
    print(f"  Full-model FP16 proxy:    {full_model_fp16_proxy:.4f}")
    print(f"  Stage 5 proxy:            {stage5_proxy:.4f}")
    print(f"  Relative reduction:       {relative_reduction:.4f}")
    print()
    print("  Per-layer policy:")
    for p in selected_policy:
        layer = p["layer"]
        dec = p["decision"]
        sel = p["selected_system_strategy"]
        sel_ref = p["selected_refinement_rate"]
        best_ref = p["best_refinement_rate"]
        gs = p["selected_group_size"]
        unstable = " [UNSTABLE]" if p["score_quantization_unstable"] else ""
        print(
            f"    L{layer:2d}  {dec:15s}  sel={sel:45s} g{gs}"
            f"  ref={sel_ref:.4f}  best_ref={best_ref:.4f}{unstable}"
        )
    print()
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
