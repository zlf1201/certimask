#!/usr/bin/env python3
"""Phase 5.5: Per-group INT8 quantization scan for conditional layers.

Compares per-vector vs per-group quantization on conditional Go layers
to determine if per-group quantization can push more layers below 20%
analytic refinement rate.
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
    p = argparse.ArgumentParser(description="CertiMask Phase 5.5 group quant scan")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--layers", type=int, nargs="+", default=[7, 13, 14, 15, 16, 21])
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--target-sparsity", type=float, default=0.85)
    p.add_argument("--group-sizes", type=int, nargs="+", default=[64, 32, 16, 8, 4])
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str, default="outputs/phase55_group_quant")
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

    all_results = []

    for layer_idx in args.layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        extracted = extract_qk_from_qwen2(
            model, input_ids, layer_index=layer_idx,
        )
        q_full = extracted.query
        k_full = expand_kv_heads(extracted.key, extracted.num_query_heads)

        summaries = mean_pool_qk_blocks(q_full, k_full, block_size=args.block_size)
        num_blocks = summaries.num_blocks

        ref_scores = reference_scores(
            summaries.query, summaries.key, scale_by_sqrt_dim=True,
        )

        valid_mask = make_block_causal_valid_mask(
            num_blocks, num_blocks, device=args.device,
        ).expand_as(ref_scores)

        thresholds = thresholds_for_target_sparsity(
            ref_scores, args.target_sparsity,
            valid_mask=valid_mask, per_query=True,
        )

        ref_mask = reference_mask(ref_scores, thresholds, valid_mask=valid_mask)

        # Baseline: per-vector coordinate analytic (group_size=64 = head_dim)
        v_result = quantized_int8_scores(
            summaries.query, summaries.key, scale_by_sqrt_dim=True,
        )
        from certimask.bounds import compute_coordinate_score_bounds
        v_bounds = compute_coordinate_score_bounds(
            v_result.scores, v_result.query_quantized,
            v_result.key_quantized, certificate_type="analytic",
            scale_by_sqrt_dim=True,
        )
        v_diag = compute_per_tile_diagnostics(
            ref_scores, v_result.scores, v_bounds, v_bounds,
            thresholds, valid_mask=valid_mask,
        )
        v_decomp = compute_refinement_decomposition(v_diag)
        baseline_ref = v_decomp.analytic_refinement_rate
        print(f"  Baseline (per-vector): ref={baseline_ref:.4f}")

        for gs in args.group_sizes:
            g_result = group_quantized_int8_scores(
                summaries.query, summaries.key,
                group_size=gs, scale_by_sqrt_dim=True,
            )
            bounds = compute_group_quantized_coordinate_bounds(
                g_result.scores, g_result.query_quantized,
                g_result.key_quantized, certificate_type="analytic",
                scale_by_sqrt_dim=True,
            )

            violations = validate_score_bounds(ref_scores, bounds)
            cert = certified_threshold_mask(
                bounds, ref_scores, thresholds, valid_mask=valid_mask,
            )
            naive_mask = naive_quantized_mask(
                g_result.scores, thresholds, valid_mask=valid_mask,
            )
            mask_m = compute_mask_metrics(
                ref_mask, naive_mask, cert, valid_mask=valid_mask,
            )

            diag = compute_per_tile_diagnostics(
                ref_scores, g_result.scores, bounds, bounds,
                thresholds, valid_mask=valid_mask,
            )
            decomp = compute_refinement_decomposition(diag)

            # Score error quantiles
            vm = diag.valid_mask
            err_vals = diag.score_error[vm]
            bound_vals = bounds.error_bound[vm]

            rho_vals = bound_vals / (err_vals + torch.finfo(torch.float32).eps)

            row = {
                "layer": layer_idx,
                "group_size": gs,
                "certificate_type": "analytic",
                "actual_sparsity": mask_m.actual_sparsity,
                "naive_mismatch_rate": mask_m.naive_mismatch_rate,
                "false_drop_rate": mask_m.false_drop_rate,
                "false_keep_rate": mask_m.false_keep_rate,
                "certificate_violations": int(violations.sum().item()),
                "certimask_exact_match_rate": mask_m.certimask_match_rate,
                "refinement_rate": decomp.analytic_refinement_rate,
                "rho_p50": float(rho_vals.quantile(0.50).item()),
                "rho_p90": float(rho_vals.quantile(0.90).item()),
                "rho_p99": float(rho_vals.quantile(0.99).item()),
                "rho_max": float(rho_vals.max().item()),
                "bound_p50": float(bound_vals.quantile(0.50).item()),
                "bound_p90": float(bound_vals.quantile(0.90).item()),
                "bound_p99": float(bound_vals.quantile(0.99).item()),
                "score_error_p50": float(err_vals.quantile(0.50).item()),
                "score_error_p90": float(err_vals.quantile(0.90).item()),
                "score_error_p99": float(err_vals.quantile(0.99).item()),
                "refinement_reduction_vs_baseline": baseline_ref - decomp.analytic_refinement_rate,
            }
            all_results.append(row)

            print(
                f"  gs={gs:2d}  ref={decomp.analytic_refinement_rate:.4f}"
                f"  naive_mm={mask_m.naive_mismatch_rate:.4f}"
                f"  viol={int(violations.sum().item())}"
                f"  match={mask_m.certimask_match_rate:.4f}"
                f"  bound_p50={float(bound_vals.quantile(0.50).item()):.4f}"
                f"  err_p50={float(err_vals.quantile(0.50).item()):.4f}"
            )

    # Per-layer best group size
    layer_best: dict[int, dict[str, object]] = {}
    for layer_idx in args.layers:
        layer_rows = [r for r in all_results if r["layer"] == layer_idx]
        if layer_rows:
            best = min(layer_rows, key=lambda r: r["refinement_rate"])
            layer_best[layer_idx] = {
                "best_group_size": best["group_size"],
                "best_refinement_rate": best["refinement_rate"],
                "best_naive_mismatch": best["naive_mismatch_rate"],
                "violations": best["certificate_violations"],
                "exact_match": best["certimask_exact_match_rate"],
            }

    # Classify layers with best group size
    converted_to_go = []
    still_conditional = []
    still_fallback = []
    for layer_idx, info in layer_best.items():
        ref_rate = float(info["best_refinement_rate"])
        if ref_rate < 0.20:
            converted_to_go.append(layer_idx)
        elif ref_rate <= 0.30:
            still_conditional.append(layer_idx)
        else:
            still_fallback.append(layer_idx)

    # Save outputs
    config = {
        "model_name": args.model_name,
        "layers": args.layers,
        "context_length": args.context_length,
        "block_size": args.block_size,
        "target_sparsity": args.target_sparsity,
        "group_sizes": args.group_sizes,
        "device": args.device,
        "dtype": args.dtype,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if all_results:
        fields = list(all_results[0].keys())
        with open(output_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_results)

    summary = {
        "all_violations_zero": all(r["certificate_violations"] == 0 for r in all_results),
        "all_exact_match": all(r["certimask_exact_match_rate"] == 1.0 for r in all_results),
        "layer_best": {str(k): v for k, v in layer_best.items()},
        "converted_to_go": converted_to_go,
        "still_conditional": still_conditional,
        "still_fallback": still_fallback,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print strategy
    print()
    print("=" * 70)
    print("PER-GROUP QUANTIZATION RESULTS")
    print("=" * 70)
    for layer_idx, info in layer_best.items():
        print(
            f"  L{layer_idx:2d}  best_gs={info['best_group_size']:2d}"
            f"  ref={info['best_refinement_rate']:.4f}"
            f"  naive_mm={info['best_naive_mismatch']:.4f}"
        )
    print()
    print(f"  Converted to Go:       {converted_to_go}")
    print(f"  Still Conditional:     {still_conditional}")
    print(f"  Still Fallback:        {still_fallback}")
    print()
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
