#!/usr/bin/env python3
"""Phase 4.5B: Groupwise and coordinate-wise certificate comparison.

Compares different certificate types on real Qwen2.5-0.5B Q/K to determine
whether tighter certificates can reduce refinement rate without changing
the quantizer.
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
    compute_groupwise_score_bounds,
    compute_score_bounds,
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
from certimask.scoring import quantized_int8_scores, reference_scores

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
    parser = argparse.ArgumentParser(description="CertiMask Phase 4.5B")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layer-indices", type=int, nargs="+", default=[0, 13, 23])
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--target-sparsity", type=float, default=0.85)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--output-dir", type=str, default="outputs/phase45b")
    return parser.parse_args()


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def prepare_text(
    tokenizer: object, text: str | None, context_length: int,
) -> torch.Tensor:
    raw_text = text if text is not None else DEFAULT_TEXT
    encoded = tokenizer(raw_text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]
    while token_ids.shape[0] < context_length:
        token_ids = torch.cat([token_ids, token_ids])
    return token_ids[:context_length].unsqueeze(0)


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
        raise ImportError("transformers required: pip install certimask[hf]") from err

    import transformers

    print(f"transformers version: {transformers.__version__}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype,
        attn_implementation="eager", trust_remote_code=True,
    )
    model.to(args.device).eval()

    input_ids = prepare_text(tokenizer, args.text, args.context_length).to(args.device)
    print(f"Sequence length: {input_ids.shape[1]}")

    from certimask.hf_extraction import extract_qk_from_qwen2

    # Certificate configs
    cert_configs: list[dict[str, object]] = [
        {"name": "global_l2_actual", "type": "global", "cert": "actual", "gs": 0},
        {"name": "global_l2_analytic", "type": "global", "cert": "analytic", "gs": 0},
        {"name": "group_l2_actual_g32", "type": "group", "cert": "actual", "gs": 32},
        {"name": "group_l2_analytic_g32", "type": "group", "cert": "analytic", "gs": 32},
        {"name": "group_l2_actual_g16", "type": "group", "cert": "actual", "gs": 16},
        {"name": "group_l2_analytic_g16", "type": "group", "cert": "analytic", "gs": 16},
        {"name": "group_l2_actual_g8", "type": "group", "cert": "actual", "gs": 8},
        {"name": "group_l2_analytic_g8", "type": "group", "cert": "analytic", "gs": 8},
        {"name": "group_l2_actual_g4", "type": "group", "cert": "actual", "gs": 4},
        {"name": "group_l2_analytic_g4", "type": "group", "cert": "analytic", "gs": 4},
        {"name": "coordinate_actual", "type": "coordinate", "cert": "actual", "gs": 0},
        {"name": "coordinate_analytic", "type": "coordinate", "cert": "analytic", "gs": 0},
    ]

    all_results = []
    global_analytic_ref: dict[int, float] = {}

    for layer_idx in args.layer_indices:
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
        q_result = quantized_int8_scores(
            summaries.query, summaries.key, scale_by_sqrt_dim=True,
        )

        valid_mask = make_block_causal_valid_mask(
            num_blocks, num_blocks, device=args.device,
        ).expand_as(ref_scores)

        thresholds = thresholds_for_target_sparsity(
            ref_scores, args.target_sparsity, valid_mask=valid_mask, per_query=True,
        )

        ref_mask = reference_mask(ref_scores, thresholds, valid_mask=valid_mask)
        naive_mask = naive_quantized_mask(
            q_result.scores, thresholds, valid_mask=valid_mask,
        )

        # Compute all certificate bounds
        for cfg in cert_configs:
            name = str(cfg["name"])
            cert_type = str(cfg["cert"])
            ctype = str(cfg["type"])
            gs = int(cfg["gs"])  # type: ignore[arg-type]

            if ctype == "global":
                bounds = compute_score_bounds(
                    q_result.scores, q_result.query_quantized,
                    q_result.key_quantized,
                    certificate_type=cert_type,  # type: ignore[arg-type]
                    scale_by_sqrt_dim=True,
                )
            elif ctype == "group":
                bounds = compute_groupwise_score_bounds(
                    q_result.scores, q_result.query_quantized,
                    q_result.key_quantized, group_size=gs,
                    certificate_type=cert_type,  # type: ignore[arg-type]
                    scale_by_sqrt_dim=True,
                )
            else:
                bounds = compute_coordinate_score_bounds(
                    q_result.scores, q_result.query_quantized,
                    q_result.key_quantized,
                    certificate_type=cert_type,  # type: ignore[arg-type]
                    scale_by_sqrt_dim=True,
                )

            violations = validate_score_bounds(ref_scores, bounds)
            cert_result = certified_threshold_mask(
                bounds, ref_scores, thresholds, valid_mask=valid_mask,
            )
            mask_m = compute_mask_metrics(
                ref_mask, naive_mask, cert_result, valid_mask=valid_mask,
            )

            # Refinement decomposition
            diag = compute_per_tile_diagnostics(
                ref_scores, q_result.scores, bounds, bounds,
                thresholds, valid_mask=valid_mask,
            )
            decomp = compute_refinement_decomposition(diag)

            # Stable ratio stats
            vm = diag.valid_mask
            ref_v = ref_scores[vm]
            # tau_v unused; margin_floor computed from ref_v only
            margin_floor = max(1e-8, 1e-6 * ref_v.abs().median().item())
            error_floor = max(1e-8, 1e-6 * diag.score_error[vm].median().item())

            stable_margin = diag.margin[vm] > margin_floor
            stable_error = diag.score_error[vm] > error_floor
            near_zero_margin_rate = 1.0 - stable_margin.float().mean().item()

            bound_vals = bounds.error_bound[vm]
            inflation = bound_vals / (diag.score_error[vm] + torch.finfo(torch.float32).eps)

            if stable_error.any():
                stab_inf = inflation[stable_error]
                si_p50 = float(stab_inf.quantile(0.50).item())
                si_p90 = float(stab_inf.quantile(0.90).item())
                si_p99 = float(stab_inf.quantile(0.99).item())
            else:
                si_p50 = si_p90 = si_p99 = float("nan")

            row = {
                "layer": layer_idx,
                "certificate_name": name,
                "group_size": gs if ctype == "group" else ("N/A" if ctype == "global" else 1),
                "certificate_type": cert_type,
                "violations": int(violations.sum().item()),
                "exact_match_rate": mask_m.certimask_match_rate,
                "refinement_rate": (
                    decomp.actual_refinement_rate if cert_type == "actual"
                    else decomp.analytic_refinement_rate
                ),
                "bound_p50": float(bound_vals.quantile(0.50).item()),
                "bound_p90": float(bound_vals.quantile(0.90).item()),
                "bound_p99": float(bound_vals.quantile(0.99).item()),
                "stable_inflation_p50": si_p50,
                "stable_inflation_p90": si_p90,
                "stable_inflation_p99": si_p99,
                "near_zero_margin_rate": near_zero_margin_rate,
                "naive_mismatch_rate": decomp.naive_mismatch_rate,
                "oracle_crossing_rate": decomp.oracle_crossing_rate,
                "margin_floor": margin_floor,
                "error_floor": error_floor,
            }
            all_results.append(row)

            # Track global analytic for comparison
            if name == "global_l2_analytic":
                global_analytic_ref[layer_idx] = decomp.analytic_refinement_rate

            print(
                f"  {name:28s} viol={int(violations.sum().item()):3d}"
                f"  match={mask_m.certimask_match_rate:.4f}"
                f"  ref={row['refinement_rate']:.4f}"
                f"  bound_p50={row['bound_p50']:.4f}"
                f"  stab_inf_p50={si_p50:.2f}"
            )

    # Add relative improvements vs global analytic
    for row in all_results:
        layer = int(row["layer"])
        g_ref = global_analytic_ref.get(layer, 0.0)
        if g_ref > 0:
            row["refinement_reduction_vs_global"] = 1.0 - row["refinement_rate"] / g_ref
        else:
            row["refinement_reduction_vs_global"] = 0.0

    # Save
    config = {
        "model_name": args.model_name,
        "layer_indices": args.layer_indices,
        "context_length": args.context_length,
        "block_size": args.block_size,
        "target_sparsity": args.target_sparsity,
        "threshold_source": "reference_oracle",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if all_results:
        fields = list(all_results[0].keys())
        with open(output_dir / "certificate_comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_results)

    summary = {
        "global_analytic_refinement": global_analytic_ref,
        "best_analytic_per_layer": {},
        "all_results": all_results,
    }
    for layer in args.layer_indices:
        layer_rows = [
            r for r in all_results
            if r["layer"] == layer and r["certificate_type"] == "analytic"
        ]
        if layer_rows:
            best = min(layer_rows, key=lambda r: r["refinement_rate"])
            summary["best_analytic_per_layer"][str(layer)] = {
                "name": best["certificate_name"],
                "refinement_rate": best["refinement_rate"],
                "violations": best["violations"],
                "exact_match": best["exact_match_rate"],
            }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
