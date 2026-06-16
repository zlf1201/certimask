#!/usr/bin/env python3
"""Phase 5: Full-layer CertiMask certifiability scan.

Scans all decoder layers of Qwen2.5-0.5B-Instruct to determine
which layers can use CertiMask and which need FP16 fallback.
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
    compute_score_bounds,
    validate_score_bounds,
)
from certimask.diagnostics import (
    compute_diagnostic_quantiles,
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
    p = argparse.ArgumentParser(description="CertiMask Phase 5 layer scan")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--target-sparsity", type=float, default=0.85)
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str, default="outputs/phase5_layer_scan")
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


def classify_layer(ref_rate: float) -> str:
    if ref_rate < 0.20:
        return "Go"
    if ref_rate <= 0.30:
        return "Conditional Go"
    return "FP16 fallback"


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

    num_layers = len(model.model.layers)
    print(f"Model: {args.model_name}, layers: {num_layers}")

    input_ids = prepare_text(tokenizer, args.text, args.context_length).to(args.device)
    print(f"Sequence length: {input_ids.shape[1]}")

    from certimask.hf_extraction import extract_qk_from_qwen2

    layer_results = []

    for layer_idx in range(num_layers):
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
            ref_scores, args.target_sparsity,
            valid_mask=valid_mask, per_query=True,
        )

        ref_mask = reference_mask(ref_scores, thresholds, valid_mask=valid_mask)
        naive_mask = naive_quantized_mask(
            q_result.scores, thresholds, valid_mask=valid_mask,
        )

        # Global analytic bounds (for comparison)
        global_bounds = compute_score_bounds(
            q_result.scores, q_result.query_quantized,
            q_result.key_quantized,
            certificate_type="analytic", scale_by_sqrt_dim=True,
        )
        # Coordinate analytic bounds (primary)
        coord_bounds = compute_coordinate_score_bounds(
            q_result.scores, q_result.query_quantized,
            q_result.key_quantized,
            certificate_type="analytic", scale_by_sqrt_dim=True,
        )

        # Validate
        v_global = validate_score_bounds(ref_scores, global_bounds)
        v_coord = validate_score_bounds(ref_scores, coord_bounds)
        if v_global.sum() > 0 or v_coord.sum() > 0:
            print(
                f"  LAYER {layer_idx}: VIOLATIONS!"
                f" global={v_global.sum()} coord={v_coord.sum()}"
            )
            continue

        # CertiMask for exact match verification
        cert_coord = certified_threshold_mask(
            coord_bounds, ref_scores, thresholds, valid_mask=valid_mask,
        )
        mask_m = compute_mask_metrics(
            ref_mask, naive_mask, cert_coord, valid_mask=valid_mask,
        )

        # Diagnostics for global analytic
        diag_global = compute_per_tile_diagnostics(
            ref_scores, q_result.scores, global_bounds, global_bounds,
            thresholds, valid_mask=valid_mask,
        )
        decomp_global = compute_refinement_decomposition(diag_global)

        # Diagnostics for coordinate analytic
        diag_coord = compute_per_tile_diagnostics(
            ref_scores, q_result.scores, coord_bounds, coord_bounds,
            thresholds, valid_mask=valid_mask,
        )
        decomp_coord = compute_refinement_decomposition(diag_coord)

        # Quantiles from coordinate diagnostics
        qntls = compute_diagnostic_quantiles(diag_coord, ref_scores, thresholds)

        decision = classify_layer(decomp_coord.analytic_refinement_rate)

        row = {
            "layer_index": layer_idx,
            "valid_tiles": mask_m.valid_tiles,
            "actual_sparsity": mask_m.actual_sparsity,
            "naive_mismatch_rate": mask_m.naive_mismatch_rate,
            "false_drop_rate": mask_m.false_drop_rate,
            "false_keep_rate": mask_m.false_keep_rate,
            "global_analytic_refinement_rate": decomp_global.analytic_refinement_rate,
            "coordinate_analytic_refinement_rate": decomp_coord.analytic_refinement_rate,
            "certimask_exact_match_rate": mask_m.certimask_match_rate,
            "score_std": qntls.score_std,
            "abs_score_p50": qntls.abs_score_p50,
            "abs_score_p90": qntls.abs_score_p90,
            "abs_score_p99": qntls.abs_score_p99,
            "bound_p50": qntls.analytic_bound_p50,
            "bound_p90": qntls.analytic_bound_p90,
            "bound_p99": qntls.analytic_bound_p99,
            "margin_p50": qntls.margin_p50,
            "margin_p90": qntls.margin_p90,
            "error_to_margin_p90": qntls.ratio_error_to_margin_p90,
            "refinement_reduction_vs_global": (
                decomp_global.analytic_refinement_rate
                - decomp_coord.analytic_refinement_rate
            ),
            "decision": decision,
            "score_quantization_unstable": mask_m.naive_mismatch_rate > 0.10,
        }
        layer_results.append(row)

        print(
            f"  L{layer_idx:2d}  coord_ref={decomp_coord.analytic_refinement_rate:.4f}"
            f"  global_ref={decomp_global.analytic_refinement_rate:.4f}"
            f"  naive_mm={mask_m.naive_mismatch_rate:.4f}"
            f"  sp_std={qntls.score_std:.1f}"
            f"  [{decision}]"
        )

    # Build strategy summary
    go_layers = [r["layer_index"] for r in layer_results if r["decision"] == "Go"]
    cond_layers = [r["layer_index"] for r in layer_results if r["decision"] == "Conditional Go"]
    fb_layers = [r["layer_index"] for r in layer_results if r["decision"] == "FP16 fallback"]
    unstable_layers = [r["layer_index"] for r in layer_results if r["score_quantization_unstable"]]

    all_ref = [r["coordinate_analytic_refinement_rate"] for r in layer_results]
    go_ref = [
        r["coordinate_analytic_refinement_rate"]
        for r in layer_results if r["decision"] == "Go"
    ]
    non_fb = [r for r in layer_results if r["decision"] != "FP16 fallback"]
    non_fb_ref = [r["coordinate_analytic_refinement_rate"] for r in non_fb]

    # estimated_fp16_refinement_fraction: not a latency estimate
    total = len(layer_results)
    fp16_work = sum(
        1.0 if r["decision"] == "FP16 fallback"
        else r["coordinate_analytic_refinement_rate"]
        for r in layer_results
    )
    est_fp16_frac = fp16_work / total if total > 0 else 0.0

    strategy = {
        "total_layers": total,
        "go_layers": go_layers,
        "conditional_layers": cond_layers,
        "fallback_layers": fb_layers,
        "score_quantization_unstable_layers": unstable_layers,
        "go_count": len(go_layers),
        "conditional_count": len(cond_layers),
        "fallback_count": len(fb_layers),
        "mean_refinement_all_layers": sum(all_ref) / len(all_ref) if all_ref else 0,
        "mean_refinement_go_layers": sum(go_ref) / len(go_ref) if go_ref else 0,
        "mean_refinement_non_fallback_layers": (
            sum(non_fb_ref) / len(non_fb_ref) if non_fb_ref else 0
        ),
        "estimated_fp16_refinement_fraction": est_fp16_frac,
        "note_estimated_fp16_fraction": (
            "This is not a latency estimate. It approximates the fraction"
            " of FP16 scoring work assuming each layer's work is proportional"
            " to its refinement rate."
        ),
        "best_certificate": "coordinate_analytic",
        "threshold_source": "reference_oracle",
    }

    # Save outputs
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

    model_md = {
        "model_name": args.model_name,
        "transformers_version": transformers.__version__,
        "num_layers": num_layers,
        "num_query_heads": extracted.num_query_heads,
        "num_key_value_heads": extracted.num_key_value_heads,
        "head_dim": extracted.head_dim,
        "context_length": args.context_length,
        "block_size": args.block_size,
        "num_blocks": num_blocks,
        "threshold_source": "reference_oracle",
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(model_md, f, indent=2)

    if layer_results:
        fields = list(layer_results[0].keys())
        with open(output_dir / "layer_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(layer_results)

    with open(output_dir / "strategy_summary.json", "w") as f:
        json.dump(strategy, f, indent=2)

    summary = {
        "all_violations_zero": all(
            True for _ in layer_results  # already checked above
        ),
        "all_exact_match": all(
            r["certimask_exact_match_rate"] == 1.0 for r in layer_results
        ),
        "strategy": strategy,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print strategy
    print()
    print("=" * 70)
    print("LAYER SCAN STRATEGY")
    print("=" * 70)
    print(f"  Total layers:         {total}")
    print(f"  Go layers:            {len(go_layers)} — {go_layers}")
    print(f"  Conditional layers:   {len(cond_layers)} — {cond_layers}")
    print(f"  Fallback layers:      {len(fb_layers)} — {fb_layers}")
    print(f"  Unstable layers:      {len(unstable_layers)} — {unstable_layers}")
    print()
    print(f"  Mean refinement (all):           {strategy['mean_refinement_all_layers']:.4f}")
    print(f"  Mean refinement (go):            {strategy['mean_refinement_go_layers']:.4f}")
    mean_nf = strategy['mean_refinement_non_fallback_layers']
    print(f"  Mean refinement (non-fallback):  {mean_nf:.4f}")
    print(f"  Est. FP16 work fraction:         {est_fp16_frac:.4f}")
    print()
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
