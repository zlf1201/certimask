#!/usr/bin/env python3
"""Phase 4.5A diagnostic experiment for CertiMask.

Diagnoses why refinement rate is high on real Qwen2.5-0.5B by decomposing
the contribution of score quantization error vs certificate looseness.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from certimask.block_summary import expand_kv_heads, mean_pool_qk_blocks
from certimask.bounds import compute_score_bounds, validate_score_bounds
from certimask.diagnostics import (
    compute_diagnostic_quantiles,
    compute_per_tile_diagnostics,
    compute_refinement_decomposition,
    compute_row_subset_stats,
)
from certimask.masking import (
    certified_threshold_mask,
    make_block_causal_valid_mask,
    naive_quantized_mask,
    reference_mask,
    thresholds_for_target_sparsity,
)
from certimask.metrics import compute_bound_metrics, compute_mask_metrics
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
    parser = argparse.ArgumentParser(description="CertiMask Phase 4.5A diagnostics")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--target-sparsity", type=float, default=0.85)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--output-dir", type=str, default="outputs/phase4_diag")
    return parser.parse_args()


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def prepare_text(
    tokenizer: object,
    text: str | None,
    text_file: str | None,
    context_length: int,
) -> torch.Tensor:
    if text_file is not None:
        with open(text_file) as f:
            raw_text = f.read()
    elif text is not None:
        raw_text = text
    else:
        raw_text = DEFAULT_TEXT

    encoded = tokenizer(raw_text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]
    while token_ids.shape[0] < context_length:
        token_ids = torch.cat([token_ids, token_ids])
    token_ids = token_ids[:context_length]
    return token_ids.unsqueeze(0)


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
    print(f"Loading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, attn_implementation="eager",
        trust_remote_code=True,
    )
    model.to(args.device).eval()

    input_ids = prepare_text(
        tokenizer, args.text, args.text_file, args.context_length,
    ).to(args.device)
    actual_seq_len = input_ids.shape[1]
    print(f"Sequence length: {actual_seq_len}")

    # Extract Q/K
    from certimask.hf_extraction import extract_qk_from_qwen2

    print(f"Extracting Q/K from layer {args.layer_index}...")
    extracted = extract_qk_from_qwen2(model, input_ids, layer_index=args.layer_index)

    q_full = extracted.query
    k_full = expand_kv_heads(extracted.key, extracted.num_query_heads)

    print(f"  Q: {q_full.shape}, K: {k_full.shape}")
    print(f"  Q finite: {torch.isfinite(q_full).all()}")
    print(f"  K finite: {torch.isfinite(k_full).all()}")

    # Block pooling
    summaries = mean_pool_qk_blocks(q_full, k_full, block_size=args.block_size)
    num_blocks = summaries.num_blocks
    print(
        f"  Blocks: {num_blocks}, used_length: {summaries.used_sequence_length},"
        f" dropped: {summaries.dropped_tail_tokens}"
    )

    # Scores
    ref_scores = reference_scores(summaries.query, summaries.key, scale_by_sqrt_dim=True)
    q_result = quantized_int8_scores(summaries.query, summaries.key, scale_by_sqrt_dim=True)

    # Bounds (both types)
    bounds_actual = compute_score_bounds(
        q_result.scores, q_result.query_quantized, q_result.key_quantized,
        certificate_type="actual", scale_by_sqrt_dim=True,
    )
    bounds_analytic = compute_score_bounds(
        q_result.scores, q_result.query_quantized, q_result.key_quantized,
        certificate_type="analytic", scale_by_sqrt_dim=True,
    )

    # Validate
    v_actual = validate_score_bounds(ref_scores, bounds_actual)
    v_analytic = validate_score_bounds(ref_scores, bounds_analytic)
    print(f"  Actual violations: {v_actual.sum().item()}")
    print(f"  Analytic violations: {v_analytic.sum().item()}")

    # Valid mask (causal)
    valid_mask = make_block_causal_valid_mask(num_blocks, num_blocks, device=args.device)
    valid_mask_expanded = valid_mask.expand_as(ref_scores)

    # Threshold from reference scores
    thresholds = thresholds_for_target_sparsity(
        ref_scores, args.target_sparsity, valid_mask=valid_mask_expanded, per_query=True,
    )

    # Masks
    ref_mask = reference_mask(ref_scores, thresholds, valid_mask=valid_mask_expanded)
    naive_mask = naive_quantized_mask(q_result.scores, thresholds, valid_mask=valid_mask_expanded)

    # Mask metrics
    cert_actual = certified_threshold_mask(
        bounds_actual, ref_scores, thresholds, valid_mask=valid_mask_expanded,
    )
    cert_analytic = certified_threshold_mask(
        bounds_analytic, ref_scores, thresholds, valid_mask=valid_mask_expanded,
    )

    mask_m_actual = compute_mask_metrics(
        ref_mask, naive_mask, cert_actual, valid_mask=valid_mask_expanded,
    )
    mask_m_analytic = compute_mask_metrics(
        ref_mask, naive_mask, cert_analytic, valid_mask=valid_mask_expanded,
    )

    # Per-tile diagnostics
    diag = compute_per_tile_diagnostics(
        ref_scores, q_result.scores, bounds_actual, bounds_analytic,
        thresholds, valid_mask=valid_mask_expanded,
    )

    # Refinement decomposition
    decomp = compute_refinement_decomposition(diag)

    # Diagnostic quantiles
    qntls = compute_diagnostic_quantiles(diag, ref_scores, thresholds)

    # Row subset stats
    subsets = []
    for min_keys in [1, 4, 8, 16]:
        label = f">={min_keys} valid keys"
        subsets.append(
            compute_row_subset_stats(diag, ref_mask, min_valid_keys=min_keys, label=label)
        )

    # Save config
    config = {
        "model_name": args.model_name,
        "layer_index": args.layer_index,
        "context_length": args.context_length,
        "block_size": args.block_size,
        "target_sparsity": args.target_sparsity,
        "device": args.device,
        "dtype": args.dtype,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Model metadata
    model_metadata = {
        "model_name": extracted.model_name,
        "transformers_version": transformers.__version__,
        "layer_index": extracted.layer_index,
        "num_query_heads": extracted.num_query_heads,
        "num_key_value_heads": extracted.num_key_value_heads,
        "head_dim": extracted.head_dim,
        "sequence_length": extracted.sequence_length,
        "block_size": args.block_size,
        "num_blocks": num_blocks,
        "device": args.device,
        "dtype": args.dtype,
        "threshold_source": "reference_oracle",
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)

    # Summary
    summary = {
        "model_metadata": model_metadata,
        "actual_sparsity": mask_m_actual.actual_sparsity,
        "naive_mismatch_rate": mask_m_actual.naive_mismatch_rate,
        "naive_mismatch_count": mask_m_actual.naive_mismatch_count,
        "valid_tiles": mask_m_actual.valid_tiles,
        "refinement_decomposition": {
            "naive_mismatch_rate": decomp.naive_mismatch_rate,
            "oracle_crossing_rate": decomp.oracle_crossing_rate,
            "actual_refinement_rate": decomp.actual_refinement_rate,
            "analytic_refinement_rate": decomp.analytic_refinement_rate,
        },
        "certimask_match_rate_actual": mask_m_actual.certimask_match_rate,
        "certimask_match_rate_analytic": mask_m_analytic.certimask_match_rate,
        "score_distribution": {
            "std": qntls.score_std,
            "mean": qntls.score_mean,
            "p01": qntls.score_p01,
            "p10": qntls.score_p10,
            "p50": qntls.score_p50,
            "p90": qntls.score_p90,
            "p99": qntls.score_p99,
            "abs_p50": qntls.abs_score_p50,
            "abs_p90": qntls.abs_score_p90,
            "abs_p99": qntls.abs_score_p99,
        },
        "threshold_distribution": {
            "mean": qntls.threshold_mean,
            "std": qntls.threshold_std,
            "p10": qntls.threshold_p10,
            "p50": qntls.threshold_p50,
            "p90": qntls.threshold_p90,
        },
        "error_quantiles": {
            "p50": qntls.abs_error_p50,
            "p90": qntls.abs_error_p90,
            "p99": qntls.abs_error_p99,
        },
        "actual_bound_quantiles": {
            "p50": qntls.actual_bound_p50,
            "p90": qntls.actual_bound_p90,
            "p99": qntls.actual_bound_p99,
        },
        "analytic_bound_quantiles": {
            "p50": qntls.analytic_bound_p50,
            "p90": qntls.analytic_bound_p90,
            "p99": qntls.analytic_bound_p99,
        },
        "margin_quantiles": {
            "p10": qntls.margin_p10,
            "p50": qntls.margin_p50,
            "p90": qntls.margin_p90,
        },
        "bound_inflation_actual": {
            "p50": qntls.ratio_actual_inflation_p50,
            "p90": qntls.ratio_actual_inflation_p90,
            "p99": qntls.ratio_actual_inflation_p99,
        },
        "bound_inflation_analytic": {
            "p50": qntls.ratio_analytic_inflation_p50,
            "p90": qntls.ratio_analytic_inflation_p90,
            "p99": qntls.ratio_analytic_inflation_p99,
        },
        "error_to_margin": {
            "p50": qntls.ratio_error_to_margin_p50,
            "p90": qntls.ratio_error_to_margin_p90,
            "p99": qntls.ratio_error_to_margin_p99,
        },
        "margin_to_actual_bound": {
            "p50": qntls.ratio_margin_to_actual_p50,
            "p90": qntls.ratio_margin_to_actual_p90,
            "p99": qntls.ratio_margin_to_actual_p99,
        },
        "margin_to_analytic_bound": {
            "p50": qntls.ratio_margin_to_analytic_p50,
            "p90": qntls.ratio_margin_to_analytic_p90,
            "p99": qntls.ratio_margin_to_analytic_p99,
        },
        "row_subsets": [
            {
                "label": s.label,
                "num_rows": s.num_rows,
                "naive_mismatch_rate": s.naive_mismatch_rate,
                "oracle_crossing_rate": s.oracle_crossing_rate,
                "actual_refinement_rate": s.actual_refinement_rate,
                "analytic_refinement_rate": s.analytic_refinement_rate,
                "actual_sparsity": s.actual_sparsity,
            }
            for s in subsets
        ],
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # per_head.csv
    per_head_rows = []
    for h in range(extracted.num_query_heads):
        for cert_type in ("actual", "analytic"):
            h_ref = ref_scores[:, h:h+1]
            h_quant = q_result.scores[:, h:h+1]
            h_bounds = compute_score_bounds(
                h_quant,
                _slice_qt(q_result.query_quantized, h),
                _slice_qt(q_result.key_quantized, h),
                certificate_type=cert_type,
                scale_by_sqrt_dim=True,
            )
            h_valid = valid_mask_expanded[:, h:h+1]
            h_thr = thresholds[:, h:h+1]
            h_ref_mask = reference_mask(h_ref, h_thr, valid_mask=h_valid)
            h_naive = naive_quantized_mask(h_quant, h_thr, valid_mask=h_valid)
            h_cert = certified_threshold_mask(
                h_bounds, h_ref, h_thr, valid_mask=h_valid,
            )
            h_mm = compute_mask_metrics(
                h_ref_mask, h_naive, h_cert, valid_mask=h_valid,
            )
            h_diag = compute_per_tile_diagnostics(
                h_ref, h_quant, h_bounds, h_bounds, h_thr, valid_mask=h_valid,
            )
            h_decomp = compute_refinement_decomposition(h_diag)
            h_bm = compute_bound_metrics(
                h_ref, h_quant, h_bounds, h_thr, valid_mask=h_valid,
            )
            per_head_rows.append({
                "head_index": h, "certificate_type": cert_type,
                "target_sparsity": args.target_sparsity,
                "valid_tiles": h_mm.valid_tiles,
                "naive_mismatch_rate": h_mm.naive_mismatch_rate,
                "false_drop_rate": h_mm.false_drop_rate,
                "false_keep_rate": h_mm.false_keep_rate,
                "oracle_crossing_rate": h_decomp.oracle_crossing_rate,
                "actual_refinement_rate": h_decomp.actual_refinement_rate,
                "analytic_refinement_rate": h_decomp.analytic_refinement_rate,
                "certimask_match_rate": h_mm.certimask_match_rate,
                "rho_p90": h_bm.rho_p90, "rho_max": h_bm.rho_max,
            })

    if per_head_rows:
        with open(output_dir / "per_head.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_head_rows[0].keys())
            w.writeheader()
            w.writerows(per_head_rows)

    # per_query_row.csv
    vm = diag.valid_mask
    b_idx, h_idx, q_idx = torch.where(diag.valid_key_blocks_per_query > 0)
    row_data = []
    for i in range(len(b_idx)):
        bi, hi, qi = b_idx[i].item(), h_idx[i].item(), q_idx[i].item()
        n_keys = diag.valid_key_blocks_per_query[bi, hi, qi].item()
        row_slice = vm[bi:bi+1, hi:hi+1, qi:qi+1, :]
        flip_row = diag.flip_mask[bi:bi+1, hi:hi+1, qi:qi+1, :]
        cross_row = diag.oracle_crossing[bi:bi+1, hi:hi+1, qi:qi+1, :]
        margin_row = diag.margin[bi:bi+1, hi:hi+1, qi:qi+1, :]
        actual_row = diag.actual_bound[bi:bi+1, hi:hi+1, qi:qi+1, :]
        analytic_row = diag.analytic_bound[bi:bi+1, hi:hi+1, qi:qi+1, :]
        error_row = diag.score_error[bi:bi+1, hi:hi+1, qi:qi+1, :]
        n = int(row_slice.sum().item())
        if n == 0:
            continue
        row_data.append({
            "batch": bi, "head": hi, "query_block": qi,
            "valid_key_blocks": n_keys,
            "naive_mismatch": int((flip_row & row_slice).sum().item()) / n,
            "oracle_crossing": int((cross_row & row_slice).sum().item()) / n,
            "actual_refinement": int((margin_row <= actual_row).sum().item()) / n,
            "analytic_refinement": int((margin_row <= analytic_row).sum().item()) / n,
            "mean_error": float(error_row[row_slice].mean().item()),
            "mean_margin": float(margin_row[row_slice].mean().item()),
            "mean_actual_bound": float(actual_row[row_slice].mean().item()),
            "mean_analytic_bound": float(analytic_row[row_slice].mean().item()),
        })

    if row_data:
        with open(output_dir / "per_query_row.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row_data[0].keys())
            w.writeheader()
            w.writerows(row_data)

    # diagnostic_quantiles.csv
    qntls_dict = {k: v for k, v in vars(qntls).items()}
    with open(output_dir / "diagnostic_quantiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        for k, v in qntls_dict.items():
            w.writerow({"metric": k, "value": v})

    # Print summary
    print()
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  CertiMask exact match (actual):   {mask_m_actual.certimask_match_rate:.4f}")
    print(f"  CertiMask exact match (analytic): {mask_m_analytic.certimask_match_rate:.4f}")
    print(f"  Actual sparsity:                  {mask_m_actual.actual_sparsity:.4f}")
    print()
    print("--- Refinement Decomposition ---")
    print(f"  Naive mismatch rate:     {decomp.naive_mismatch_rate:.4f}")
    print(f"  Oracle crossing rate:    {decomp.oracle_crossing_rate:.4f}")
    print(f"  Actual-L2 refinement:    {decomp.actual_refinement_rate:.4f}")
    print(f"  Analytic refinement:     {decomp.analytic_refinement_rate:.4f}")
    print()
    print("--- Score Distribution ---")
    print(f"  std={qntls.score_std:.4f}  mean={qntls.score_mean:.4f}")
    print(
        f"  p01={qntls.score_p01:.4f}  p10={qntls.score_p10:.4f}"
        f"  p50={qntls.score_p50:.4f}  p90={qntls.score_p90:.4f}"
        f"  p99={qntls.score_p99:.4f}"
    )
    print(
        f"  |score| p50={qntls.abs_score_p50:.4f}"
        f"  p90={qntls.abs_score_p90:.4f}  p99={qntls.abs_score_p99:.4f}"
    )
    print()
    print("--- Error vs Margin ---")
    print(
        f"  |s - s_tilde| p50={qntls.abs_error_p50:.6f}"
        f"  p90={qntls.abs_error_p90:.6f}  p99={qntls.abs_error_p99:.6f}"
    )
    print(
        f"  |s - tau|     p10={qntls.margin_p10:.6f}"
        f"  p50={qntls.margin_p50:.6f}  p90={qntls.margin_p90:.6f}"
    )
    print(
        f"  error/margin  p50={qntls.ratio_error_to_margin_p50:.4f}"
        f"  p90={qntls.ratio_error_to_margin_p90:.4f}"
        f"  p99={qntls.ratio_error_to_margin_p99:.4f}"
    )
    print()
    print("--- Certificate Inflation (bound / error) ---")
    print(
        f"  Actual   p50={qntls.ratio_actual_inflation_p50:.4f}"
        f"  p90={qntls.ratio_actual_inflation_p90:.4f}"
        f"  p99={qntls.ratio_actual_inflation_p99:.4f}"
    )
    print(
        f"  Analytic p50={qntls.ratio_analytic_inflation_p50:.4f}"
        f"  p90={qntls.ratio_analytic_inflation_p90:.4f}"
        f"  p99={qntls.ratio_analytic_inflation_p99:.4f}"
    )
    print()
    print("--- Margin / Bound ---")
    print(
        f"  margin/actual   p50={qntls.ratio_margin_to_actual_p50:.4f}"
        f"  p90={qntls.ratio_margin_to_actual_p90:.4f}"
        f"  p99={qntls.ratio_margin_to_actual_p99:.4f}"
    )
    print(
        f"  margin/analytic p50={qntls.ratio_margin_to_analytic_p50:.4f}"
        f"  p90={qntls.ratio_margin_to_analytic_p90:.4f}"
        f"  p99={qntls.ratio_margin_to_analytic_p99:.4f}"
    )
    print()
    print("--- Row Subsets ---")
    for s in subsets:
        print(
            f"  {s.label:20s} rows={s.num_rows:4d}"
            f"  naive_mm={s.naive_mismatch_rate:.4f}"
            f"  oracle_cross={s.oracle_crossing_rate:.4f}"
            f"  actual_ref={s.actual_refinement_rate:.4f}"
            f"  analytic_ref={s.analytic_refinement_rate:.4f}"
            f"  sparsity={s.actual_sparsity:.4f}"
        )

    print()
    print(f"Results saved to {output_dir}")


def _slice_qt(qt: object, h: int) -> torch.Tensor:
    from certimask.quantization import QuantizedTensor
    assert isinstance(qt, QuantizedTensor)
    return QuantizedTensor(
        values=qt.values[:, h:h+1], scale=qt.scale[:, h:h+1],
        dequantized=qt.dequantized[:, h:h+1],
        actual_l2_error=qt.actual_l2_error[:, h:h+1],
        analytic_l2_bound=qt.analytic_l2_bound[:, h:h+1],
    )


if __name__ == "__main__":
    main()
