#!/usr/bin/env python3
"""Real-model threshold experiment for CertiMask Phase 4.

Loads a Qwen2/Qwen2.5 model, extracts post-RoPE Q/K from a specified layer,
applies block pooling, and evaluates CertiMask metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from certimask.block_summary import expand_kv_heads, mean_pool_qk_blocks
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
    parser = argparse.ArgumentParser(
        description="CertiMask real-model threshold experiment"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model name or path",
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--target-sparsities",
        type=float,
        nargs="+",
        default=[0.70, 0.80, 0.85, 0.90],
    )
    parser.add_argument(
        "--certificate-types",
        type=str,
        nargs="+",
        default=["actual", "analytic"],
    )
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--output-dir", type=str, default="outputs/phase4")
    return parser.parse_args()


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def prepare_text(
    tokenizer: object, text: str | None, text_file: str | None, context_length: int
) -> torch.Tensor:
    """Tokenize and prepare input text.

    Args:
        tokenizer: Hugging Face tokenizer.
        text: Direct text input.
        text_file: Path to text file.
        context_length: Target sequence length.

    Returns:
        input_ids tensor of shape [1, L].
    """
    if text_file is not None:
        with open(text_file) as f:
            raw_text = f.read()
    elif text is not None:
        raw_text = text
    else:
        raw_text = DEFAULT_TEXT

    # Repeat text until we have enough tokens
    encoded = tokenizer(raw_text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]

    # Repeat if needed
    while token_ids.shape[0] < context_length:
        token_ids = torch.cat([token_ids, token_ids])

    # Truncate to context_length
    token_ids = token_ids[:context_length]

    return token_ids.unsqueeze(0)  # [1, L]


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
        raise ImportError(
            "transformers is required. Install with: pip install certimask[hf]"
        ) from err

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()

    # Prepare input
    input_ids = prepare_text(tokenizer, args.text, args.text_file, args.context_length)
    input_ids = input_ids.to(args.device)

    actual_seq_len = input_ids.shape[1]
    print(f"Sequence length: {actual_seq_len}")

    # Extract Q/K
    from certimask.hf_extraction import extract_qk_from_qwen2

    print(f"Extracting Q/K from layer {args.layer_index}...")
    extracted = extract_qk_from_qwen2(
        model, input_ids, layer_index=args.layer_index
    )

    print(f"  Q shape: {extracted.query.shape}")
    print(f"  K shape: {extracted.key.shape}")
    print(f"  num_query_heads: {extracted.num_query_heads}")
    print(f"  num_kv_heads: {extracted.num_key_value_heads}")
    print(f"  head_dim: {extracted.head_dim}")

    # GQA expansion
    q = extracted.query
    k = expand_kv_heads(extracted.key, extracted.num_query_heads)
    print(f"  After GQA expansion: Q={q.shape}, K={k.shape}")

    # Block pooling
    summaries = mean_pool_qk_blocks(q, k, block_size=args.block_size)
    print(f"  Blocks: {summaries.num_blocks}, "
          f"used_length: {summaries.used_sequence_length}, "
          f"dropped_tail: {summaries.dropped_tail_tokens}")

    # Scores
    ref_scores = reference_scores(
        summaries.query, summaries.key, scale_by_sqrt_dim=True
    )
    q_result = quantized_int8_scores(
        summaries.query, summaries.key, scale_by_sqrt_dim=True
    )

    # Causal valid mask
    num_blocks = summaries.num_blocks
    valid_mask = make_block_causal_valid_mask(num_blocks, num_blocks, device=args.device)
    valid_mask_expanded = valid_mask.expand_as(ref_scores)

    # Save config
    config = {
        "model_name": args.model_name,
        "layer_index": args.layer_index,
        "context_length": args.context_length,
        "block_size": args.block_size,
        "target_sparsities": args.target_sparsities,
        "certificate_types": args.certificate_types,
        "device": args.device,
        "dtype": args.dtype,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save model metadata
    import transformers

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

    # Run experiments
    results = []
    per_head_results = []

    for cert_type in args.certificate_types:
        bounds = compute_score_bounds(
            q_result.scores,
            q_result.query_quantized,
            q_result.key_quantized,
            certificate_type=cert_type,  # type: ignore[arg-type]
            scale_by_sqrt_dim=True,
        )

        validate_score_bounds(ref_scores, bounds)

        for target_sp in args.target_sparsities:
            thresholds = thresholds_for_target_sparsity(
                ref_scores, target_sp, valid_mask=valid_mask_expanded, per_query=True
            )

            ref_mask = reference_mask(
                ref_scores, thresholds, valid_mask=valid_mask_expanded
            )
            naive_mask = naive_quantized_mask(
                q_result.scores, thresholds, valid_mask=valid_mask_expanded
            )
            cert_result = certified_threshold_mask(
                bounds, ref_scores, thresholds, valid_mask=valid_mask_expanded
            )

            mask_metrics = compute_mask_metrics(
                ref_mask, naive_mask, cert_result, valid_mask=valid_mask_expanded
            )
            bound_metrics = compute_bound_metrics(
                ref_scores,
                q_result.scores,
                bounds,
                thresholds,
                valid_mask=valid_mask_expanded,
            )

            row = {
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
                f"  cert={cert_type:8s} sp={target_sp:.2f} "
                f"actual_sp={mask_metrics.actual_sparsity:.4f} "
                f"naive_mm={mask_metrics.naive_mismatch_count} "
                f"fd={mask_metrics.false_drop_count} "
                f"fk={mask_metrics.false_keep_count} "
                f"match={mask_metrics.certimask_match_rate:.4f} "
                f"refine={mask_metrics.refinement_rate:.4f} "
                f"rho_p90={bound_metrics.rho_p90:.4f} "
                f"rho_max={bound_metrics.rho_max:.4f}"
            )

            # Per-head metrics
            for head_idx in range(extracted.num_query_heads):
                head_ref = ref_scores[:, head_idx : head_idx + 1]
                head_quant = q_result.scores[:, head_idx : head_idx + 1]
                head_bounds = compute_score_bounds(
                    head_quant,
                    _slice_quantized(q_result.query_quantized, head_idx),
                    _slice_quantized(q_result.key_quantized, head_idx),
                    certificate_type=cert_type,  # type: ignore[arg-type]
                    scale_by_sqrt_dim=True,
                )
                head_valid = valid_mask_expanded[:, head_idx : head_idx + 1]
                head_ref_mask = reference_mask(
                    head_ref, thresholds[:, head_idx : head_idx + 1], valid_mask=head_valid
                )
                head_naive = naive_quantized_mask(
                    head_quant, thresholds[:, head_idx : head_idx + 1], valid_mask=head_valid
                )
                head_cert = certified_threshold_mask(
                    head_bounds,
                    head_ref,
                    thresholds[:, head_idx : head_idx + 1],
                    valid_mask=head_valid,
                )
                head_mask_m = compute_mask_metrics(
                    head_ref_mask, head_naive, head_cert, valid_mask=head_valid
                )
                head_bound_m = compute_bound_metrics(
                    head_ref,
                    head_quant,
                    head_bounds,
                    thresholds[:, head_idx : head_idx + 1],
                    valid_mask=head_valid,
                )
                per_head_results.append({
                    "head_index": head_idx,
                    "certificate_type": cert_type,
                    "target_sparsity": target_sp,
                    "valid_tiles": head_mask_m.valid_tiles,
                    "naive_mismatch_rate": head_mask_m.naive_mismatch_rate,
                    "false_drop_rate": head_mask_m.false_drop_rate,
                    "false_keep_rate": head_mask_m.false_keep_rate,
                    "refinement_rate": head_mask_m.refinement_rate,
                    "certimask_match_rate": head_mask_m.certimask_match_rate,
                    "rho_p90": head_bound_m.rho_p90,
                    "rho_max": head_bound_m.rho_max,
                })

    # Save results CSV
    if results:
        with open(output_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    if per_head_results:
        with open(output_dir / "per_head.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_head_results[0].keys())
            writer.writeheader()
            writer.writerows(per_head_results)

    # Save summary
    summary = {
        "total_configs": len(results),
        "all_violations_zero": all(r["certificate_violations"] == 0 for r in results),
        "all_exact_match": all(r["certimask_match_rate"] == 1.0 for r in results),
        "model_metadata": model_metadata,
        "configs": results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"All violations zero: {summary['all_violations_zero']}")
    print(f"All exact match: {summary['all_exact_match']}")


def _slice_quantized(qt: object, head_idx: int) -> torch.Tensor:
    """Slice a QuantizedTensor along the heads dimension."""
    from certimask.quantization import QuantizedTensor

    assert isinstance(qt, QuantizedTensor)
    return QuantizedTensor(
        values=qt.values[:, head_idx : head_idx + 1],
        scale=qt.scale[:, head_idx : head_idx + 1],
        dequantized=qt.dequantized[:, head_idx : head_idx + 1],
        actual_l2_error=qt.actual_l2_error[:, head_idx : head_idx + 1],
        analytic_l2_bound=qt.analytic_l2_bound[:, head_idx : head_idx + 1],
    )


if __name__ == "__main__":
    main()
