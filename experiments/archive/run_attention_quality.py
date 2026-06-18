#!/usr/bin/env python3
"""Phase 6A: Sparse mask attention quality and benefit verification.

Compares sparse attention output vs dense attention output to verify
that the reference sparse mask preserves sufficient attention mass.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from certimask.attention_quality import (
    block_sparse_attention_output,
    compute_attention_quality,
    compute_benefit_proxy,
    dense_attention_output,
)
from certimask.block_summary import expand_kv_heads, mean_pool_qk_blocks
from certimask.bounds import compute_k_only_per_group_bounds, validate_score_bounds
from certimask.hf_extraction import extract_qkv_from_qwen2
from certimask.masking import (
    certified_threshold_mask,
    make_block_causal_valid_mask,
    naive_quantized_mask,
    reference_mask,
    thresholds_for_target_sparsity,
)
from certimask.metrics import compute_mask_metrics
from certimask.scoring import k_only_per_group_scores, reference_scores

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
    p = argparse.ArgumentParser(description="Phase 6A attention quality")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--target-sparsity", type=float, default=0.85)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 1, 8, 12, 13, 22, 23])
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str, default="outputs/phase6a_quality")
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


# Layer policy from Phase 5.6
LAYER_POLICY: dict[int, dict[str, str | int]] = {
    0: {"decision": "FP16 fallback", "strategy": "none", "gs": 0},
    1: {"decision": "Conditional Go", "strategy": "k_only_per_group_g16", "gs": 16},
    2: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    3: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    4: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    5: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    6: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    7: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    8: {"decision": "Go", "strategy": "k_only_per_group_g4", "gs": 4},
    9: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    10: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    11: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    12: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    13: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    14: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    15: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    16: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    17: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    18: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    19: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    20: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    21: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    22: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
    23: {"decision": "Go", "strategy": "k_only_per_group_g16", "gs": 16},
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
        args.model_name, torch_dtype=dtype, attn_implementation="eager", trust_remote_code=True,
    )
    model.to(args.device).eval()

    num_layers = len(model.model.layers)
    layers = list(range(num_layers)) if args.all_layers else args.layers

    input_ids = prepare_text(tokenizer, args.text, args.context_length).to(args.device)
    print(f"Sequence length: {input_ids.shape[1]}")

    config = {
        "model_name": args.model_name, "context_length": args.context_length,
        "block_size": args.block_size, "target_sparsity": args.target_sparsity,
        "layers": layers, "device": args.device, "dtype": args.dtype,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    model_metadata = {
        "model_name": args.model_name, "transformers_version": transformers.__version__,
        "num_layers": num_layers, "context_length": args.context_length,
        "block_size": args.block_size, "threshold_source": "reference_oracle",
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)

    quality_rows: list[dict[str, float | int | str]] = []
    benefit_rows: list[dict[str, float | int | str]] = []

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        policy = LAYER_POLICY.get(layer_idx, LAYER_POLICY[2])
        decision = str(policy["decision"])
        strategy = str(policy["strategy"])
        gs = int(policy["gs"])

        # Extract Q/K/V
        qkv = extract_qkv_from_qwen2(model, input_ids, layer_index=layer_idx)
        q = qkv.query
        k = expand_kv_heads(qkv.key, qkv.num_query_heads)
        v = expand_kv_heads(qkv.value, qkv.num_query_heads)

        # Dense attention reference
        dense_out, dense_probs = dense_attention_output(q, k, v, causal=True)
        print(f"  Dense output shape: {dense_out.shape}")

        # Block summary and scoring for mask generation
        summaries = mean_pool_qk_blocks(q, k, block_size=args.block_size)
        num_blocks = summaries.num_blocks
        ref_scores = reference_scores(summaries.query, summaries.key, scale_by_sqrt_dim=True)
        valid_mask = make_block_causal_valid_mask(
            num_blocks, num_blocks, device=args.device
        ).expand_as(ref_scores)

        thresholds = thresholds_for_target_sparsity(
            ref_scores, args.target_sparsity, valid_mask=valid_mask, per_query=True
        )
        ref_mask = reference_mask(ref_scores, thresholds, valid_mask=valid_mask)

        # CertiMask verification
        if decision == "FP16 fallback":
            # For fallback, use dense attention (block_mask = all valid)
            block_mask = valid_mask.clone()
            refinement_rate = 1.0
            cert_match = 1.0
            cert_violations = 0
        else:
            ko_result = k_only_per_group_scores(
                summaries.query, summaries.key, group_size=gs, scale_by_sqrt_dim=True,
            )
            bounds = compute_k_only_per_group_bounds(
                ko_result.scores, ko_result.query, ko_result.key_quantized,
                certificate_type="analytic", scale_by_sqrt_dim=True,
            )
            violations = validate_score_bounds(ref_scores, bounds)
            cert_violations = int(violations.sum().item())

            cert_result = certified_threshold_mask(
                bounds, ref_scores, thresholds, valid_mask=valid_mask
            )
            naive = naive_quantized_mask(ko_result.scores, thresholds, valid_mask=valid_mask)
            mm = compute_mask_metrics(ref_mask, naive, cert_result, valid_mask=valid_mask)
            cert_match = mm.certimask_match_rate

            # Use CertiMask mask as block_mask
            block_mask = cert_result.mask

            # Compute refinement rate
            from certimask.diagnostics import (
                compute_per_tile_diagnostics,
                compute_refinement_decomposition,
            )
            diag = compute_per_tile_diagnostics(
                ref_scores, ko_result.scores, bounds, bounds, thresholds, valid_mask=valid_mask,
            )
            decomp = compute_refinement_decomposition(diag)
            refinement_rate = decomp.analytic_refinement_rate

        print(f"  Block mask shape: {block_mask.shape}")
        print(f"  CertiMask match: {cert_match:.4f}")
        print(f"  Certificate violations: {cert_violations}")
        print(f"  Refinement rate: {refinement_rate:.4f}")

        # Sparse attention
        sparse_out, sparse_probs = block_sparse_attention_output(
            q, k, v, block_mask, block_size=args.block_size, causal=True,
        )

        # Quality metrics
        quality = compute_attention_quality(
            dense_out, dense_probs, sparse_out, sparse_probs,
            block_mask, args.block_size,
            layer_index=layer_idx, target_sparsity=args.target_sparsity,
        )
        quality_rows.append(vars(quality))

        # Benefit proxy
        benefit = compute_benefit_proxy(
            layer_idx, decision, strategy, block_mask, refinement_rate,
        )
        benefit_rows.append(vars(benefit))

        print(f"  Tile sparsity: {quality.actual_tile_sparsity:.4f}")
        print(f"  Kept mass mean: {quality.kept_attention_mass_mean:.4f}")
        print(f"  Kept mass P90: {quality.kept_attention_mass_p90:.4f}")
        print(f"  Output L2 rel mean: {quality.output_l2_relative_mean:.6f}")
        print(f"  Output cosine mean: {quality.output_cosine_mean:.6f}")
        print(f"  Prob L1 mean: {quality.prob_l1_mean:.6f}")
        print(f"  KL mean: {quality.prob_kl_mean:.6f}")

    # Save CSVs
    if quality_rows:
        with open(output_dir / "quality_by_layer.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(quality_rows[0].keys()))
            w.writeheader()
            w.writerows(quality_rows)

    if benefit_rows:
        with open(output_dir / "benefit_proxy_by_layer.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(benefit_rows[0].keys()))
            w.writeheader()
            w.writerows(benefit_rows)

    # Summary
    n_q = len(quality_rows)
    n_b = len(benefit_rows)
    mean_kept_mass = sum(r["kept_attention_mass_mean"] for r in quality_rows) / n_q
    mean_dropped_mass = sum(r["dropped_attention_mass_mean"] for r in quality_rows) / n_q
    mean_output_l2 = sum(r["output_l2_relative_mean"] for r in quality_rows) / n_q
    mean_output_cosine = sum(r["output_cosine_mean"] for r in quality_rows) / n_q
    mean_tile_sparsity = sum(r["actual_tile_sparsity"] for r in quality_rows) / n_q
    mean_score_work = sum(r["score_work_fraction_proxy"] for r in benefit_rows) / n_b
    mean_attn_work = sum(r["attention_tile_work_fraction"] for r in benefit_rows) / n_b

    fallback_count = sum(1 for r in benefit_rows if r["decision"] == "FP16 fallback")
    certimask_count = len(benefit_rows) - fallback_count

    # Quality check
    quality_ok = (
        mean_kept_mass >= 0.95
        and mean_output_cosine >= 0.98
        and mean_output_l2 <= 0.10
    )
    benefit_ok = mean_attn_work <= 0.20 and mean_score_work <= 0.25

    summary = {
        "mean_kept_attention_mass": mean_kept_mass,
        "mean_dropped_attention_mass": mean_dropped_mass,
        "mean_output_relative_l2": mean_output_l2,
        "mean_output_cosine": mean_output_cosine,
        "mean_tile_sparsity": mean_tile_sparsity,
        "mean_score_work_fraction_proxy": mean_score_work,
        "mean_attention_tile_work_fraction": mean_attn_work,
        "fallback_layers": fallback_count,
        "certimask_layers": certimask_count,
        "quality_acceptable": quality_ok,
        "benefit_acceptable": benefit_ok,
        "recommend_triton": quality_ok and benefit_ok,
        "note": "These are quality and work proxies, not end-to-end latency measurements.",
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # README
    readme = f"""# Phase 6A Results

## Quality Summary
- Mean kept attention mass: {mean_kept_mass:.4f}
- Mean output cosine: {mean_output_cosine:.4f}
- Mean output relative L2: {mean_output_l2:.6f}
- Quality acceptable (mass>=0.95, cosine>=0.98, L2<=0.10): {quality_ok}

## Benefit Summary
- Mean tile sparsity: {mean_tile_sparsity:.4f}
- Mean score work proxy: {mean_score_work:.4f}
- Mean attention tile work: {mean_attn_work:.4f}
- Benefit acceptable (attn<=0.20, score<=0.25): {benefit_ok}

## Recommendation
- Enter Triton: {quality_ok and benefit_ok}

## Note
These are quality and work proxies, not end-to-end latency measurements.
"""
    with open(output_dir / "README_results.md", "w") as f:
        f.write(readme)

    print()
    print("=" * 70)
    print("ATTENTION QUALITY SUMMARY")
    print("=" * 70)
    print(f"  Mean kept attention mass:  {mean_kept_mass:.4f}")
    print(f"  Mean output cosine:        {mean_output_cosine:.4f}")
    print(f"  Mean output relative L2:   {mean_output_l2:.6f}")
    print(f"  Mean tile sparsity:        {mean_tile_sparsity:.4f}")
    print(f"  Mean score work proxy:     {mean_score_work:.4f}")
    print(f"  Mean attn tile work:       {mean_attn_work:.4f}")
    print(f"  Fallback layers:           {fallback_count}")
    print(f"  CertiMask layers:          {certimask_count}")
    print(f"  Quality acceptable:        {quality_ok}")
    print(f"  Benefit acceptable:        {benefit_ok}")
    print(f"  Recommend Triton:          {quality_ok and benefit_ok}")
    print()
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
