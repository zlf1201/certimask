#!/usr/bin/env python3
"""Phase 7C: AGLR-C v1 layer-wise quality policy scan.

Scans multiple layers with best AGLR-C v1 config to establish
layer-wise sparse attention policy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from certimask.aglr_indexer import (
    aglr_local_plus_landmark_mask,
    compute_antidiagonal_block_scores,
)
from certimask.attention_quality import (
    block_sparse_attention_output,
    compute_attention_quality,
    dense_attention_output,
    oracle_block_mass_mask,
)
from certimask.block_summary import expand_kv_heads, mean_pool_qk_blocks
from certimask.hf_extraction import extract_qkv_from_qwen2
from certimask.masking import (
    make_block_causal_valid_mask,
    reference_mask,
    thresholds_for_target_sparsity,
)
from certimask.scoring import reference_scores

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
    p = argparse.ArgumentParser(description="Phase 7C AGLR layer-wise policy scan")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument(
        "--target-sparsities", type=float, nargs="+",
        default=[0.30, 0.40, 0.50, 0.60],
    )
    p.add_argument(
        "--layers", type=int, nargs="+",
        default=[0, 1, 2, 3, 4, 5, 8, 12, 13, 16, 20, 22, 23],
    )
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--local-blocks", type=int, default=2)
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument(
        "--output-dir", type=str,
        default="outputs/phase7c_aglr_layerwise_policy",
    )
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


def compute_quality(
    dense_out: torch.Tensor,
    dense_probs: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
    valid_block_mask: torch.Tensor,
    layer_index: int,
    target_sparsity: float,
    oracle_kept_mass: float,
    oracle_l2: float,
    mp_kept_mass: float,
    mp_l2: float,
) -> dict[str, float | int | bool]:
    sparse_out, sparse_probs = block_sparse_attention_output(
        q, k, v, block_mask, block_size=block_size, causal=True,
    )
    metrics = compute_attention_quality(
        dense_out, dense_probs, sparse_out, sparse_probs,
        block_mask, block_size,
        layer_index=layer_index, target_sparsity=target_sparsity,
        valid_block_mask=valid_block_mask,
    )
    kept_mass = metrics.kept_attention_mass_mean
    cosine = metrics.output_cosine_mean
    l2_rel = metrics.output_l2_relative_mean
    valid_tiles = valid_block_mask.sum().item()
    kept_tiles = (block_mask & valid_block_mask).sum().item()
    work_frac = kept_tiles / valid_tiles if valid_tiles > 0 else 0.0

    strict = (
        kept_mass >= 0.90 and cosine >= 0.95
        and l2_rel <= 0.20 and work_frac <= 0.50
    )
    practical = (
        kept_mass >= 0.90 and cosine >= 0.95
        and l2_rel <= 0.20 and work_frac <= 0.55
    )
    strong = (
        kept_mass >= 0.95 and cosine >= 0.98
        and l2_rel <= 0.10 and work_frac <= 0.55
    )

    return {
        "actual_tile_sparsity": metrics.actual_tile_sparsity,
        "attention_tile_work_fraction": work_frac,
        "kept_attention_mass_mean": kept_mass,
        "kept_attention_mass_p50": metrics.kept_attention_mass_p50,
        "kept_attention_mass_p90": metrics.kept_attention_mass_p90,
        "dropped_attention_mass_mean": metrics.dropped_attention_mass_mean,
        "output_l2_relative_mean": l2_rel,
        "output_l2_relative_p90": metrics.output_l2_relative_p90,
        "output_cosine_mean": cosine,
        "output_cosine_p10": metrics.output_cosine_p10,
        "prob_l1_mean": metrics.prob_l1_mean,
        "prob_kl_mean": metrics.prob_kl_mean,
        "token_mask_sparsity": metrics.token_mask_sparsity,
        "strict_quality_pass": strict,
        "practical_quality_pass": practical,
        "strong_quality_pass": strong,
        "oracle_gap_mass": oracle_kept_mass - kept_mass,
        "oracle_gap_l2": l2_rel - oracle_l2,
        "improvement_over_mean_pooled_mass": kept_mass - mp_kept_mass,
        "improvement_over_mean_pooled_l2": mp_l2 - l2_rel,
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
        args.model_name, torch_dtype=dtype, attn_implementation="eager",
        trust_remote_code=True,
    )
    model.to(args.device).eval()

    input_ids = prepare_text(tokenizer, args.text, args.context_length).to(args.device)
    print(f"Sequence length: {input_ids.shape[1]}")

    layers = args.layers
    if args.all_layers:
        layers = list(range(len(model.model.layers)))
    print(f"Scanning layers: {layers}")

    config = {
        "model_name": args.model_name,
        "context_length": args.context_length,
        "block_size": args.block_size,
        "target_sparsities": args.target_sparsities,
        "layers": layers,
        "local_blocks": args.local_blocks,
        "device": args.device,
        "dtype": args.dtype,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    block_size = args.block_size
    local_blocks = args.local_blocks
    all_rows: list[dict[str, float | int | str | bool]] = []

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        qkv = extract_qkv_from_qwen2(model, input_ids, layer_index=layer_idx)
        q_full = qkv.query
        k_full = expand_kv_heads(qkv.key, qkv.num_query_heads)
        v_full = expand_kv_heads(qkv.value, qkv.num_query_heads)

        dense_out, dense_probs = dense_attention_output(q_full, k_full, v_full, causal=True)

        seq_len = q_full.shape[2]
        num_blocks = seq_len // block_size
        if num_blocks == 0:
            print(f"  Skipping: seq_len {seq_len} < block_size {block_size}")
            continue

        valid_block_mask = make_block_causal_valid_mask(
            num_blocks, num_blocks, device=args.device,
        ).expand(q_full.shape[0], q_full.shape[1], num_blocks, num_blocks)

        summaries = mean_pool_qk_blocks(q_full, k_full, block_size=block_size)
        ref_scores = reference_scores(
            summaries.query, summaries.key, scale_by_sqrt_dim=True,
        )
        nb = summaries.num_blocks
        valid_scores = valid_block_mask[:, :, :nb, :nb]

        # Precompute antidiagonal scores (both_diagonals + logsumexp)
        ad_logsumexp = compute_antidiagonal_block_scores(
            q_full, k_full, block_size=block_size,
            sample_pattern="both_diagonals", aggregation="logsumexp",
            valid_mask=valid_scores,
        )
        ad_topk_mean = compute_antidiagonal_block_scores(
            q_full, k_full, block_size=block_size,
            sample_pattern="both_diagonals", aggregation="topk_mean",
            valid_mask=valid_scores,
        )

        for target_sp in args.target_sparsities:
            # Oracle
            oracle_mask = oracle_block_mass_mask(
                dense_probs, block_size=block_size,
                target_sparsity=target_sp, valid_block_mask=valid_block_mask,
            )
            oracle_q = compute_quality(
                dense_out, dense_probs, q_full, k_full, v_full,
                oracle_mask, block_size, valid_block_mask, layer_idx, target_sp,
                1.0, 0.0, 0.0, 0.0,
            )
            oracle_m = oracle_q["kept_attention_mass_mean"]
            oracle_l = oracle_q["output_l2_relative_mean"]

            # Dense
            dense_mask = valid_block_mask.clone()
            dense_q = compute_quality(
                dense_out, dense_probs, q_full, k_full, v_full,
                dense_mask, block_size, valid_block_mask, layer_idx, 0.0,
                oracle_m, oracle_l, 0.0, 0.0,
            )

            # Mean-pooled score
            thresholds = thresholds_for_target_sparsity(
                ref_scores, target_sp, valid_mask=valid_scores, per_query=True,
            )
            mp_mask = reference_mask(ref_scores, thresholds, valid_mask=valid_scores)
            mp_q = compute_quality(
                dense_out, dense_probs, q_full, k_full, v_full,
                mp_mask, block_size, valid_block_mask, layer_idx, target_sp,
                oracle_m, oracle_l, 0.0, 0.0,
            )
            mp_m = mp_q["kept_attention_mass_mean"]
            mp_l = mp_q["output_l2_relative_mean"]

            # AGLR antidiagonal logsumexp
            aglr_ls_result = aglr_local_plus_landmark_mask(
                ad_logsumexp, target_sparsity=target_sp,
                local_blocks=local_blocks, valid_mask=valid_scores,
            )
            aglr_ls_mask = aglr_ls_result.mask
            if aglr_ls_mask.shape != valid_block_mask.shape:
                padded = torch.zeros_like(valid_block_mask)
                n2, k2 = aglr_ls_mask.shape[2], aglr_ls_mask.shape[3]
                padded[:, :, :n2, :k2] = aglr_ls_mask
                aglr_ls_mask = padded
            aglr_ls_q = compute_quality(
                dense_out, dense_probs, q_full, k_full, v_full,
                aglr_ls_mask, block_size, valid_block_mask, layer_idx, target_sp,
                oracle_m, oracle_l, mp_m, mp_l,
            )

            # AGLR antidiagonal topk_mean
            aglr_tk_result = aglr_local_plus_landmark_mask(
                ad_topk_mean, target_sparsity=target_sp,
                local_blocks=local_blocks, valid_mask=valid_scores,
            )
            aglr_tk_mask = aglr_tk_result.mask
            if aglr_tk_mask.shape != valid_block_mask.shape:
                padded = torch.zeros_like(valid_block_mask)
                n2, k2 = aglr_tk_mask.shape[2], aglr_tk_mask.shape[3]
                padded[:, :, :n2, :k2] = aglr_tk_mask
                aglr_tk_mask = padded
            aglr_tk_q = compute_quality(
                dense_out, dense_probs, q_full, k_full, v_full,
                aglr_tk_mask, block_size, valid_block_mask, layer_idx, target_sp,
                oracle_m, oracle_l, mp_m, mp_l,
            )

            for mask_type, scores_dict in [
                ("dense", dense_q),
                ("mean_pooled_score", mp_q),
                ("oracle_block_mass", oracle_q),
                ("aglr_antidiagonal_logsumexp", aglr_ls_q),
                ("aglr_antidiagonal_topk_mean", aglr_tk_q),
            ]:
                row: dict[str, float | int | str | bool] = {
                    "layer": layer_idx,
                    "target_sparsity": target_sp,
                    "mask_type": mask_type,
                    "sample_pattern": "both_diagonals" if "aglr" in mask_type else "none",
                    "aggregation": (
                        "logsumexp" if "logsumexp" in mask_type
                        else "topk_mean" if "topk" in mask_type
                        else "none"
                    ),
                    **scores_dict,
                }
                all_rows.append(row)

            print(
                f"  sp={target_sp:.2f}:"
                f" mp={mp_m:.4f}"
                f" oracle={oracle_m:.4f}"
                f" aglr_ls={aglr_ls_q['kept_attention_mass_mean']:.4f}"
                f" aglr_tk={aglr_tk_q['kept_attention_mass_mean']:.4f}"
            )

    # Save quality CSV
    if all_rows:
        fields = list(all_rows[0].keys())
        with open(output_dir / "quality_by_layer.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)

    # Build policy per layer
    policy_rows = []
    for layer_idx in layers:
        layer_rows = [r for r in all_rows if r["layer"] == layer_idx]
        if not layer_rows:
            continue

        # Find best practical pass among AGLR configs
        aglr_rows = [
            r for r in layer_rows
            if "aglr" in r["mask_type"] and r["practical_quality_pass"]
        ]

        decision = "fallback"
        selected: dict[str, float | str | int | bool] | None = None
        fallback_reason = ""

        if aglr_rows:
            # Pick lowest work fraction
            best = min(aglr_rows, key=lambda r: r["attention_tile_work_fraction"])
            decision = "go"
            selected = best
        else:
            # Check conditional: kept_mass >= 0.85, l2 <= 0.30, work <= 0.60
            cond_rows = [
                r for r in layer_rows
                if "aglr" in r["mask_type"]
                and r["kept_attention_mass_mean"] >= 0.85
                and r["output_l2_relative_mean"] <= 0.30
                and r["attention_tile_work_fraction"] <= 0.60
            ]
            if cond_rows:
                best = min(cond_rows, key=lambda r: r["output_l2_relative_mean"])
                decision = "conditional"
                selected = best
            else:
                # Find best AGLR for fallback info
                aglr_all = [r for r in layer_rows if "aglr" in r["mask_type"]]
                if aglr_all:
                    best = max(aglr_all, key=lambda r: r["kept_attention_mass_mean"])
                    selected = best
                fallback_reason = "no_practical_or_conditional_pass"

        if selected is not None:
            policy_rows.append({
                "layer": layer_idx,
                "decision": decision,
                "selected_mask_type": selected.get("mask_type", ""),
                "selected_target_sparsity": selected.get("target_sparsity", ""),
                "selected_work_fraction": selected.get("attention_tile_work_fraction", ""),
                "selected_kept_mass": selected.get("kept_attention_mass_mean", ""),
                "selected_cosine": selected.get("output_cosine_mean", ""),
                "selected_l2": selected.get("output_l2_relative_mean", ""),
                "strict_quality_pass": selected.get("strict_quality_pass", False),
                "practical_quality_pass": selected.get("practical_quality_pass", False),
                "strong_quality_pass": selected.get("strong_quality_pass", False),
                "fallback_reason": fallback_reason,
            })

    if policy_rows:
        fields = list(policy_rows[0].keys())
        with open(output_dir / "policy_by_layer.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(policy_rows)

    # Oracle gap CSV
    oracle_gap_rows = [
        r for r in all_rows
        if r["mask_type"] != "dense" and r["layer"] in layers
    ]
    if oracle_gap_rows:
        with open(output_dir / "oracle_gap_by_layer.csv", "w", newline="") as f:
            fields = [
                "layer", "target_sparsity", "mask_type",
                "oracle_gap_mass", "oracle_gap_l2",
                "improvement_over_mean_pooled_mass",
                "improvement_over_mean_pooled_l2",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in oracle_gap_rows:
                w.writerow({k: r[k] for k in fields})

    # Summary
    go_layers = [p["layer"] for p in policy_rows if p["decision"] == "go"]
    cond_layers = [p["layer"] for p in policy_rows if p["decision"] == "conditional"]
    fb_layers = [p["layer"] for p in policy_rows if p["decision"] == "fallback"]

    selected_rows = [p for p in policy_rows if p["selected_work_fraction"] != ""]
    mean_wf = (
        sum(float(p["selected_work_fraction"]) for p in selected_rows)
        / len(selected_rows) if selected_rows else 0.0
    )
    mean_km = (
        sum(float(p["selected_kept_mass"]) for p in selected_rows)
        / len(selected_rows) if selected_rows else 0.0
    )
    mean_l2 = (
        sum(float(p["selected_l2"]) for p in selected_rows)
        / len(selected_rows) if selected_rows else 0.0
    )

    # AGLR vs mean-pooled average improvement
    aglr_rows_all = [r for r in all_rows if "aglr_antidiagonal_logsumexp" in r["mask_type"]]
    mp_rows_all = [r for r in all_rows if r["mask_type"] == "mean_pooled_score"]
    if aglr_rows_all and mp_rows_all:
        avg_aglr_mass = (
            sum(r["kept_attention_mass_mean"] for r in aglr_rows_all)
            / len(aglr_rows_all)
        )
        avg_mp_mass = (
            sum(r["kept_attention_mass_mean"] for r in mp_rows_all)
            / len(mp_rows_all)
        )
        avg_aglr_l2 = (
            sum(r["output_l2_relative_mean"] for r in aglr_rows_all)
            / len(aglr_rows_all)
        )
        avg_mp_l2 = (
            sum(r["output_l2_relative_mean"] for r in mp_rows_all)
            / len(mp_rows_all)
        )
        mass_gain = avg_aglr_mass - avg_mp_mass
        l2_reduction = avg_mp_l2 - avg_aglr_l2
    else:
        mass_gain = 0.0
        l2_reduction = 0.0

    summary = {
        "total_layers": len(model.model.layers),
        "scanned_layers": layers,
        "go_layers": go_layers,
        "conditional_layers": cond_layers,
        "fallback_layers": fb_layers,
        "mean_selected_work_fraction": mean_wf,
        "mean_selected_kept_mass": mean_km,
        "mean_selected_l2": mean_l2,
        "aglr_vs_mean_pooled_average_mass_gain": mass_gain,
        "aglr_vs_mean_pooled_average_l2_reduction": l2_reduction,
        "recommended_next_step": "",
    }

    # Determine recommendation
    if len(go_layers) >= len(layers) * 0.5:
        summary["recommended_next_step"] = (
            "Proceed to anchor-layer reuse or CertiMask top-k certificate"
        )
    elif len(go_layers) + len(cond_layers) >= len(layers) * 0.5:
        summary["recommended_next_step"] = (
            "Test per-group quantization on conditional layers, "
            "or lower quality thresholds"
        )
    else:
        summary["recommended_next_step"] = (
            "Current AGLR-C v1 insufficient; consider larger model "
            "or different block summary approach"
        )

    with open(output_dir / "summary_by_decision.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print()
    print("=" * 70)
    print("LAYER-WISE POLICY SUMMARY")
    print("=" * 70)
    print(f"  Scanned layers: {layers}")
    print(f"  Go:           {go_layers}")
    print(f"  Conditional:  {cond_layers}")
    print(f"  Fallback:     {fb_layers}")
    print()
    print(f"  Mean selected work fraction: {mean_wf:.4f}")
    print(f"  Mean selected kept mass:     {mean_km:.4f}")
    print(f"  Mean selected L2:            {mean_l2:.4f}")
    print(f"  AGLR vs mean-pooled mass gain:   {mass_gain:+.4f}")
    print(f"  AGLR vs mean-pooled L2 reduction: {l2_reduction:+.4f}")
    print()
    print("  Per-layer policy:")
    for p in policy_rows:
        dec = p["decision"]
        lay = p["layer"]
        km = p["selected_kept_mass"]
        wf = p["selected_work_fraction"]
        l2 = p["selected_l2"]
        print(f"    L{lay:2d}  {dec:12s}  mass={km}  l2={l2}  work={wf}")
    print()
    print(f"  Recommended: {summary['recommended_next_step']}")
    print()
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
