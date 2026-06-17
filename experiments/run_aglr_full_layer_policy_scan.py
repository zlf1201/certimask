#!/usr/bin/env python3
"""Phase 7E: Full 24-layer AGLR-C v1 policy scan.

Uses Phase 7D optimized search space (block_size=8, logsumexp aggregation)
to establish final layer-wise policy for all 24 layers.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
    p = argparse.ArgumentParser(description="Phase 7E full 24-layer policy scan")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--block-sizes", type=int, nargs="+", default=[8])
    p.add_argument(
        "--target-sparsities", type=float, nargs="+",
        default=[0.30, 0.50, 0.65, 0.70, 0.75],
    )
    p.add_argument("--layers", type=int, nargs="+",
                    default=list(range(24)))
    p.add_argument("--local-blocks", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--aggregations", type=str, nargs="+",
                    default=["logsumexp", "topk_mean"])
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str,
                    default="outputs/phase7e_aglr_full_layer_policy")
    return p.parse_args()


DTYPE_MAP = {
    "float16": torch.float16, "bfloat16": torch.bfloat16,
    "float32": torch.float32, "float64": torch.float64,
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

    strict = kept_mass >= 0.90 and cosine >= 0.95 and l2_rel <= 0.20 and work_frac <= 0.50
    practical = kept_mass >= 0.90 and cosine >= 0.95 and l2_rel <= 0.20 and work_frac <= 0.55
    relaxed = kept_mass >= 0.85 and cosine >= 0.95 and l2_rel <= 0.20 and work_frac <= 0.55
    quality_only = kept_mass >= 0.90 and cosine >= 0.95 and l2_rel <= 0.20

    return {
        "actual_tile_sparsity": metrics.actual_tile_sparsity,
        "attention_tile_work_fraction": work_frac,
        "kept_attention_mass_mean": kept_mass,
        "output_cosine_mean": cosine,
        "output_l2_relative_mean": l2_rel,
        "kept_attention_mass_p50": metrics.kept_attention_mass_p50,
        "kept_attention_mass_p90": metrics.kept_attention_mass_p90,
        "output_l2_relative_p90": metrics.output_l2_relative_p90,
        "output_cosine_p10": metrics.output_cosine_p10,
        "token_mask_sparsity": metrics.token_mask_sparsity,
        "strict_pass": strict,
        "practical_pass": practical,
        "relaxed_pass": relaxed,
        "quality_only_pass": quality_only,
        "oracle_gap_mass": oracle_kept_mass - kept_mass,
        "oracle_gap_l2": l2_rel - oracle_l2,
        "improvement_over_mean_pooled_mass": kept_mass - mp_kept_mass,
        "improvement_over_mean_pooled_l2": mp_l2 - l2_rel,
    }


def select_policy(
    results: list[dict[str, float | int | str | bool]],
) -> dict[str, float | int | str | bool]:
    """Select best policy per layer following priority rules."""
    for pass_type, key in [
        ("strict_pass", "go_strict"),
        ("practical_pass", "go_practical"),
        ("relaxed_pass", "conditional_relaxed"),
        ("quality_only_pass", "quality_only_high_work"),
    ]:
        passing = [r for r in results if r[pass_type]]
        if passing:
            best = min(passing, key=lambda r: r["attention_tile_work_fraction"])
            policy = dict(best)
            policy["decision"] = key
            policy["fallback_reason"] = ""
            return policy

    # Fallback with detailed reason
    best = min(results, key=lambda r: r["output_l2_relative_mean"])
    policy = dict(best)
    policy["decision"] = "fallback_quality"

    # Determine specific fallback reason
    km = float(best["kept_attention_mass_mean"])
    cos = float(best["output_cosine_mean"])
    l2 = float(best["output_l2_relative_mean"])
    wf = float(best["attention_tile_work_fraction"])

    quality_only = km >= 0.90 and cos >= 0.95 and l2 <= 0.20
    if not quality_only:
        failures = []
        if km < 0.90:
            failures.append("mass_failure")
        if cos < 0.95:
            failures.append("cosine_failure")
        if l2 > 0.20:
            failures.append("l2_failure")
        policy["fallback_reason"] = failures[0] if len(failures) == 1 else "multiple_failures"
    elif wf > 0.55:
        policy["fallback_reason"] = "work_failure"
    else:
        policy["fallback_reason"] = "unknown"

    return policy


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
    print(f"Scanning layers: {layers}")

    config = {
        "model_name": args.model_name, "context_length": args.context_length,
        "block_sizes": args.block_sizes, "target_sparsities": args.target_sparsities,
        "layers": layers, "local_blocks": args.local_blocks,
        "aggregations": args.aggregations,
        "device": args.device, "dtype": args.dtype,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_rows: list[dict[str, float | int | str | bool]] = []
    policy_rows: list[dict[str, float | int | str | bool]] = []

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        qkv = extract_qkv_from_qwen2(model, input_ids, layer_index=layer_idx)
        q_full = qkv.query
        k_full = expand_kv_heads(qkv.key, qkv.num_query_heads)
        v_full = expand_kv_heads(qkv.value, qkv.num_query_heads)

        dense_out, dense_probs = dense_attention_output(
            q_full, k_full, v_full, causal=True,
        )

        layer_results: list[dict[str, float | int | str | bool]] = []

        for block_size in args.block_sizes:
            seq_len = q_full.shape[2]
            num_blocks = seq_len // block_size
            if num_blocks == 0:
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

            mp_results: dict[float, dict[str, float]] = {}

            for target_sp in args.target_sparsities:
                thresholds = thresholds_for_target_sparsity(
                    ref_scores, target_sp, valid_mask=valid_scores, per_query=True,
                )
                mp_mask = reference_mask(ref_scores, thresholds, valid_mask=valid_scores)
                oracle_mask = oracle_block_mass_mask(
                    dense_probs, block_size=block_size,
                    target_sparsity=target_sp, valid_block_mask=valid_block_mask,
                )

                oracle_q = compute_quality(
                    dense_out, dense_probs, q_full, k_full, v_full,
                    oracle_mask, block_size, valid_block_mask, layer_idx,
                    target_sp, 1.0, 0.0, 0.0, 0.0,
                )
                oracle_m = oracle_q["kept_attention_mass_mean"]
                oracle_l = oracle_q["output_l2_relative_mean"]

                mp_q = compute_quality(
                    dense_out, dense_probs, q_full, k_full, v_full,
                    mp_mask, block_size, valid_block_mask, layer_idx,
                    target_sp, oracle_m, oracle_l, 0.0, 0.0,
                )
                mp_m = mp_q["kept_attention_mass_mean"]
                mp_l = mp_q["output_l2_relative_mean"]
                mp_results[target_sp] = {"mass": mp_m, "l2": mp_l}

                # Oracle row
                row = {
                    "layer": layer_idx, "block_size": block_size,
                    "target_sparsity": target_sp, "local_blocks": 0,
                    "aggregation": "none", "mask_type": "oracle_block_mass",
                    **oracle_q,
                }
                all_rows.append(row)

                # Mean-pooled row
                row = {
                    "layer": layer_idx, "block_size": block_size,
                    "target_sparsity": target_sp, "local_blocks": 0,
                    "aggregation": "none", "mask_type": "mean_pooled_score",
                    **mp_q,
                }
                all_rows.append(row)

                # AGLR variants
                for aggregation in args.aggregations:
                    ad_scores = compute_antidiagonal_block_scores(
                        q_full, k_full, block_size=block_size,
                        sample_pattern="both_diagonals", aggregation=aggregation,
                        valid_mask=valid_scores,
                    )

                    for local_blks in args.local_blocks:
                        if local_blks > num_blocks:
                            continue

                        aglr_result = aglr_local_plus_landmark_mask(
                            ad_scores, target_sparsity=target_sp,
                            local_blocks=local_blks, valid_mask=valid_scores,
                        )
                        aglr_mask = aglr_result.mask
                        if aglr_mask.shape != valid_block_mask.shape:
                            padded = torch.zeros_like(valid_block_mask)
                            n2, k2 = aglr_mask.shape[2], aglr_mask.shape[3]
                            padded[:, :, :n2, :k2] = aglr_mask
                            aglr_mask = padded

                        q_result = compute_quality(
                            dense_out, dense_probs, q_full, k_full, v_full,
                            aglr_mask, block_size, valid_block_mask, layer_idx,
                            target_sp, oracle_m, oracle_l, mp_m, mp_l,
                        )

                        row = {
                            "layer": layer_idx, "block_size": block_size,
                            "target_sparsity": target_sp,
                            "local_blocks": local_blks,
                            "aggregation": aggregation,
                            "mask_type": "aglr_antidiagonal",
                            **q_result,
                        }
                        all_rows.append(row)
                        layer_results.append(row)

        # Select policy (AGLR configs only)
        aglr_results = [
            r for r in layer_results if r["mask_type"] == "aglr_antidiagonal"
        ]
        policy = select_policy(aglr_results)
        policy["layer"] = layer_idx
        policy_rows.append(policy)

        print(
            f"  Decision: {policy['decision']}"
            f"  mass={policy['kept_attention_mass_mean']:.4f}"
            f"  cosine={policy['output_cosine_mean']:.4f}"
            f"  l2={policy['output_l2_relative_mean']:.4f}"
            f"  work={policy['attention_tile_work_fraction']:.4f}"
        )

    # Save outputs
    def save_csv(rows: list[dict[str, float | int | str | bool]], name: str) -> None:
        if rows:
            all_keys: list[str] = []
            seen: set[str] = set()
            for r in rows:
                for k in r:
                    if k not in seen:
                        all_keys.append(k)
                        seen.add(k)
            with open(output_dir / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

    save_csv(all_rows, "quality_by_layer_config.csv")

    # Policy CSV (column names match Phase 7E spec)
    clean_policy = []
    for p in policy_rows:
        clean_policy.append({
            "layer": p["layer"],
            "decision": p["decision"],
            "selected_block_size": p.get("block_size", ""),
            "selected_target_sparsity": p.get("target_sparsity", ""),
            "selected_local_blocks": p.get("local_blocks", ""),
            "selected_aggregation": p.get("aggregation", ""),
            "selected_work_fraction": p["attention_tile_work_fraction"],
            "selected_kept_mass": p["kept_attention_mass_mean"],
            "selected_cosine": p["output_cosine_mean"],
            "selected_l2": p["output_l2_relative_mean"],
            "strict_pass": p["strict_pass"],
            "practical_pass": p["practical_pass"],
            "relaxed_pass": p["relaxed_pass"],
            "quality_only_pass": p["quality_only_pass"],
            "fallback_reason": p.get("fallback_reason", ""),
        })
    save_csv(clean_policy, "policy_by_layer.csv")

    # Oracle gap CSV
    def _oracle_gap_for(
        lay: int,
    ) -> tuple[float, float, float, float]:
        """Return (oracle_m, oracle_l, mp_m, mp_l) for a given layer."""
        oracle_rows_l = [
            r for r in all_rows
            if r["layer"] == lay and r["mask_type"] == "oracle_block_mass"
        ]
        mp_rows_l = [
            r for r in all_rows
            if r["layer"] == lay and r["mask_type"] == "mean_pooled_score"
        ]
        def _max_field(rows: list[dict], key: str, default: float) -> float:
            return max(float(r[key]) for r in rows) if rows else default

        def _min_field(rows: list[dict], key: str, default: float) -> float:
            return min(float(r[key]) for r in rows) if rows else default

        km_key = "kept_attention_mass_mean"
        l2_key = "output_l2_relative_mean"
        o_m = _max_field(oracle_rows_l, km_key, 1.0)
        o_l = _min_field(oracle_rows_l, l2_key, 0.0)
        m_m = _max_field(mp_rows_l, km_key, 0.0)
        m_l = _max_field(mp_rows_l, l2_key, 1.0)
        return o_m, o_l, m_m, m_l

    oracle_rows: list[dict[str, float | int | str]] = []
    for p in policy_rows:
        lay = p["layer"]
        oracle_m, oracle_l, mp_m, mp_l = _oracle_gap_for(lay)

        oracle_rows.append({
            "layer": lay,
            "oracle_mass": oracle_m,
            "oracle_l2": oracle_l,
            "mean_pooled_mass": mp_m,
            "mean_pooled_l2": mp_l,
            "selected_mass": float(p["kept_attention_mass_mean"]),
            "selected_l2": float(p["output_l2_relative_mean"]),
            "oracle_gap_mass": oracle_m - float(p["kept_attention_mass_mean"]),
            "oracle_gap_l2": float(p["output_l2_relative_mean"]) - oracle_l,
            "improvement_over_mp_mass": float(p["kept_attention_mass_mean"]) - mp_m,
            "improvement_over_mp_l2": mp_l - float(p["output_l2_relative_mean"]),
        })
    save_csv(oracle_rows, "oracle_gap_by_layer.csv")

    # Summary
    go_strict = [p["layer"] for p in policy_rows if p["decision"] == "go_strict"]
    go_practical = [p["layer"] for p in policy_rows if p["decision"] == "go_practical"]
    cond_relaxed = [p["layer"] for p in policy_rows if p["decision"] == "conditional_relaxed"]
    quality_only = [p["layer"] for p in policy_rows if p["decision"] == "quality_only_high_work"]
    fallback = [p["layer"] for p in policy_rows if p["decision"] == "fallback_quality"]

    sel = [p for p in policy_rows]
    mean_wf = sum(float(p["attention_tile_work_fraction"]) for p in sel) / len(sel) if sel else 0
    mean_km = sum(float(p["kept_attention_mass_mean"]) for p in sel) / len(sel) if sel else 0
    mean_cos = sum(float(p["output_cosine_mean"]) for p in sel) / len(sel) if sel else 0
    mean_l2 = sum(float(p["output_l2_relative_mean"]) for p in sel) / len(sel) if sel else 0

    # AGLR vs mean-pooled
    aglr_rows_all = [r for r in all_rows if r["mask_type"] == "aglr_antidiagonal"]
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
        mean_imp_mass = avg_aglr_mass - avg_mp_mass
        mean_imp_l2 = avg_mp_l2 - avg_aglr_l2
    else:
        mean_imp_mass = 0.0
        mean_imp_l2 = 0.0

    # CertiMask readiness
    go_count = len(go_strict) + len(go_practical)
    ready = (
        go_count >= 12
        and len(fallback) <= 3
        and mean_wf <= 0.50
        and mean_km >= 0.90
        and mean_cos >= 0.98
        and mean_l2 <= 0.12
    )
    readiness = (
        "ready_for_topk_certificate_design"
        if ready
        else "needs_indexer_or_policy_improvement"
    )

    bs_counts = dict(Counter(int(p.get("block_size", 0)) for p in policy_rows))
    lb_counts = dict(Counter(int(p.get("local_blocks", 0)) for p in policy_rows))
    sp_counts = dict(Counter(float(p.get("target_sparsity", 0)) for p in policy_rows))
    agg_counts = dict(Counter(str(p.get("aggregation", "")) for p in policy_rows))

    summary = {
        "total_layers": len(layers),
        "go_strict_layers": go_strict,
        "go_practical_layers": go_practical,
        "conditional_relaxed_layers": cond_relaxed,
        "quality_only_high_work_layers": quality_only,
        "fallback_quality_layers": fallback,
        "mean_selected_work_fraction": mean_wf,
        "mean_selected_kept_mass": mean_km,
        "mean_selected_cosine": mean_cos,
        "mean_selected_l2": mean_l2,
        "mean_improvement_over_mean_pooled_mass": mean_imp_mass,
        "mean_improvement_over_mean_pooled_l2": mean_imp_l2,
        "selected_block_size_counts": bs_counts,
        "selected_local_blocks_counts": lb_counts,
        "selected_target_sparsity_counts": sp_counts,
        "selected_aggregation_counts": agg_counts,
        "certimask_readiness": readiness,
    }

    with open(output_dir / "summary_by_decision.json", "w") as f:
        json.dump(summary, f, indent=2)

    # README
    readme_lines = [
        "# Phase 7E: Full 24-Layer AGLR-C v1 Policy Scan",
        "",
        f"**Model:** {args.model_name}",
        f"**Context length:** {args.context_length}",
        f"**Layers scanned:** {len(layers)} ({layers[0]}..{layers[-1]})",
        "",
        "## Decision Distribution",
        "",
        "| Decision | Count | Layers |",
        "|---|---|---|",
        f"| go_strict | {len(go_strict)} | {go_strict} |",
        f"| go_practical | {len(go_practical)} | {go_practical} |",
        f"| conditional_relaxed | {len(cond_relaxed)} | {cond_relaxed} |",
        f"| quality_only_high_work | {len(quality_only)} | {quality_only} |",
        f"| fallback_quality | {len(fallback)} | {fallback} |",
        "",
        "## Aggregate Metrics",
        "",
        f"- Mean selected work fraction: {mean_wf:.4f}",
        f"- Mean selected kept mass: {mean_km:.4f}",
        f"- Mean selected cosine: {mean_cos:.4f}",
        f"- Mean selected L2: {mean_l2:.4f}",
        f"- Mean improvement over mean-pooled (mass): +{mean_imp_mass:.4f}",
        f"- Mean improvement over mean-pooled (L2): -{mean_imp_l2:.4f}",
        "",
        f"## CertiMask Readiness: {readiness}",
        "",
        "## Per-Layer Policy",
        "",
        "| Layer | Decision | BS | Sparsity | LB | Agg | Work | Mass | Cosine | L2 | Fallback |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in clean_policy:
        readme_lines.append(
            f"| {p['layer']} | {p['decision']} | {p['selected_block_size']} "
            f"| {p['selected_target_sparsity']} | {p['selected_local_blocks']} "
            f"| {p['selected_aggregation']} | {p['selected_work_fraction']:.4f} "
            f"| {p['selected_kept_mass']:.4f} | {p['selected_cosine']:.4f} "
            f"| {p['selected_l2']:.4f} | {p['fallback_reason']} |"
        )
    with open(output_dir / "README_results.md", "w") as f:
        f.write("\n".join(readme_lines) + "\n")

    # Print summary
    print()
    print("=" * 70)
    print("FULL 24-LAYER AGLR-C v1 POLICY SCAN")
    print("=" * 70)
    print(f"  Scanned layers: {layers}")
    print(f"  Go strict:          {go_strict}")
    print(f"  Go practical:       {go_practical}")
    print(f"  Conditional:        {cond_relaxed}")
    print(f"  Quality-only:       {quality_only}")
    print(f"  Fallback:           {fallback}")
    print()
    print(f"  Mean work fraction: {mean_wf:.4f}")
    print(f"  Mean kept mass:     {mean_km:.4f}")
    print(f"  Mean cosine:        {mean_cos:.4f}")
    print(f"  Mean L2:            {mean_l2:.4f}")
    print(f"  AGLR vs MP mass:    +{mean_imp_mass:.4f}")
    print(f"  AGLR vs MP L2:      -{mean_imp_l2:.4f}")
    print()
    print(f"  CertiMask readiness: {readiness}")
    print()
    print("  Per-layer policy:")
    for p in clean_policy:
        dec = p["decision"]
        lay = p["layer"]
        km = p["selected_kept_mass"]
        cos = p["selected_cosine"]
        l2 = p["selected_l2"]
        wf = p["selected_work_fraction"]
        reason = p["fallback_reason"]
        print(
            f"    L{lay:2d}  {dec:25s}  mass={km:.4f}  cos={cos:.4f}"
            f"  l2={l2:.4f}  work={wf:.4f}  {reason}"
        )
    print()
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
