#!/usr/bin/env python3
"""Phase 7D: AGLR-C v1 quality-work frontier scan.

Scans sparsity, block_size, local_blocks, and aggregation to build
per-layer Pareto frontiers and establish final layer-wise policy.
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
    p = argparse.ArgumentParser(description="Phase 7D quality-work frontier")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--block-sizes", type=int, nargs="+", default=[8, 16])
    p.add_argument(
        "--target-sparsities", type=float, nargs="+",
        default=[0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80],
    )
    p.add_argument("--layers", type=int, nargs="+",
                    default=[0, 1, 2, 4, 5, 8, 12, 13, 16, 20, 22, 23])
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--local-blocks", type=int, nargs="+", default=[0, 1, 2, 4])
    p.add_argument("--aggregations", type=str, nargs="+",
                    default=["logsumexp", "topk_mean"])
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str,
                    default="outputs/phase7d_aglr_quality_work_frontier")
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


def compute_pareto_frontier(
    results: list[dict[str, float | int | str | bool]],
) -> list[dict[str, float | int | str | bool]]:
    """Compute Pareto frontier: minimize work_fraction, maximize kept_mass, minimize l2.

    A dominates B if A.work <= B.work AND A.mass >= B.mass AND A.l2 <= B.l2
    with at least one strict inequality.
    """
    def _dominates(a: dict[str, float | int | str | bool],
                  b: dict[str, float | int | str | bool]) -> bool:
        """Check if a dominates b."""
        aw = a["attention_tile_work_fraction"]
        bw = b["attention_tile_work_fraction"]
        am = a["kept_attention_mass_mean"]
        bm = b["kept_attention_mass_mean"]
        al = a["output_l2_relative_mean"]
        bl = b["output_l2_relative_mean"]
        return (aw <= bw and am >= bm and al <= bl
                and (aw < bw or am > bm or al < bl))

    frontier: list[dict[str, float | int | str | bool]] = []
    for candidate in results:
        dominated = False
        to_remove: list[int] = []
        for i, existing in enumerate(frontier):
            if _dominates(existing, candidate):
                dominated = True
                break
            if _dominates(candidate, existing):
                to_remove.append(i)
        if not dominated:
            for i in reversed(to_remove):
                frontier.pop(i)
            frontier.append(candidate)
    return frontier


def select_policy(
    results: list[dict[str, float | int | str | bool]],
) -> dict[str, float | int | str | bool]:
    """Select best policy per layer following priority rules."""
    # Priority: strict > practical > relaxed > quality_only > fallback
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

    # Fallback
    best = min(results, key=lambda r: r["output_l2_relative_mean"])
    policy = dict(best)
    policy["decision"] = "fallback_quality"

    # Determine fallback reason
    has_quality = any(r["quality_only_pass"] for r in results)
    has_work = any(r["attention_tile_work_fraction"] <= 0.55 for r in results)
    if not has_quality and not has_work:
        policy["fallback_reason"] = "both_quality_and_work_failure"
    elif not has_quality:
        policy["fallback_reason"] = "quality_failure"
    else:
        policy["fallback_reason"] = "work_failure"
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
    if args.all_layers:
        layers = list(range(len(model.model.layers)))
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
    pareto_rows: list[dict[str, float | int | str | bool]] = []

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

            # Mean-pooled baseline per block_size
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
                layer_results.append(row)

                # Mean-pooled row
                row = {
                    "layer": layer_idx, "block_size": block_size,
                    "target_sparsity": target_sp, "local_blocks": 0,
                    "aggregation": "none", "mask_type": "mean_pooled_score",
                    **mp_q,
                }
                all_rows.append(row)
                layer_results.append(row)

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

        # Compute Pareto frontier for this layer (AGLR configs only)
        aglr_results = [
            r for r in layer_results if r["mask_type"] == "aglr_antidiagonal"
        ]
        frontier = compute_pareto_frontier(aglr_results)
        for rank, entry in enumerate(frontier):
            entry["frontier_rank"] = rank
            pareto_rows.append(entry)

        # Select policy (AGLR configs only; baselines are for comparison only)
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
            with open(output_dir / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    save_csv(all_rows, "quality_work_results.csv")
    save_csv(pareto_rows, "pareto_frontier_by_layer.csv")

    # Policy CSV with selected fields
    clean_policy = []
    for p in policy_rows:
        clean_policy.append({
            "layer": p["layer"],
            "decision": p["decision"],
            "selected_block_size": p.get("block_size", ""),
            "selected_target_sparsity": p.get("target_sparsity", ""),
            "selected_local_blocks": p.get("local_blocks", ""),
            "selected_aggregation": p.get("aggregation", ""),
            "attention_tile_work_fraction": p["attention_tile_work_fraction"],
            "kept_attention_mass_mean": p["kept_attention_mass_mean"],
            "output_cosine_mean": p["output_cosine_mean"],
            "output_l2_relative_mean": p["output_l2_relative_mean"],
            "fallback_reason": p.get("fallback_reason", ""),
        })
    save_csv(clean_policy, "policy_by_layer.csv")

    # Summary
    go_strict = [p["layer"] for p in policy_rows if p["decision"] == "go_strict"]
    go_practical = [p["layer"] for p in policy_rows if p["decision"] == "go_practical"]
    cond_relaxed = [p["layer"] for p in policy_rows if p["decision"] == "conditional_relaxed"]
    quality_only = [p["layer"] for p in policy_rows if p["decision"] == "quality_only_high_work"]
    fallback = [p["layer"] for p in policy_rows if p["decision"] == "fallback_quality"]

    sel = [p for p in policy_rows if p["attention_tile_work_fraction"] > 0]
    mean_wf = sum(float(p["attention_tile_work_fraction"]) for p in sel) / len(sel) if sel else 0
    mean_km = sum(float(p["kept_attention_mass_mean"]) for p in sel) / len(sel) if sel else 0
    mean_cos = sum(float(p["output_cosine_mean"]) for p in sel) / len(sel) if sel else 0
    mean_l2 = sum(float(p["output_l2_relative_mean"]) for p in sel) / len(sel) if sel else 0

    # Phase 7C comparison
    phase7c_go = {3, 8, 12, 13, 20}
    phase7c_cond = {0, 1, 2, 4, 5}

    current_go = set(go_strict) | set(go_practical)
    current_cond = set(cond_relaxed) | set(quality_only)

    improved = []
    degraded = []
    for lay in layers:
        was_go = lay in phase7c_go
        was_cond = lay in phase7c_cond
        is_go = lay in current_go
        is_cond = lay in current_cond

        rank_old = 0 if was_go else (1 if was_cond else 2)
        rank_new = 0 if is_go else (1 if is_cond else 2)
        if rank_new < rank_old:
            improved.append(lay)
        elif rank_new > rank_old:
            degraded.append(lay)

    summary = {
        "scanned_layers": layers,
        "go_strict_layers": go_strict,
        "go_practical_layers": go_practical,
        "conditional_relaxed_layers": cond_relaxed,
        "quality_only_high_work_layers": quality_only,
        "fallback_quality_layers": fallback,
        "mean_selected_work_fraction": mean_wf,
        "mean_selected_kept_mass": mean_km,
        "mean_selected_cosine": mean_cos,
        "mean_selected_l2": mean_l2,
        "layers_improved_from_phase7c": improved,
        "layers_degraded_from_phase7c": degraded,
        "recommended_next_step": "",
    }

    if len(go_strict) + len(go_practical) >= len(layers) * 0.6:
        summary["recommended_next_step"] = "Proceed to CertiMask top-k certificate"
    elif len(go_strict) + len(go_practical) + len(cond_relaxed) >= len(layers) * 0.6:
        summary["recommended_next_step"] = "Lower quality thresholds or test larger model"
    else:
        summary["recommended_next_step"] = (
            "Current indexer insufficient; consider alternative approaches"
        )

    with open(output_dir / "summary_by_decision.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print()
    print("=" * 70)
    print("QUALITY-WORK FRONTIER SUMMARY")
    print("=" * 70)
    print(f"  Go strict:       {go_strict}")
    print(f"  Go practical:    {go_practical}")
    print(f"  Conditional:     {cond_relaxed}")
    print(f"  Quality-only:    {quality_only}")
    print(f"  Fallback:        {fallback}")
    print()
    print(f"  Mean work fraction: {mean_wf:.4f}")
    print(f"  Mean kept mass:     {mean_km:.4f}")
    print(f"  Mean cosine:        {mean_cos:.4f}")
    print(f"  Mean L2:            {mean_l2:.4f}")
    print()
    print(f"  Improved from Phase 7C: {improved}")
    print(f"  Degraded from Phase 7C: {degraded}")
    print()
    print("  Per-layer policy:")
    for p in clean_policy:
        dec = p["decision"]
        lay = p["layer"]
        km = p["kept_attention_mass_mean"]
        cos = p["output_cosine_mean"]
        l2 = p["output_l2_relative_mean"]
        wf = p["attention_tile_work_fraction"]
        reason = p["fallback_reason"]
        print(
            f"    L{lay:2d}  {dec:25s}  mass={km:.4f}  cos={cos:.4f}"
            f"  l2={l2:.4f}  work={wf:.4f}  {reason}"
        )
    print()
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
