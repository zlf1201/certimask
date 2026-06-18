#!/usr/bin/env python3
"""Phase 8B: Group-size scan and boundary fallback optimization.

Scans K-only per-group quantization group sizes, implements tighter
partition-aware ambiguity mode, diagnoses Layer 8 anomaly, and runs
full 24-layer CertiMask scan with best configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from certimask.aglr_certimask import (
    aglr_certimask_topk,
    compute_aglr_certimask_metrics,
)
from certimask.block_summary import expand_kv_heads
from certimask.hf_extraction import extract_qkv_from_qwen2
from certimask.masking import make_block_causal_valid_mask
from certimask.quantization import quantize_int8_per_group

# Phase 7E per-layer policy
LAYER_POLICY: dict[int, dict[str, float | int | str]] = {
    0: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp",
        "decision": "conditional_relaxed"},
    1: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp",
        "decision": "conditional_relaxed"},
    2: {"target_sparsity": 0.50, "local_blocks": 2, "aggregation": "topk_mean",
        "decision": "go_practical"},
    3: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp",
        "decision": "go_strict"},
    4: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp",
        "decision": "go_practical"},
    5: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp",
        "decision": "go_practical"},
    6: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp",
        "decision": "go_practical"},
    7: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp",
        "decision": "go_strict"},
    8: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp",
        "decision": "go_strict"},
    9: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp",
        "decision": "go_strict"},
    10: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "go_practical"},
    11: {"target_sparsity": 0.70, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "go_strict"},
    12: {"target_sparsity": 0.65, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "go_strict"},
    13: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "go_strict"},
    14: {"target_sparsity": 0.70, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "go_strict"},
    15: {"target_sparsity": 0.70, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "go_strict"},
    16: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "go_strict"},
    17: {"target_sparsity": 0.70, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "go_strict"},
    18: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "go_strict"},
    19: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "go_strict"},
    20: {"target_sparsity": 0.65, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "go_strict"},
    21: {"target_sparsity": 0.70, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "go_strict"},
    22: {"target_sparsity": 0.30, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "quality_only_high_work"},
    23: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp",
         "decision": "conditional_relaxed"},
}

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
    p = argparse.ArgumentParser(description="Phase 8B group-size scan")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--group-scan-layers", type=int, nargs="+",
                    default=[3, 8, 12, 13, 16, 20])
    p.add_argument("--group-sizes", type=int, nargs="+",
                    default=[32, 16, 8, 4])
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str,
                    default="outputs/phase8b_aglr_certimask_group_scan")
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


def run_single_layer(
    q: torch.Tensor,
    k: torch.Tensor,
    layer_idx: int,
    *,
    block_size: int,
    target_sparsity: float,
    local_blocks: int,
    aggregation: str,
    group_size: int,
    ambiguity_mode: str,
) -> dict[str, float | int | str | bool]:
    """Run CertiMask for a single layer and return metrics row."""
    seq_len = q.shape[2]
    num_blocks = seq_len // block_size
    valid_mask = make_block_causal_valid_mask(
        num_blocks, num_blocks, device=q.device,
    ).expand(q.shape[0], q.shape[1], num_blocks, num_blocks)

    result = aglr_certimask_topk(
        q, k,
        block_size=block_size,
        target_sparsity=target_sparsity,
        local_blocks=local_blocks,
        sample_pattern="both_diagonals",
        aggregation=aggregation,
        group_size=group_size,
        ambiguity_mode=ambiguity_mode,
    )

    metrics = compute_aglr_certimask_metrics(result, valid_mask)

    return {
        "layer": layer_idx,
        "group_size": group_size,
        "ambiguity_mode": ambiguity_mode,
        "target_sparsity": target_sparsity,
        "aggregation": aggregation,
        "exact_match": metrics.exact_mask_match,
        "mismatch_count": metrics.mismatch_count,
        "valid_tiles": metrics.valid_tiles,
        "selected_tiles": metrics.selected_tiles,
        "row_certification_rate": metrics.row_certification_rate,
        "ambiguous_rate": metrics.ambiguous_rate,
        "fallback_rate": metrics.fallback_rate,
        "mean_interval_width": metrics.mean_interval_width,
        "p50_interval_width": metrics.p50_interval_width,
        "p90_interval_width": metrics.p90_interval_width,
        "p99_interval_width": metrics.p99_interval_width,
        "selected_ambiguous_rate": metrics.selected_ambiguous_rate,
        "rejected_ambiguous_rate": metrics.rejected_ambiguous_rate,
        "boundary_band_size_mean": metrics.boundary_band_size_mean,
        "boundary_band_size_p90": metrics.boundary_band_size_p90,
        "certified_keep_rate": metrics.certified_keep_rate,
        "certified_drop_rate": metrics.certified_drop_rate,
        "mean_margin_to_boundary": metrics.mean_margin_to_boundary,
        "p10_margin_to_boundary": metrics.p10_margin_to_boundary,
        "score_interval_width_over_margin_p50": metrics.score_interval_width_over_margin_p50,
        "score_interval_width_over_margin_p90": metrics.score_interval_width_over_margin_p90,
    }


def diagnose_layer8(
    q: torch.Tensor,
    k: torch.Tensor,
    group_sizes: list[int],
    block_size: int,
    target_sparsity: float,
    local_blocks: int,
) -> dict[str, object]:
    """Diagnose Layer 8 interval width anomaly."""
    diagnosis: dict[str, object] = {"layer": 8, "group_size_results": {}}

    for gs in group_sizes:
        k_q = quantize_int8_per_group(k, group_size=gs)

        # K scale statistics
        scales = k_q.scale.reshape(-1)
        k_l2 = torch.linalg.vector_norm(
            k.reshape(-1, k.shape[-1]).to(torch.float32), ord=2, dim=-1,
        )

        # Q statistics
        q_abs = q.abs().reshape(-1, q.shape[-1])
        q_abs_mean = float(q_abs.mean().item())
        q_abs_p99 = float(q_abs.quantile(0.99).item())

        # Run certimask to get interval stats
        seq_len = q.shape[2]
        num_blocks = seq_len // block_size
        valid_mask = make_block_causal_valid_mask(
            num_blocks, num_blocks, device=q.device,
        ).expand(q.shape[0], q.shape[1], num_blocks, num_blocks)

        result = aglr_certimask_topk(
            q, k, block_size=block_size,
            target_sparsity=target_sparsity,
            local_blocks=local_blocks,
            aggregation="logsumexp", group_size=gs,
        )
        metrics = compute_aglr_certimask_metrics(result, valid_mask)

        # Sample dot statistics
        from certimask.aglr_indexer import _generate_sample_positions
        positions = _generate_sample_positions(block_size, "both_diagonals")
        q_blocks = q[:, :, :num_blocks * block_size, :].reshape(
            q.shape[0], q.shape[1], num_blocks, block_size, q.shape[-1],
        )
        k_blocks = k[:, :, :num_blocks * block_size, :].reshape(
            k.shape[0], k.shape[1], num_blocks, block_size, k.shape[-1],
        )
        sqrt_d = (q.shape[-1] ** 0.5)
        dot_vals: list[float] = []
        for qi_pos, ki_pos in positions:
            q_s = q_blocks[:, :, :, qi_pos, :]
            k_s = k_blocks[:, :, :, ki_pos, :]
            dots = torch.einsum("bhqd,bhkd->bhqk", q_s, k_s) / sqrt_d
            dot_vals.extend(dots.reshape(-1).tolist())
        dot_tensor = torch.tensor(dot_vals)

        gs_result = {
            "k_scale_mean": float(scales.mean().item()),
            "k_scale_p90": float(scales.quantile(0.90).item()),
            "k_scale_p99": float(scales.quantile(0.99).item()),
            "k_scale_max": float(scales.max().item()),
            "k_l2_norm_mean": float(k_l2.mean().item()),
            "k_l2_norm_p90": float(k_l2.quantile(0.90).item()),
            "k_l2_norm_p99": float(k_l2.quantile(0.99).item()),
            "q_abs_mean": q_abs_mean,
            "q_abs_p99": q_abs_p99,
            "sample_dot_abs_mean": float(dot_tensor.abs().mean().item()),
            "sample_dot_abs_p99": float(dot_tensor.abs().quantile(0.99).item()),
            "interval_width_mean": metrics.mean_interval_width,
            "interval_width_p99": metrics.p99_interval_width,
            "fallback_rate": metrics.fallback_rate,
        }
        diagnosis["group_size_results"][str(gs)] = gs_result

    # Determine primary cause
    gs16 = diagnosis["group_size_results"]["16"]
    primary = "unknown"
    if gs16["k_scale_p99"] > 0.1:
        primary = "large_k_scales"
    elif gs16["q_abs_p99"] > 5.0:
        primary = "large_q_magnitude"
    elif gs16["interval_width_mean"] > 2.0:
        primary = "logsumexp_interval_amplification"
    else:
        primary = "small_topk_margin"
    diagnosis["primary_cause"] = primary

    return diagnosis


def save_csv(rows: list[dict[str, float | int | str | bool]], path: Path) -> None:
    if not rows:
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


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

    config = {
        "model_name": args.model_name,
        "context_length": args.context_length,
        "group_scan_layers": args.group_scan_layers,
        "group_sizes": args.group_sizes,
        "block_size": args.block_size,
        "device": args.device,
        "dtype": args.dtype,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # =============================================
    # Phase 1: Group-size scan on representative layers
    # =============================================
    print("\n" + "=" * 70)
    print("PHASE 1: GROUP-SIZE SCAN")
    print("=" * 70)

    group_scan_rows: list[dict[str, float | int | str | bool]] = []

    for layer_idx in args.group_scan_layers:
        policy = LAYER_POLICY[layer_idx]
        target_sp = float(policy["target_sparsity"])
        local_blks = int(policy["local_blocks"])
        agg = str(policy["aggregation"])

        print(f"\nLayer {layer_idx} (sparsity={target_sp}, agg={agg})")

        qkv = extract_qkv_from_qwen2(model, input_ids, layer_index=layer_idx)
        q = qkv.query
        k = expand_kv_heads(qkv.key, qkv.num_query_heads)

        for gs in args.group_sizes:
            for mode in ["partition", "threshold"]:
                row = run_single_layer(
                    q, k, layer_idx,
                    block_size=args.block_size,
                    target_sparsity=target_sp,
                    local_blocks=local_blks,
                    aggregation=agg,
                    group_size=gs,
                    ambiguity_mode=mode,
                )
                group_scan_rows.append(row)

                if mode == "partition":
                    print(
                        f"  gs={gs:2d} mode={mode:9s}  "
                        f"exact={row['exact_match']}  "
                        f"fallback={row['fallback_rate']:.4f}  "
                        f"cert_keep={row['certified_keep_rate']:.4f}  "
                        f"cert_drop={row['certified_drop_rate']:.4f}  "
                        f"width={row['mean_interval_width']:.4f}"
                    )

    save_csv(group_scan_rows, output_dir / "group_size_scan.csv")

    # =============================================
    # Phase 2: Ambiguity mode comparison
    # =============================================
    print("\n" + "=" * 70)
    print("PHASE 2: AMBIGUITY MODE COMPARISON")
    print("=" * 70)

    ambig_rows = [r for r in group_scan_rows if r["group_size"] == 16]
    save_csv(ambig_rows, output_dir / "ambiguity_mode_comparison.csv")

    partition_rows = [r for r in ambig_rows if r["ambiguity_mode"] == "partition"]
    threshold_rows = [r for r in ambig_rows if r["ambiguity_mode"] == "threshold"]

    if partition_rows and threshold_rows:
        mean_p = sum(r["fallback_rate"] for r in partition_rows) / len(partition_rows)
        mean_t = sum(r["fallback_rate"] for r in threshold_rows) / len(threshold_rows)
        print(f"  Partition mode mean fallback: {mean_p:.4f}")
        print(f"  Threshold mode mean fallback: {mean_t:.4f}")
        print(f"  Improvement: {(mean_t - mean_p) / mean_t * 100:.1f}%")

    # =============================================
    # Phase 3: Layer 8 anomaly diagnosis
    # =============================================
    print("\n" + "=" * 70)
    print("PHASE 3: LAYER 8 ANOMALY DIAGNOSIS")
    print("=" * 70)

    qkv8 = extract_qkv_from_qwen2(model, input_ids, layer_index=8)
    q8 = qkv8.query
    k8 = expand_kv_heads(qkv8.key, qkv8.num_query_heads)

    layer8_diag = diagnose_layer8(
        q8, k8, args.group_sizes,
        block_size=args.block_size,
        target_sparsity=0.75,
        local_blocks=0,
    )

    with open(output_dir / "layer8_anomaly_diagnosis.json", "w") as f:
        json.dump(layer8_diag, f, indent=2)

    print(f"  Primary cause: {layer8_diag['primary_cause']}")
    for gs_str, gs_data in layer8_diag["group_size_results"].items():
        print(f"  gs={gs_str}: width_mean={gs_data['interval_width_mean']:.4f}  "
              f"width_p99={gs_data['interval_width_p99']:.4f}  "
              f"fallback={gs_data['fallback_rate']:.4f}")

    # =============================================
    # Phase 4: Select best group_size
    # =============================================
    print("\n" + "=" * 70)
    print("PHASE 4: SELECT BEST GROUP SIZE")
    print("=" * 70)

    # Use partition mode results for selection
    partition_scan = [r for r in group_scan_rows if r["ambiguity_mode"] == "partition"]

    gs_fallback: dict[int, list[float]] = {}
    for r in partition_scan:
        gs = int(r["group_size"])
        gs_fallback.setdefault(gs, []).append(float(r["fallback_rate"]))

    gs_mean_fallback = {
        gs: sum(vals) / len(vals) for gs, vals in gs_fallback.items()
    }

    # Select: lowest fallback, prefer larger group_size if close
    sorted_gs = sorted(gs_mean_fallback.items(), key=lambda x: x[1])
    best_gs = sorted_gs[0][0]
    # If the best is close to a larger group_size, prefer the larger
    if len(sorted_gs) > 1:
        best_fallback = sorted_gs[0][1]
        for gs, fb in sorted_gs[1:]:
            if fb < best_fallback * 1.10 and gs > best_gs:
                best_gs = gs
                break

    print(f"  Mean fallback by group_size: {gs_mean_fallback}")
    print(f"  Best group_size: {best_gs}")

    # =============================================
    # Phase 5: Full 24-layer scan
    # =============================================
    print("\n" + "=" * 70)
    print("PHASE 5: FULL 24-LAYER SCAN")
    print("=" * 70)

    full24_rows: list[dict[str, float | int | str | bool]] = []
    all_layers = list(range(24)) if args.all_layers else list(range(24))

    for layer_idx in all_layers:
        policy = LAYER_POLICY[layer_idx]
        target_sp = float(policy["target_sparsity"])
        local_blks = int(policy["local_blocks"])
        agg = str(policy["aggregation"])
        decision = str(policy["decision"])

        qkv = extract_qkv_from_qwen2(model, input_ids, layer_index=layer_idx)
        q = qkv.query
        k = expand_kv_heads(qkv.key, qkv.num_query_heads)

        if agg != "logsumexp":
            print(f"  L{layer_idx:2d}  unsupported aggregation '{agg}', FP fallback")
            full24_rows.append({
                "layer": layer_idx,
                "decision_from_phase7e": decision,
                "target_sparsity": target_sp,
                "aggregation": agg,
                "group_size": best_gs,
                "ambiguity_mode": "partition",
                "unsupported_aggregation": True,
            })
            continue

        seq_len = q.shape[2]
        num_blocks = seq_len // args.block_size
        valid_mask = make_block_causal_valid_mask(
            num_blocks, num_blocks, device=q.device,
        ).expand(q.shape[0], q.shape[1], num_blocks, num_blocks)

        result = aglr_certimask_topk(
            q, k,
            block_size=args.block_size,
            target_sparsity=target_sp,
            local_blocks=local_blks,
            aggregation=agg,
            group_size=best_gs,
            ambiguity_mode="partition",
        )
        metrics = compute_aglr_certimask_metrics(result, valid_mask)

        row = {
            "layer": layer_idx,
            "decision_from_phase7e": decision,
            "target_sparsity": target_sp,
            "aggregation": agg,
            "group_size": best_gs,
            "ambiguity_mode": "partition",
            "valid_tiles": metrics.valid_tiles,
            "selected_tiles": metrics.selected_tiles,
            "exact_match": metrics.exact_mask_match,
            "mismatch_count": metrics.mismatch_count,
            "row_certification_rate": metrics.row_certification_rate,
            "ambiguous_rate": metrics.ambiguous_rate,
            "fallback_rate": metrics.fallback_rate,
            "mean_interval_width": metrics.mean_interval_width,
            "p90_interval_width": metrics.p90_interval_width,
            "p99_interval_width": metrics.p99_interval_width,
            "certified_keep_rate": metrics.certified_keep_rate,
            "certified_drop_rate": metrics.certified_drop_rate,
            "unsupported_aggregation": False,
        }
        full24_rows.append(row)

        print(
            f"  L{layer_idx:2d}  exact={metrics.exact_mask_match}  "
            f"fallback={metrics.fallback_rate:.4f}  "
            f"cert_keep={metrics.certified_keep_rate:.4f}  "
            f"cert_drop={metrics.certified_drop_rate:.4f}"
        )

    save_csv(full24_rows, output_dir / "full24_certimask_by_layer.csv")

    # Interval width by layer
    interval_rows = []
    for r in full24_rows:
        if r.get("unsupported_aggregation"):
            continue
        interval_rows.append({
            "layer": r["layer"],
            "mean_interval_width": r["mean_interval_width"],
            "p90_interval_width": r["p90_interval_width"],
            "p99_interval_width": r["p99_interval_width"],
        })
    save_csv(interval_rows, output_dir / "interval_width_by_layer.csv")

    # =============================================
    # Summary
    # =============================================
    supported_rows = [r for r in full24_rows if not r.get("unsupported_aggregation")]
    all_exact = all(r.get("exact_match", False) for r in supported_rows)
    total_mismatch = sum(r.get("mismatch_count", 0) for r in supported_rows)
    mean_fallback = (
        sum(r["fallback_rate"] for r in supported_rows) / len(supported_rows)
        if supported_rows else 0.0
    )
    mean_row_cert = (
        sum(r["row_certification_rate"] for r in supported_rows) / len(supported_rows)
        if supported_rows else 0.0
    )

    if all_exact and supported_rows:
        next_step = (
            "Phase 8C: full AGLR-C + CertiMask quality/work summary "
            "with per-layer certified vs fallback breakdown"
        )
    else:
        next_step = "Fix exact match issues before proceeding"

    summary = {
        "best_group_size": best_gs,
        "best_ambiguity_mode": "partition",
        "tested_layers_group_scan": args.group_scan_layers,
        "full24_exact_match": all_exact,
        "full24_total_mismatch_count": total_mismatch,
        "full24_mean_fallback_rate": mean_fallback,
        "full24_mean_row_certification_rate": mean_row_cert,
        "layer8_primary_cause": layer8_diag["primary_cause"],
        "gs_mean_fallback_partition": gs_mean_fallback,
        "recommended_next_step": next_step,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # README
    readme_lines = [
        "# Phase 8B: Group-Size Scan and Boundary Fallback Optimization",
        "",
        f"**Model:** {args.model_name}",
        f"**Best group_size:** {best_gs}",
        "**Best ambiguity_mode:** partition",
        "",
        "## Group-Size Scan (partition mode, gs=16)",
        "",
        "| Layer | gs=32 | gs=16 | gs=8 | gs=4 |",
        "|---|---|---|---|---|",
    ]
    for layer_idx in args.group_scan_layers:
        vals = []
        for gs in args.group_sizes:
            match = [r for r in group_scan_rows
                     if r["layer"] == layer_idx
                     and r["group_size"] == gs
                     and r["ambiguity_mode"] == "partition"]
            if match:
                vals.append(f"{match[0]['fallback_rate']:.4f}")
            else:
                vals.append("N/A")
        readme_lines.append(f"| {layer_idx} | {' | '.join(vals)} |")

    readme_lines += [
        "",
        f"## Layer 8 Primary Cause: {layer8_diag['primary_cause']}",
        "",
        "## Full 24-Layer Results",
        "",
        f"- All exact match: {all_exact}",
        f"- Total mismatch: {total_mismatch}",
        f"- Mean fallback rate: {mean_fallback:.4f}",
        f"- Mean row certification rate: {mean_row_cert:.4f}",
        "",
        f"## Recommended Next Step: {next_step}",
    ]
    with open(output_dir / "README_results.md", "w") as f:
        f.write("\n".join(readme_lines) + "\n")

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 8B SUMMARY")
    print("=" * 70)
    print(f"  Best group_size: {best_gs}")
    print("  Best ambiguity_mode: partition")
    print(f"  Full 24-layer exact match: {all_exact}")
    print(f"  Full 24-layer mismatch: {total_mismatch}")
    print(f"  Full 24-layer mean fallback: {mean_fallback:.4f}")
    print(f"  Full 24-layer mean row cert: {mean_row_cert:.4f}")
    print(f"  Layer 8 primary cause: {layer8_diag['primary_cause']}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
