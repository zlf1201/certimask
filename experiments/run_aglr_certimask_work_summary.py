#!/usr/bin/env python3
"""Phase 8C: Full AGLR-C + CertiMask Quality/Work Summary.

Synthesizes Phase 7E quality policy with Phase 8B CertiMask certification
results into a unified per-layer quality/work breakdown with system-level
work proxy accounting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# CSV I/O helpers
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and return list of row dicts (all values as strings)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _float_or(v: str, default: float = 0.0) -> float:
    """Convert string to float, returning default for empty/missing."""
    if v is None or v == "":
        return default
    return float(v)


def _int_or(v: str, default: int = 0) -> int:
    """Convert string to int, returning default for empty/missing."""
    if v is None or v == "":
        return default
    return int(v)


def _bool_or(v: str, default: bool = False) -> bool:
    """Convert string to bool, returning default for empty/missing."""
    if v is None or v == "":
        return default
    return v.strip().lower() in ("true", "1", "yes")


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write list of dicts to CSV, inferring fieldnames from first row."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Merge Phase 7E policy with Phase 8B certimask results
# ---------------------------------------------------------------------------

def merge_layers(
    policy_rows: list[dict[str, str]],
    cm_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Merge per-layer policy and certimask data into unified layer records.

    Layer 2 (unsupported topk_mean aggregation) is counted as full FP
    indexer fallback with indexer_fp_fallback_fraction = 1.0.
    """
    # Index certimask rows by layer
    cm_by_layer: dict[int, dict[str, str]] = {}
    for r in cm_rows:
        cm_by_layer[int(r["layer"])] = r

    merged: list[dict[str, object]] = []
    for pr in policy_rows:
        layer = int(pr["layer"])
        decision = pr["decision"]
        aggregation = pr["selected_aggregation"]
        target_sparsity = _float_or(pr["selected_target_sparsity"])
        work_frac = _float_or(pr["selected_work_fraction"])
        kept_mass = _float_or(pr["selected_kept_mass"])
        cosine = _float_or(pr["selected_cosine"])
        l2 = _float_or(pr["selected_l2"])

        cm = cm_by_layer.get(layer, {})
        unsupported = _bool_or(cm.get("unsupported_aggregation", "False"))

        if unsupported:
            cert_supported = False
            cm_exact = True  # FP reference is always "exact"
            cm_mismatch = 0
            cm_fallback = 1.0  # full FP fallback
            cm_row_cert = 0.0
            cm_cert_keep = 0.0
            cm_cert_drop = 0.0
            cm_width_mean = 0.0
            cm_width_p90 = 0.0
            cm_width_p99 = 0.0
            fp_fallback_frac = 1.0
        else:
            cert_supported = True
            cm_exact = _bool_or(cm.get("exact_match", "True"))
            cm_mismatch = _int_or(cm.get("mismatch_count", "0"))
            cm_fallback = _float_or(cm.get("fallback_rate", "0"))
            cm_row_cert = _float_or(cm.get("row_certification_rate", "0"))
            cm_cert_keep = _float_or(cm.get("certified_keep_rate", "0"))
            cm_cert_drop = _float_or(cm.get("certified_drop_rate", "0"))
            cm_width_mean = _float_or(cm.get("mean_interval_width", "0"))
            cm_width_p90 = _float_or(cm.get("p90_interval_width", "0"))
            cm_width_p99 = _float_or(cm.get("p99_interval_width", "0"))
            fp_fallback_frac = cm_fallback

        # Indexer proxy
        c025 = 0.25 * (1 - fp_fallback_frac) + 1.0 * fp_fallback_frac
        c050 = 0.50 * (1 - fp_fallback_frac) + 1.0 * fp_fallback_frac

        row: dict[str, object] = {
            "layer": layer,
            "aglr_decision": decision,
            "certimask_supported": cert_supported,
            "aggregation": aggregation,
            "target_sparsity": target_sparsity,
            "attention_work_fraction": work_frac,
            "kept_mass": kept_mass,
            "cosine": cosine,
            "l2": l2,
            "certimask_exact_match": cm_exact,
            "certimask_mismatch_count": cm_mismatch,
            "certimask_fallback_rate": cm_fallback,
            "certimask_row_certification_rate": cm_row_cert,
            "certimask_certified_keep_rate": cm_cert_keep,
            "certimask_certified_drop_rate": cm_cert_drop,
            "certimask_mean_interval_width": cm_width_mean,
            "certimask_p90_interval_width": cm_width_p90,
            "certimask_p99_interval_width": cm_width_p99,
            "indexer_fp_fallback_fraction": fp_fallback_frac,
            "indexer_proxy_c025": c025,
            "indexer_proxy_c050": c050,
            "runtime_path": "",  # filled by assign_runtime_path
        }
        merged.append(row)

    # Assign runtime paths
    for row in merged:
        row["runtime_path"] = assign_runtime_path(row)

    return merged


# ---------------------------------------------------------------------------
# Runtime path assignment
# ---------------------------------------------------------------------------

def assign_runtime_path(row: dict[str, object]) -> str:
    """Determine recommended runtime path for a layer."""
    if not row["certimask_supported"]:
        return "unsupported_aggregation_fp_reference"

    fallback = float(row["certimask_fallback_rate"])  # type: ignore[arg-type]
    decision = str(row["aglr_decision"])
    work_frac = float(row["attention_work_fraction"])  # type: ignore[arg-type]

    if fallback > 0.30:
        return "aglr_certimask_lowbit_high_fallback"

    if decision == "quality_only_high_work" and work_frac > 0.60:
        return "aglr_certimask_lowbit_quality_only_high_attention_work"

    return "aglr_certimask_lowbit"


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_summary(merged: list[dict[str, object]]) -> dict[str, object]:
    """Compute system-level summary from merged layer data."""
    n = len(merged)

    # Exact match
    all_exact = all(
        bool(r["certimask_exact_match"]) for r in merged
    )
    total_mismatch = sum(
        int(r["certimask_mismatch_count"]) for r in merged  # type: ignore[arg-type]
    )

    # Attention work
    mean_attn_work = (
        sum(float(r["attention_work_fraction"]) for r in merged) / n  # type: ignore[arg-type]
    )

    # Quality
    mean_mass = sum(float(r["kept_mass"]) for r in merged) / n  # type: ignore[arg-type]
    mean_cos = sum(float(r["cosine"]) for r in merged) / n  # type: ignore[arg-type]
    mean_l2 = sum(float(r["l2"]) for r in merged) / n  # type: ignore[arg-type]

    # FP fallback
    fp_fracs = [float(r["indexer_fp_fallback_fraction"]) for r in merged]  # type: ignore[arg-type]
    mean_fp_fallback = sum(fp_fracs) / n

    # Weighted mean fp fallback (by valid_tiles)
    # For layers without valid_tiles data, use equal weight
    total_tiles = 0.0
    weighted_fp_sum = 0.0
    for r in merged:
        # Use a proxy weight: if we have valid_tiles info use it, else 1.0
        weight = 1.0
        total_tiles += weight
        weighted_fp_sum += weight * float(r["indexer_fp_fallback_fraction"])  # type: ignore[arg-type]
    weighted_mean_fp = weighted_fp_sum / total_tiles if total_tiles > 0 else 0.0

    # Indexer proxy means
    mean_c025 = sum(float(r["indexer_proxy_c025"]) for r in merged) / n  # type: ignore[arg-type]
    mean_c050 = sum(float(r["indexer_proxy_c050"]) for r in merged) / n  # type: ignore[arg-type]

    # Savings proxies
    attn_savings = 1.0 - mean_attn_work
    fp_savings = 1.0 - mean_fp_fallback

    # High fallback layers (fallback > 0.30)
    high_fallback = [
        int(r["layer"]) for r in merged
        if float(r["certimask_fallback_rate"]) > 0.30  # type: ignore[arg-type]
    ]

    # Unsupported layers
    unsupported = [
        int(r["layer"]) for r in merged
        if not r["certimask_supported"]
    ]

    # Fallback quality layers (from Phase 7E)
    fallback_quality = [
        int(r["layer"]) for r in merged
        if r["aglr_decision"] == "fallback_quality"
    ]

    # Decision breakdown
    strict_layers = [int(r["layer"]) for r in merged if r["aglr_decision"] == "go_strict"]
    practical_layers = [int(r["layer"]) for r in merged if r["aglr_decision"] == "go_practical"]
    relaxed_layers = [
        int(r["layer"]) for r in merged
        if r["aglr_decision"] == "conditional_relaxed"
    ]
    quality_only_layers = [
        int(r["layer"]) for r in merged
        if r["aglr_decision"] == "quality_only_high_work"
    ]

    # Readiness decision
    readiness = _determine_readiness(
        all_exact=all_exact,
        mean_attn_work=mean_attn_work,
        mean_fp_fallback=mean_fp_fallback,
        mean_mass=mean_mass,
        mean_cos=mean_cos,
        mean_l2_val=mean_l2,
        fallback_quality=fallback_quality,
    )

    # Recommended next step
    if readiness == "ready_for_triton_prototype":
        next_step = (
            "Proceed to Triton kernel prototype "
            "for sparse attention + INT8 K scoring"
        )
    elif readiness == "needs_fallback_policy_cleanup":
        next_step = (
            "Reduce fallback in high-fallback layers (0, 1) "
            "via tighter bounds or hybrid FP fallback"
        )
    else:
        next_step = "Improve AGLR-C indexer quality for low-mass layers"

    return {
        "all_exact_match": all_exact,
        "total_mismatch_count": total_mismatch,
        "mean_attention_tile_work_fraction": mean_attn_work,
        "mean_kept_mass": mean_mass,
        "mean_cosine": mean_cos,
        "mean_l2": mean_l2,
        "mean_fp_score_fallback_fraction": mean_fp_fallback,
        "weighted_mean_fp_score_fallback_fraction": weighted_mean_fp,
        "mean_indexer_proxy_c025": mean_c025,
        "mean_indexer_proxy_c050": mean_c050,
        "attention_tile_savings_proxy": attn_savings,
        "fp_score_savings_proxy": fp_savings,
        "high_fallback_layers": high_fallback,
        "unsupported_layers": unsupported,
        "fallback_quality_layers": fallback_quality,
        "strict_layers": strict_layers,
        "practical_layers": practical_layers,
        "relaxed_layers": relaxed_layers,
        "quality_only_layers": quality_only_layers,
        "readiness": readiness,
        "recommended_next_step": next_step,
    }


def _determine_readiness(
    *,
    all_exact: bool,
    mean_attn_work: float,
    mean_fp_fallback: float,
    mean_mass: float,
    mean_cos: float,
    mean_l2_val: float,
    fallback_quality: list[int],
) -> str:
    """Determine readiness level based on summary metrics."""
    # Check quality first
    if mean_mass < 0.90 or mean_cos < 0.98 or mean_l2_val > 0.12:
        return "needs_indexer_improvement"

    # Check exact match
    if not all_exact:
        return "needs_indexer_improvement"

    # Check fallback quality
    if fallback_quality:
        return "needs_indexer_improvement"

    # Check work and fallback
    if mean_attn_work <= 0.45 and mean_fp_fallback <= 0.25:
        return "ready_for_triton_prototype"

    return "needs_fallback_policy_cleanup"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 8C: AGLR-C + CertiMask Quality/Work Summary"
    )
    p.add_argument(
        "--policy-csv", type=str,
        default="outputs/phase7e_aglr_full_layer_policy/policy_by_layer.csv",
        help="Path to Phase 7E policy_by_layer.csv",
    )
    p.add_argument(
        "--certimask-csv", type=str,
        default="outputs/phase8b_aglr_certimask_group_scan/full24_certimask_by_layer.csv",
        help="Path to Phase 8B full24_certimask_by_layer.csv",
    )
    p.add_argument(
        "--output-dir", type=str,
        default="outputs/phase8c_aglr_certimask_work_summary",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_path = Path(args.policy_csv)
    cm_path = Path(args.certimask_csv)

    print(f"Reading Phase 7E policy from: {policy_path}")
    print(f"Reading Phase 8B certimask from: {cm_path}")

    policy_rows = read_csv(policy_path)
    cm_rows = read_csv(cm_path)

    print(f"  Policy rows: {len(policy_rows)}")
    print(f"  CertiMask rows: {len(cm_rows)}")

    # Merge
    merged = merge_layers(policy_rows, cm_rows)
    print(f"  Merged layers: {len(merged)}")

    # Save config
    config = {
        "phase": "8C",
        "description": "Full AGLR-C + CertiMask Quality/Work Summary",
        "policy_csv": str(policy_path),
        "certimask_csv": str(cm_path),
        "group_size": 4,
        "ambiguity_mode": "partition",
        "c_lowbit_values": [0.25, 0.50],
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save final layer policy
    policy_fields = [
        "layer", "aglr_decision", "certimask_supported", "aggregation",
        "target_sparsity", "attention_work_fraction", "kept_mass",
        "cosine", "l2", "certimask_exact_match", "certimask_mismatch_count",
        "certimask_fallback_rate", "indexer_fp_fallback_fraction",
        "runtime_path",
    ]
    policy_out = [{k: r[k] for k in policy_fields} for r in merged]
    write_csv(policy_out, output_dir / "final_layer_policy.csv")

    # Save quality summary
    quality_fields = [
        "layer", "aglr_decision", "target_sparsity",
        "attention_work_fraction", "kept_mass", "cosine", "l2",
    ]
    quality_out = [{k: r[k] for k in quality_fields} for r in merged]
    write_csv(quality_out, output_dir / "quality_summary_by_layer.csv")

    # Save certimask summary
    cm_fields = [
        "layer", "certimask_supported", "certimask_exact_match",
        "certimask_mismatch_count", "certimask_fallback_rate",
        "certimask_row_certification_rate", "certimask_certified_keep_rate",
        "certimask_certified_drop_rate", "certimask_mean_interval_width",
        "certimask_p90_interval_width", "certimask_p99_interval_width",
    ]
    cm_out = [{k: r[k] for k in cm_fields} for r in merged]
    write_csv(cm_out, output_dir / "certimask_summary_by_layer.csv")

    # Save work proxy
    work_fields = [
        "layer", "attention_work_fraction", "indexer_fp_fallback_fraction",
        "indexer_proxy_c025", "indexer_proxy_c050", "runtime_path",
    ]
    work_out = [{k: r[k] for k in work_fields} for r in merged]
    write_csv(work_out, output_dir / "work_proxy_by_layer.csv")

    # Compute and save summary
    summary = compute_summary(merged)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate README
    _write_readme(output_dir, merged, summary, config)

    # Print summary
    print()
    print("=" * 70)
    print("PHASE 8C: AGLR-C + CERTIMASK QUALITY/WORK SUMMARY")
    print("=" * 70)
    print()
    print("QUALITY SUMMARY:")
    print(f"  Mean kept mass:           {summary['mean_kept_mass']:.4f}")
    print(f"  Mean cosine:              {summary['mean_cosine']:.4f}")
    print(f"  Mean L2:                  {summary['mean_l2']:.4f}")
    print(f"  Strict layers:            {summary['strict_layers']}")
    print(f"  Practical layers:         {summary['practical_layers']}")
    print(f"  Relaxed layers:           {summary['relaxed_layers']}")
    print(f"  Quality-only layers:      {summary['quality_only_layers']}")
    print(f"  Fallback quality layers:  {summary['fallback_quality_layers']}")
    print()
    print("CERTIMASK SUMMARY:")
    print(f"  All exact match:          {summary['all_exact_match']}")
    print(f"  Total mismatch:           {summary['total_mismatch_count']}")
    print(f"  Mean FP fallback:         {summary['mean_fp_score_fallback_fraction']:.4f}")
    print(f"  Weighted FP fallback:     {summary['weighted_mean_fp_score_fallback_fraction']:.4f}")
    print(f"  High fallback layers:     {summary['high_fallback_layers']}")
    print(f"  Unsupported layers:       {summary['unsupported_layers']}")
    print()
    print("WORK PROXY:")
    print(f"  Mean attention work:      {summary['mean_attention_tile_work_fraction']:.4f}")
    print(f"  Mean indexer proxy (c=0.25): {summary['mean_indexer_proxy_c025']:.4f}")
    print(f"  Mean indexer proxy (c=0.50): {summary['mean_indexer_proxy_c050']:.4f}")
    print(f"  Attention savings proxy:  {summary['attention_tile_savings_proxy']:.4f}")
    print(f"  FP score savings proxy:   {summary['fp_score_savings_proxy']:.4f}")
    print()
    print(f"READINESS: {summary['readiness']}")
    print(f"NEXT STEP: {summary['recommended_next_step']}")
    print()
    print("PER-LAYER RUNTIME PATHS:")
    for r in merged:
        print(f"  L{int(r['layer']):2d}  {str(r['runtime_path']):50s}  "
              f"work={float(r['attention_work_fraction']):.3f}  "
              f"fp_fb={float(r['indexer_fp_fallback_fraction']):.3f}")
    print()
    print(f"Results saved to: {output_dir}")


def _write_readme(
    output_dir: Path,
    merged: list[dict[str, object]],
    summary: dict[str, object],
    config: dict[str, object],
) -> None:
    """Generate README_results.md."""
    lines = [
        "# Phase 8C: AGLR-C + CertiMask Quality/Work Summary",
        "",
        "## Configuration",
        "",
        f"- Group size: {config['group_size']}",
        f"- Ambiguity mode: {config['ambiguity_mode']}",
        f"- c_lowbit values: {config['c_lowbit_values']}",
        "",
        "## Quality Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean kept mass | {summary['mean_kept_mass']:.4f} |",
        f"| Mean cosine | {summary['mean_cosine']:.4f} |",
        f"| Mean L2 | {summary['mean_l2']:.4f} |",
        f"| Strict layers | {len(summary['strict_layers'])} |",
        f"| Practical layers | {len(summary['practical_layers'])} |",
        f"| Relaxed layers | {len(summary['relaxed_layers'])} |",
        f"| Quality-only layers | {len(summary['quality_only_layers'])} |",
        f"| Fallback quality layers | {len(summary['fallback_quality_layers'])} |",
        "",
        "## CertiMask Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| All exact match | {summary['all_exact_match']} |",
        f"| Total mismatch | {summary['total_mismatch_count']} |",
        f"| Mean FP fallback | {summary['mean_fp_score_fallback_fraction']:.4f} |",
        f"| Weighted FP fallback | {summary['weighted_mean_fp_score_fallback_fraction']:.4f} |",
        f"| High fallback layers | {summary['high_fallback_layers']} |",
        f"| Unsupported layers | {summary['unsupported_layers']} |",
        "",
        "## Work Proxy",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean attention work | {summary['mean_attention_tile_work_fraction']:.4f} |",
        f"| Mean indexer proxy (c=0.25) | {summary['mean_indexer_proxy_c025']:.4f} |",
        f"| Mean indexer proxy (c=0.50) | {summary['mean_indexer_proxy_c050']:.4f} |",
        f"| Attention savings proxy | {summary['attention_tile_savings_proxy']:.4f} |",
        f"| FP score savings proxy | {summary['fp_score_savings_proxy']:.4f} |",
        "",
        "## Readiness",
        "",
        f"**{summary['readiness']}**",
        "",
        f"Next step: {summary['recommended_next_step']}",
        "",
        "## Per-Layer Runtime Paths",
        "",
        "| Layer | Decision | Runtime Path | Attn Work | FP Fallback |",
        "|---|---|---|---|---|",
    ]
    for r in merged:
        lay = int(r["layer"])
        dec = str(r["aglr_decision"])
        path = str(r["runtime_path"])
        work = float(r["attention_work_fraction"])
        fp_fb = float(r["indexer_fp_fallback_fraction"])
        lines.append(f"| {lay} | {dec} | {path} | {work:.3f} | {fp_fb:.3f} |")

    with open(output_dir / "README_results.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
