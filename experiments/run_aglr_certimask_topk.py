#!/usr/bin/env python3
"""Phase 8A: CertiMask top-k certificate for AGLR-C v1.

Certifies AGLR-C v1 block selection using INT8 per-group K-only
quantization with logsumexp score interval analysis.
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

# Phase 7E per-layer policy
LAYER_POLICY: dict[int, dict[str, float | int | str]] = {
    0: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp"},
    1: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp"},
    2: {"target_sparsity": 0.50, "local_blocks": 2, "aggregation": "topk_mean"},
    3: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp"},
    4: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp"},
    5: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp"},
    6: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp"},
    7: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp"},
    8: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp"},
    9: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp"},
    10: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp"},
    11: {"target_sparsity": 0.70, "local_blocks": 0, "aggregation": "logsumexp"},
    12: {"target_sparsity": 0.65, "local_blocks": 0, "aggregation": "logsumexp"},
    13: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp"},
    14: {"target_sparsity": 0.70, "local_blocks": 0, "aggregation": "logsumexp"},
    15: {"target_sparsity": 0.70, "local_blocks": 0, "aggregation": "logsumexp"},
    16: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp"},
    17: {"target_sparsity": 0.70, "local_blocks": 0, "aggregation": "logsumexp"},
    18: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp"},
    19: {"target_sparsity": 0.75, "local_blocks": 0, "aggregation": "logsumexp"},
    20: {"target_sparsity": 0.65, "local_blocks": 0, "aggregation": "logsumexp"},
    21: {"target_sparsity": 0.70, "local_blocks": 0, "aggregation": "logsumexp"},
    22: {"target_sparsity": 0.30, "local_blocks": 0, "aggregation": "logsumexp"},
    23: {"target_sparsity": 0.50, "local_blocks": 0, "aggregation": "logsumexp"},
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
    p = argparse.ArgumentParser(description="Phase 8A CertiMask top-k certificate")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--layers", type=int, nargs="+", default=[3, 8, 12, 13, 16, 20])
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str,
                    default="outputs/phase8a_aglr_certimask_topk")
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
    print(f"Testing layers: {layers}")

    config = {
        "model_name": args.model_name,
        "context_length": args.context_length,
        "layers": layers,
        "block_size": args.block_size,
        "group_size": args.group_size,
        "device": args.device,
        "dtype": args.dtype,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    certimask_rows: list[dict[str, float | int | str | bool]] = []
    interval_rows: list[dict[str, float | int | str]] = []

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        policy = LAYER_POLICY.get(layer_idx)
        if policy is None:
            print(f"  No policy for layer {layer_idx}, skipping")
            continue

        aggregation = str(policy["aggregation"])
        target_sparsity = float(policy["target_sparsity"])
        local_blocks = int(policy["local_blocks"])

        if aggregation != "logsumexp":
            print(f"  Unsupported aggregation '{aggregation}', skipping")
            certimask_rows.append({
                "layer": layer_idx,
                "target_sparsity": target_sparsity,
                "block_size": args.block_size,
                "aggregation": aggregation,
                "group_size": args.group_size,
                "status": "unsupported_aggregation_fallback",
            })
            continue

        qkv = extract_qkv_from_qwen2(model, input_ids, layer_index=layer_idx)
        q = qkv.query
        k = expand_kv_heads(qkv.key, qkv.num_query_heads)

        result = aglr_certimask_topk(
            q, k,
            block_size=args.block_size,
            target_sparsity=target_sparsity,
            local_blocks=local_blocks,
            sample_pattern="both_diagonals",
            aggregation=aggregation,
            group_size=args.group_size,
        )

        seq_len = q.shape[2]
        num_blocks = seq_len // args.block_size
        valid_mask = make_block_causal_valid_mask(
            num_blocks, num_blocks, device=args.device,
        ).expand(q.shape[0], q.shape[1], num_blocks, num_blocks)

        metrics = compute_aglr_certimask_metrics(result, valid_mask)

        print(f"  Exact match: {metrics.exact_mask_match}")
        print(f"  Mismatch count: {metrics.mismatch_count}")
        print(f"  Row certification rate: {metrics.row_certification_rate:.4f}")
        print(f"  Ambiguous rate: {metrics.ambiguous_rate:.4f}")
        print(f"  Fallback rate: {metrics.fallback_rate:.4f}")
        print(f"  Mean interval width: {metrics.mean_interval_width:.6f}")

        certimask_rows.append({
            "layer": layer_idx,
            "target_sparsity": target_sparsity,
            "block_size": args.block_size,
            "aggregation": aggregation,
            "group_size": args.group_size,
            "valid_tiles": metrics.valid_tiles,
            "selected_tiles": metrics.selected_tiles,
            "row_certification_rate": metrics.row_certification_rate,
            "ambiguous_rate": metrics.ambiguous_rate,
            "fallback_rate": metrics.fallback_rate,
            "exact_mask_match": metrics.exact_mask_match,
            "mismatch_count": metrics.mismatch_count,
            "mean_interval_width": metrics.mean_interval_width,
            "p90_interval_width": metrics.p90_interval_width,
            "p99_interval_width": metrics.p99_interval_width,
        })

        interval_rows.append({
            "layer": layer_idx,
            "mean_interval_width": metrics.mean_interval_width,
            "p50_interval_width": metrics.p50_interval_width,
            "p90_interval_width": metrics.p90_interval_width,
            "p99_interval_width": metrics.p99_interval_width,
        })

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

    save_csv(certimask_rows, "certimask_topk_by_layer.csv")
    save_csv(interval_rows, "interval_stats_by_layer.csv")

    # Summary
    tested = [r for r in certimask_rows
              if r.get("status") != "unsupported_aggregation_fallback"]
    all_exact = all(r.get("exact_mask_match", False) for r in tested)
    total_mismatch = sum(r.get("mismatch_count", 0) for r in tested)
    mean_row_cert = (
        sum(r.get("row_certification_rate", 0.0) for r in tested) / len(tested)
        if tested else 0.0
    )
    mean_ambig = (
        sum(r.get("ambiguous_rate", 0.0) for r in tested) / len(tested)
        if tested else 0.0
    )
    mean_fallback = (
        sum(r.get("fallback_rate", 0.0) for r in tested) / len(tested)
        if tested else 0.0
    )

    if all_exact and tested:
        next_step = (
            "Proceed to Phase 8B: full 24-layer scan, "
            "group-size scan, or boundary optimization"
        )
    else:
        next_step = "Fix interval or fallback issues before proceeding"

    summary = {
        "tested_layers": [r["layer"] for r in tested],
        "all_exact_match": all_exact,
        "total_mismatch_count": total_mismatch,
        "mean_row_certification_rate": mean_row_cert,
        "mean_ambiguous_rate": mean_ambig,
        "mean_fallback_rate": mean_fallback,
        "group_size": args.group_size,
        "recommended_next_step": next_step,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # README
    readme_lines = [
        "# Phase 8A: CertiMask Top-k Certificate for AGLR-C v1",
        "",
        f"**Model:** {args.model_name}",
        f"**Context length:** {args.context_length}",
        f"**Group size:** {args.group_size}",
        f"**Block size:** {args.block_size}",
        "",
        "## Results by Layer",
        "",
        "| Layer | Sparsity | Exact | Row Cert | Ambiguous | Fallback | Interval Width |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in certimask_rows:
        if r.get("status") == "unsupported_aggregation_fallback":
            readme_lines.append(
                f"| {r['layer']} | {r['target_sparsity']} | "
                "N/A (unsupported aggregation) | | | | |"
            )
        else:
            readme_lines.append(
                f"| {r['layer']} | {r['target_sparsity']} | "
                f"{r.get('exact_mask_match', 'N/A')} | "
                f"{r.get('row_certification_rate', 0):.4f} | "
                f"{r.get('ambiguous_rate', 0):.4f} | "
                f"{r.get('fallback_rate', 0):.4f} | "
                f"{r.get('mean_interval_width', 0):.6f} |"
            )
    readme_lines += [
        "",
        f"## All Exact Match: {all_exact}",
        f"## Total Mismatch: {total_mismatch}",
        f"## Mean Row Certification Rate: {mean_row_cert:.4f}",
        f"## Mean Ambiguous Rate: {mean_ambig:.4f}",
        f"## Mean Fallback Rate: {mean_fallback:.4f}",
        "",
        f"## Recommended Next Step: {next_step}",
    ]
    with open(output_dir / "README_results.md", "w") as f:
        f.write("\n".join(readme_lines) + "\n")

    # Print summary
    print()
    print("=" * 70)
    print("CERTIMASK TOP-K CERTIFICATE SUMMARY")
    print("=" * 70)
    print(f"  Tested layers: {tested}")
    print(f"  All exact match: {all_exact}")
    print(f"  Total mismatch: {total_mismatch}")
    print(f"  Mean row certification rate: {mean_row_cert:.4f}")
    print(f"  Mean ambiguous rate: {mean_ambig:.4f}")
    print(f"  Mean fallback rate: {mean_fallback:.4f}")
    print()
    print("  Per-layer:")
    for r in certimask_rows:
        lay = r["layer"]
        if r.get("status") == "unsupported_aggregation_fallback":
            print(f"    L{lay:2d}  unsupported aggregation")
        else:
            print(
                f"    L{lay:2d}  exact={r.get('exact_mask_match')}"
                f"  row_cert={r.get('row_certification_rate', 0):.4f}"
                f"  ambig={r.get('ambiguous_rate', 0):.4f}"
                f"  fallback={r.get('fallback_rate', 0):.4f}"
                f"  width={r.get('mean_interval_width', 0):.6f}"
            )
    print()
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
