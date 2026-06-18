#!/usr/bin/env python3
"""Phase 6A.1: Sparse mask oracle diagnostics and sparsity scan.

Compares mean-pooled score mask with oracle block mass mask, local window
mask, and random mask across different sparsities and block sizes.
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
    dense_attention_output,
    local_window_block_mask,
    oracle_block_mass_mask,
    random_valid_block_mask,
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
    p = argparse.ArgumentParser(description="Phase 6A.1 oracle diagnostics")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--block-sizes", type=int, nargs="+", default=[16])
    p.add_argument(
        "--target-sparsities", type=float, nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
    )
    p.add_argument("--layers", type=int, nargs="+", default=[12, 13, 22, 23])
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str, default="outputs/phase6a1_oracle_diagnostics")
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


def compute_metrics_for_mask(
    dense_out: torch.Tensor,
    dense_probs: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
    valid_block_mask: torch.Tensor,
    layer_index: int,
    target_sparsity: float,
) -> dict[str, float | int]:
    """Compute quality metrics for a given block mask."""
    sparse_out, sparse_probs = block_sparse_attention_output(
        q_full, k_full, v_full, block_mask, block_size=block_size, causal=True,
    )
    metrics = compute_attention_quality(
        dense_out, dense_probs, sparse_out, sparse_probs,
        block_mask, block_size,
        layer_index=layer_index, target_sparsity=target_sparsity,
        valid_block_mask=valid_block_mask,
    )
    return {
        "actual_tile_sparsity": metrics.actual_tile_sparsity,
        "token_mask_sparsity": metrics.token_mask_sparsity,
        "kept_attention_mass_mean": metrics.kept_attention_mass_mean,
        "kept_attention_mass_p50": metrics.kept_attention_mass_p50,
        "kept_attention_mass_p90": metrics.kept_attention_mass_p90,
        "kept_attention_mass_p99": metrics.kept_attention_mass_p99,
        "dropped_attention_mass_mean": metrics.dropped_attention_mass_mean,
        "output_l2_relative_mean": metrics.output_l2_relative_mean,
        "output_l2_relative_p90": metrics.output_l2_relative_p90,
        "output_cosine_mean": metrics.output_cosine_mean,
        "output_cosine_p10": metrics.output_cosine_p10,
        "prob_l1_mean": metrics.prob_l1_mean,
        "prob_l1_p90": metrics.prob_l1_p90,
        "prob_kl_mean": metrics.prob_kl_mean,
        "prob_kl_p90": metrics.prob_kl_p90,
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

    input_ids = prepare_text(tokenizer, args.text, args.context_length).to(args.device)
    print(f"Sequence length: {input_ids.shape[1]}")

    config = {
        "model_name": args.model_name, "context_length": args.context_length,
        "block_sizes": args.block_sizes, "target_sparsities": args.target_sparsities,
        "layers": args.layers, "device": args.device, "dtype": args.dtype,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_rows: list[dict[str, float | int | str]] = []

    for layer_idx in args.layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        qkv = extract_qkv_from_qwen2(model, input_ids, layer_index=layer_idx)
        global q_full, k_full, v_full
        q_full = qkv.query
        k_full = expand_kv_heads(qkv.key, qkv.num_query_heads)
        v_full = expand_kv_heads(qkv.value, qkv.num_query_heads)

        # Dense reference
        dense_out, dense_probs = dense_attention_output(q_full, k_full, v_full, causal=True)

        for block_size in args.block_sizes:
            seq_len = q_full.shape[2]
            num_blocks = seq_len // block_size
            if num_blocks == 0:
                print(f"  Block size {block_size} too large for seq_len {seq_len}, skipping")
                continue

            valid_block_mask = make_block_causal_valid_mask(
                num_blocks, num_blocks, device=args.device,
            )
            # Expand to [1, H, Q_blk, K_blk] for consistency
            valid_block_mask = valid_block_mask.expand(
                q_full.shape[0], q_full.shape[1], num_blocks, num_blocks,
            )

            # Dense baseline (all valid tiles kept)
            dense_block_mask = valid_block_mask.clone()
            dense_metrics = compute_metrics_for_mask(
                dense_out, dense_probs, dense_block_mask, block_size,
                valid_block_mask, layer_idx, 0.0,
            )
            row = {"layer": layer_idx, "block_size": block_size,
                   "target_sparsity": 0.0, "mask_type": "dense", **dense_metrics}
            all_rows.append(row)
            km = dense_metrics['kept_attention_mass_mean']
            cs = dense_metrics['output_cosine_mean']
            print(f"  [bs={block_size}] dense: kept_mass={km:.4f} cosine={cs:.4f}")

            # Get block summaries for mean-pooled score mask
            summaries = mean_pool_qk_blocks(q_full, k_full, block_size=block_size)
            ref_scores = reference_scores(
                summaries.query, summaries.key, scale_by_sqrt_dim=True,
            )
            nb = summaries.num_blocks
            valid_scores_mask = valid_block_mask[:, :, :nb, :nb]

            for target_sp in args.target_sparsities:
                # Mean-pooled score mask
                thresholds = thresholds_for_target_sparsity(
                    ref_scores, target_sp, valid_mask=valid_scores_mask, per_query=True,
                )
                mp_mask = reference_mask(ref_scores, thresholds, valid_mask=valid_scores_mask)
                mp_metrics = compute_metrics_for_mask(
                    dense_out, dense_probs, mp_mask, block_size,
                    valid_block_mask, layer_idx, target_sp,
                )
                row = {
                    "layer": layer_idx, "block_size": block_size,
                    "target_sparsity": target_sp,
                    "mask_type": "mean_pooled_score", **mp_metrics,
                }
                all_rows.append(row)

                # Oracle block mass mask
                oracle_mask = oracle_block_mass_mask(
                    dense_probs, block_size=block_size,
                    target_sparsity=target_sp, valid_block_mask=valid_block_mask,
                )
                oracle_metrics = compute_metrics_for_mask(
                    dense_out, dense_probs, oracle_mask, block_size,
                    valid_block_mask, layer_idx, target_sp,
                )
                row = {
                    "layer": layer_idx, "block_size": block_size,
                    "target_sparsity": target_sp,
                    "mask_type": "oracle_block_mass", **oracle_metrics,
                }
                all_rows.append(row)

                # Local window mask
                window_blocks = max(1, int(num_blocks * (1.0 - target_sp)))
                local_mask = local_window_block_mask(
                    num_blocks, num_blocks, window_blocks=window_blocks,
                    device=args.device,
                ).expand_as(valid_block_mask)
                local_metrics = compute_metrics_for_mask(
                    dense_out, dense_probs, local_mask, block_size,
                    valid_block_mask, layer_idx, target_sp,
                )
                row = {"layer": layer_idx, "block_size": block_size,
                       "target_sparsity": target_sp, "mask_type": "local_window", **local_metrics}
                all_rows.append(row)

                # Random mask
                rand_mask = random_valid_block_mask(valid_block_mask, target_sparsity=target_sp)
                rand_metrics = compute_metrics_for_mask(
                    dense_out, dense_probs, rand_mask, block_size,
                    valid_block_mask, layer_idx, target_sp,
                )
                row = {"layer": layer_idx, "block_size": block_size,
                       "target_sparsity": target_sp, "mask_type": "random", **rand_metrics}
                all_rows.append(row)

                print(f"  [bs={block_size}] sp={target_sp:.2f} "
                      f"mp={mp_metrics['kept_attention_mass_mean']:.4f} "
                      f"oracle={oracle_metrics['kept_attention_mass_mean']:.4f} "
                      f"local={local_metrics['kept_attention_mass_mean']:.4f} "
                      f"rand={rand_metrics['kept_attention_mass_mean']:.4f}")

    # Save CSV
    if all_rows:
        fields = list(all_rows[0].keys())
        with open(output_dir / "quality_by_layer_mask_sparsity.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)

    # Summary by mask type
    summary_by_mask: dict[str, dict[str, float]] = {}
    mask_types = ["dense", "mean_pooled_score", "oracle_block_mass", "local_window", "random"]
    for mask_type in mask_types:
        rows = [r for r in all_rows
                if r["mask_type"] == mask_type and r["target_sparsity"] == 0.85]
        if rows:
            summary_by_mask[mask_type] = {
                "mean_kept_mass": sum(r["kept_attention_mass_mean"] for r in rows) / len(rows),
                "mean_cosine": sum(r["output_cosine_mean"] for r in rows) / len(rows),
                "mean_l2_rel": sum(r["output_l2_relative_mean"] for r in rows) / len(rows),
            }
    with open(output_dir / "summary_by_mask_type.csv", "w", newline="") as f:
        fields = ["mask_type", "mean_kept_mass", "mean_cosine", "mean_l2_rel"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mt, vals in summary_by_mask.items():
            w.writerow({"mask_type": mt, **vals})

    # Diagnosis
    oracle_85 = summary_by_mask.get("oracle_block_mass", {})
    mp_85 = summary_by_mask.get("mean_pooled_score", {})
    local_85 = summary_by_mask.get("local_window", {})
    random_85 = summary_by_mask.get("random", {})

    oracle_ok = oracle_85.get("mean_kept_mass", 0) >= 0.95
    mp_ok = mp_85.get("mean_kept_mass", 0) >= 0.80

    if oracle_ok and not mp_ok:
        failure_mode = "mean_pooled_indexer_weak"
        next_step = "improve_block_summary_indexer"
    elif not oracle_ok:
        failure_mode = "sparsity_or_block_size_too_aggressive"
        next_step = "reduce_target_sparsity_or_block_size"
    elif oracle_ok and mp_ok:
        failure_mode = "none"
        next_step = "proceed_to_triton"
    else:
        failure_mode = "mixed"
        next_step = "further_analysis"

    diagnosis = {
        "oracle_at_85_quality": oracle_85,
        "mean_pooled_at_85_quality": mp_85,
        "local_window_at_85_quality": local_85,
        "random_at_85_quality": random_85,
        "primary_failure_mode": failure_mode,
        "recommended_next_step": next_step,
    }
    with open(output_dir / "diagnosis_summary.json", "w") as f:
        json.dump(diagnosis, f, indent=2)

    # README
    readme = f"""# Phase 6A.1 Results

## Oracle Quality at 85% Sparsity
- Kept mass: {oracle_85.get('mean_kept_mass', 'N/A'):.4f}
- Cosine: {oracle_85.get('mean_cosine', 'N/A'):.4f}

## Mean-Pooled Score Quality at 85% Sparsity
- Kept mass: {mp_85.get('mean_kept_mass', 'N/A'):.4f}
- Cosine: {mp_85.get('mean_cosine', 'N/A'):.4f}

## Diagnosis
- Failure mode: {failure_mode}
- Recommended next step: {next_step}
"""
    with open(output_dir / "README_results.md", "w") as f:
        f.write(readme)

    print()
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print(f"  Oracle at 85%: kept_mass={oracle_85.get('mean_kept_mass', 0):.4f}")
    print(f"  Mean-pooled at 85%: kept_mass={mp_85.get('mean_kept_mass', 0):.4f}")
    print(f"  Local window at 85%: kept_mass={local_85.get('mean_kept_mass', 0):.4f}")
    print(f"  Random at 85%: kept_mass={random_85.get('mean_kept_mass', 0):.4f}")
    print(f"  Failure mode: {failure_mode}")
    print(f"  Next step: {next_step}")
    print()
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
