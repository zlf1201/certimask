#!/usr/bin/env python3
"""Phase 6A.3: Local+hybrid mask diagnostics.

Compares local window, score-based, oracle, and local+hybrid masks
at lower sparsities to find viable sparse attention configurations.
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
    compute_oracle_block_mass_scores,
    dense_attention_output,
    local_plus_extra_mask,
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
    p = argparse.ArgumentParser(description="Phase 6A.3 local hybrid diagnostics")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--block-sizes", type=int, nargs="+", default=[8, 16])
    p.add_argument(
        "--target-sparsities", type=float, nargs="+",
        default=[0.50, 0.60, 0.70, 0.80],
    )
    p.add_argument("--layers", type=int, nargs="+", default=[12, 13, 22, 23])
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str, default="outputs/phase6a3_local_hybrid")
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
) -> dict[str, float | int | bool]:
    """Compute quality metrics for a block mask."""
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

    quality_pass = kept_mass >= 0.90 and cosine >= 0.95 and l2_rel <= 0.20
    strong = kept_mass >= 0.95 and cosine >= 0.98 and l2_rel <= 0.10

    valid_tiles = valid_block_mask.sum().item()
    kept_tiles = (block_mask & valid_block_mask).sum().item()
    work_frac = kept_tiles / valid_tiles if valid_tiles > 0 else 0.0

    return {
        "actual_tile_sparsity": metrics.actual_tile_sparsity,
        "attention_tile_work_fraction": work_frac,
        "kept_attention_mass_mean": metrics.kept_attention_mass_mean,
        "kept_attention_mass_p50": metrics.kept_attention_mass_p50,
        "kept_attention_mass_p90": metrics.kept_attention_mass_p90,
        "dropped_attention_mass_mean": metrics.dropped_attention_mass_mean,
        "output_l2_relative_mean": metrics.output_l2_relative_mean,
        "output_l2_relative_p90": metrics.output_l2_relative_p90,
        "output_cosine_mean": metrics.output_cosine_mean,
        "output_cosine_p10": metrics.output_cosine_p10,
        "prob_l1_mean": metrics.prob_l1_mean,
        "prob_kl_mean": metrics.prob_kl_mean,
        "token_mask_sparsity": metrics.token_mask_sparsity,
        "quality_pass": quality_pass,
        "strong_quality_pass": strong,
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

    config = {
        "model_name": args.model_name, "context_length": args.context_length,
        "block_sizes": args.block_sizes, "target_sparsities": args.target_sparsities,
        "layers": args.layers, "device": args.device, "dtype": args.dtype,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_rows: list[dict[str, float | int | str | bool]] = []

    for layer_idx in args.layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        qkv = extract_qkv_from_qwen2(model, input_ids, layer_index=layer_idx)
        q_full = qkv.query
        k_full = expand_kv_heads(qkv.key, qkv.num_query_heads)
        v_full = expand_kv_heads(qkv.value, qkv.num_query_heads)

        dense_out, dense_probs = dense_attention_output(q_full, k_full, v_full, causal=True)

        for block_size in args.block_sizes:
            seq_len = q_full.shape[2]
            num_blocks = seq_len // block_size
            if num_blocks == 0:
                continue

            valid_block_mask = make_block_causal_valid_mask(
                num_blocks, num_blocks, device=args.device,
            ).expand(q_full.shape[0], q_full.shape[1], num_blocks, num_blocks)

            # Dense baseline
            dense_mask = valid_block_mask.clone()
            dense_q = compute_quality(
                dense_out, dense_probs, q_full, k_full, v_full,
                dense_mask, block_size, valid_block_mask, layer_idx, 0.0,
            )
            row = {"layer": layer_idx, "block_size": block_size,
                   "target_sparsity": 0.0, "mask_type": "dense",
                   "local_blocks": 0, "local_budget_overflow_rate": 0.0,
                   **dense_q}
            all_rows.append(row)
            km = dense_q["kept_attention_mass_mean"]
            cs = dense_q["output_cosine_mean"]
            print(f"  [bs={block_size}] dense: mass={km:.4f} cosine={cs:.4f}")

            # Get block summaries and scores for mask generation
            summaries = mean_pool_qk_blocks(q_full, k_full, block_size=block_size)
            ref_scores = reference_scores(
                summaries.query, summaries.key, scale_by_sqrt_dim=True,
            )
            nb = summaries.num_blocks
            valid_scores = valid_block_mask[:, :, :nb, :nb]

            # Oracle block mass scores
            oracle_scores = compute_oracle_block_mass_scores(
                dense_probs, block_size=block_size, causal=True,
            )

            for target_sp in args.target_sparsities:
                # Random
                rand_mask = random_valid_block_mask(valid_block_mask, target_sparsity=target_sp)
                rand_q = compute_quality(
                    dense_out, dense_probs, q_full, k_full, v_full,
                    rand_mask, block_size, valid_block_mask, layer_idx, target_sp,
                )
                row = {"layer": layer_idx, "block_size": block_size,
                       "target_sparsity": target_sp, "mask_type": "random",
                       "local_blocks": 0, "local_budget_overflow_rate": 0.0,
                       **rand_q}
                all_rows.append(row)

                # Mean-pooled score
                thresholds = thresholds_for_target_sparsity(
                    ref_scores, target_sp, valid_mask=valid_scores, per_query=True,
                )
                mp_mask = reference_mask(ref_scores, thresholds, valid_mask=valid_scores)
                mp_q = compute_quality(
                    dense_out, dense_probs, q_full, k_full, v_full,
                    mp_mask, block_size, valid_block_mask, layer_idx, target_sp,
                )
                row = {"layer": layer_idx, "block_size": block_size,
                       "target_sparsity": target_sp, "mask_type": "mean_pooled_score",
                       "local_blocks": 0, "local_budget_overflow_rate": 0.0,
                       **mp_q}
                all_rows.append(row)

                # Oracle block mass
                oracle_mask = oracle_block_mass_mask(
                    dense_probs, block_size=block_size,
                    target_sparsity=target_sp, valid_block_mask=valid_block_mask,
                )
                oracle_q = compute_quality(
                    dense_out, dense_probs, q_full, k_full, v_full,
                    oracle_mask, block_size, valid_block_mask, layer_idx, target_sp,
                )
                row = {"layer": layer_idx, "block_size": block_size,
                       "target_sparsity": target_sp, "mask_type": "oracle_block_mass",
                       "local_blocks": 0, "local_budget_overflow_rate": 0.0,
                       **oracle_q}
                all_rows.append(row)

                # Local window variants
                for local_blocks in [1, 2, 4]:
                    if local_blocks > num_blocks:
                        continue
                    window_blocks = max(1, int(num_blocks * (1.0 - target_sp)))
                    local_mask = local_window_block_mask(
                        num_blocks, num_blocks, window_blocks=window_blocks,
                        device=args.device,
                    ).expand_as(valid_block_mask)
                    local_q = compute_quality(
                        dense_out, dense_probs, q_full, k_full, v_full,
                        local_mask, block_size, valid_block_mask, layer_idx, target_sp,
                    )
                    row = {
                        "layer": layer_idx, "block_size": block_size,
                        "target_sparsity": target_sp,
                        "mask_type": "local_window",
                        "local_blocks": local_blocks,
                        "local_budget_overflow_rate": 0.0,
                        **local_q,
                    }
                    all_rows.append(row)

                # Local + score
                for local_blocks in [1, 2, 4]:
                    if local_blocks > num_blocks:
                        continue
                    lp_mask, overflow = local_plus_extra_mask(
                        ref_scores, target_sparsity=target_sp,
                        local_blocks=local_blocks, valid_mask=valid_scores,
                    )
                    # Pad to full block mask size if needed
                    if lp_mask.shape != valid_block_mask.shape:
                        padded = torch.zeros_like(valid_block_mask)
                        padded[:, :, :lp_mask.shape[2], :lp_mask.shape[3]] = lp_mask
                        lp_mask = padded
                    lp_q = compute_quality(
                        dense_out, dense_probs, q_full, k_full, v_full,
                        lp_mask, block_size, valid_block_mask, layer_idx, target_sp,
                    )
                    row = {
                        "layer": layer_idx, "block_size": block_size,
                        "target_sparsity": target_sp,
                        "mask_type": "local_plus_score",
                        "local_blocks": local_blocks,
                        "local_budget_overflow_rate": 1.0 if overflow else 0.0,
                        **lp_q,
                    }
                    all_rows.append(row)

                # Local + oracle
                for local_blocks in [1, 2, 4]:
                    if local_blocks > num_blocks:
                        continue
                    lo_mask, overflow = local_plus_extra_mask(
                        oracle_scores, target_sparsity=target_sp,
                        local_blocks=local_blocks, valid_mask=valid_block_mask,
                    )
                    lo_q = compute_quality(
                        dense_out, dense_probs, q_full, k_full, v_full,
                        lo_mask, block_size, valid_block_mask, layer_idx, target_sp,
                    )
                    row = {
                        "layer": layer_idx, "block_size": block_size,
                        "target_sparsity": target_sp,
                        "mask_type": "local_plus_oracle",
                        "local_blocks": local_blocks,
                        "local_budget_overflow_rate": 1.0 if overflow else 0.0,
                        **lo_q,
                    }
                    all_rows.append(row)

                # Print summary for this sparsity
                print(
                    f"  [bs={block_size}] sp={target_sp:.2f}:"
                    f" mp={mp_q['kept_attention_mass_mean']:.4f}"
                    f" oracle={oracle_q['kept_attention_mass_mean']:.4f}"
                    f" rand={rand_q['kept_attention_mass_mean']:.4f}"
                )

    # Save CSV
    if all_rows:
        fields = list(all_rows[0].keys())
        with open(output_dir / "quality_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)

    # Find best configs (quality_pass AND work_fraction <= 0.50)
    viable = [
        r for r in all_rows
        if r["quality_pass"] and r["attention_tile_work_fraction"] <= 0.50
    ]
    viable.sort(key=lambda r: (r["output_l2_relative_mean"], r["attention_tile_work_fraction"]))

    if viable:
        with open(output_dir / "best_configs.csv", "w", newline="") as f:
            fields = list(viable[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(viable[:20])

    # Summary by mask type at key sparsities
    summary_rows = []
    for mt in ["dense", "random", "mean_pooled_score", "oracle_block_mass",
               "local_window", "local_plus_score", "local_plus_oracle"]:
        for sp in [0.50, 0.60, 0.70, 0.80]:
            rows = [r for r in all_rows if r["mask_type"] == mt and r["target_sparsity"] == sp]
            if rows:
                summary_rows.append({
                    "mask_type": mt,
                    "target_sparsity": sp,
                    "mean_kept_mass": sum(r["kept_attention_mass_mean"] for r in rows) / len(rows),
                    "mean_cosine": sum(r["output_cosine_mean"] for r in rows) / len(rows),
                    "mean_l2_rel": sum(r["output_l2_relative_mean"] for r in rows) / len(rows),
                    "mean_work_frac": (
                        sum(r["attention_tile_work_fraction"] for r in rows) / len(rows)
                    ),
                    "quality_pass_count": sum(1 for r in rows if r["quality_pass"]),
                })
    if summary_rows:
        with open(output_dir / "summary_by_mask_type.csv", "w", newline="") as f:
            fields = list(summary_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary_rows)

    # Diagnosis
    has_viable = len(viable) > 0
    best_config = viable[0] if viable else None
    strong_viable = [
        r for r in all_rows
        if r["strong_quality_pass"] and r["attention_tile_work_fraction"] <= 0.50
    ]
    best_strong = strong_viable[0] if strong_viable else None

    # Check viability of each mask type at 50% sparsity
    def is_viable(mt: str) -> bool:
        rows = [r for r in all_rows if r["mask_type"] == mt and r["target_sparsity"] == 0.50]
        return any(r["quality_pass"] and r["attention_tile_work_fraction"] <= 0.50 for r in rows)

    diagnosis = {
        "has_viable_sparse_mask": has_viable,
        "best_quality_pass_config": {
            "mask_type": best_config["mask_type"],
            "block_size": best_config["block_size"],
            "target_sparsity": best_config["target_sparsity"],
            "kept_mass": best_config["kept_attention_mass_mean"],
            "cosine": best_config["output_cosine_mean"],
            "l2_rel": best_config["output_l2_relative_mean"],
            "work_frac": best_config["attention_tile_work_fraction"],
        } if best_config else None,
        "best_strong_quality_pass_config": {
            "mask_type": best_strong["mask_type"],
            "block_size": best_strong["block_size"],
            "target_sparsity": best_strong["target_sparsity"],
            "kept_mass": best_strong["kept_attention_mass_mean"],
        } if best_strong else None,
        "mean_pooled_viable": is_viable("mean_pooled_score"),
        "local_window_viable": is_viable("local_window"),
        "local_plus_score_viable": is_viable("local_plus_score"),
        "oracle_upper_bound_viable": is_viable("oracle_block_mass"),
    }
    with open(output_dir / "diagnosis_summary.json", "w") as f:
        json.dump(diagnosis, f, indent=2)

    # Print summary
    print()
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print(f"  Viable configs found: {has_viable}")
    if best_config:
        print(f"  Best config: {best_config['mask_type']} bs={best_config['block_size']} "
              f"sp={best_config['target_sparsity']:.2f}")
        print(f"    mass={best_config['kept_attention_mass_mean']:.4f} "
              f"cosine={best_config['output_cosine_mean']:.4f} "
              f"l2={best_config['output_l2_relative_mean']:.4f} "
              f"work={best_config['attention_tile_work_fraction']:.4f}")
    print(f"  Mean-pooled viable: {diagnosis['mean_pooled_viable']}")
    print(f"  Local window viable: {diagnosis['local_window_viable']}")
    print(f"  Local+score viable: {diagnosis['local_plus_score_viable']}")
    print(f"  Oracle viable: {diagnosis['oracle_upper_bound_viable']}")
    print()
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
