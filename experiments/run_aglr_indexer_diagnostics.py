#!/usr/bin/env python3
"""Phase 7A: AGLR-C v0 indexer diagnostics.

Compares mean-pooled, landmark-based, and AGLR masks against oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from certimask.aglr_indexer import (
    aglr_adaptive_mass_budget_mask,
    aglr_local_plus_landmark_mask,
    compute_landmark_block_scores,
    select_block_landmarks,
)
from certimask.attention_quality import (
    block_sparse_attention_output,
    compute_attention_quality,
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
    p = argparse.ArgumentParser(description="Phase 7A AGLR indexer diagnostics")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--block-sizes", type=int, nargs="+", default=[16])
    p.add_argument(
        "--target-sparsities", type=float, nargs="+", default=[0.40, 0.50, 0.60],
    )
    p.add_argument("--layers", type=int, nargs="+", default=[12, 23])
    p.add_argument("--local-blocks", type=int, nargs="+", default=[1, 2])
    p.add_argument(
        "--landmark-methods", type=str, nargs="+",
        default=["mean_plus_max_norm", "mean_plus_topk_norm"],
    )
    p.add_argument("--num-landmarks", type=int, default=2)
    p.add_argument(
        "--score-methods", type=str, nargs="+", default=["max", "hybrid"],
    )
    p.add_argument(
        "--adaptive-mass-targets", type=float, nargs="+", default=[0.90, 0.95],
    )
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str, default="outputs/phase7a_aglr_indexer")
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
) -> dict[str, float | int | bool]:
    """Compute quality metrics and oracle gap."""
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

    quality_pass = (
        kept_mass >= 0.90 and cosine >= 0.95
        and l2_rel <= 0.20 and work_frac <= 0.50
    )
    strong = (
        kept_mass >= 0.95 and cosine >= 0.98
        and l2_rel <= 0.10 and work_frac <= 0.50
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
        "quality_pass": quality_pass,
        "strong_quality_pass": strong,
        "oracle_gap_mass": oracle_kept_mass - kept_mass,
        "oracle_gap_l2": l2_rel - oracle_l2,
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
        "layers": args.layers, "local_blocks": args.local_blocks,
        "landmark_methods": args.landmark_methods,
        "num_landmarks": args.num_landmarks,
        "score_methods": args.score_methods,
        "adaptive_mass_targets": args.adaptive_mass_targets,
        "device": args.device, "dtype": args.dtype,
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
            # Oracle at each sparsity for reference
            oracle_masks: dict[float, torch.Tensor] = {}
            oracle_masses: dict[float, float] = {}
            oracle_l2s: dict[float, float] = {}

            # Compute oracle for all sparsities first
            for target_sp in args.target_sparsities:
                oracle_mask = oracle_block_mass_mask(
                    dense_probs, block_size=block_size,
                    target_sparsity=target_sp, valid_block_mask=valid_block_mask,
                )
                oracle_masks[target_sp] = oracle_mask
                om = compute_quality(
                    dense_out, dense_probs, q_full, k_full, v_full,
                    oracle_mask, block_size, valid_block_mask, layer_idx, target_sp,
                    1.0, 0.0,
                )
                oracle_masses[target_sp] = om["kept_attention_mass_mean"]
                oracle_l2s[target_sp] = om["output_l2_relative_mean"]

            # Dense quality
            dq = compute_quality(
                dense_out, dense_probs, q_full, k_full, v_full,
                dense_mask, block_size, valid_block_mask, layer_idx, 0.0,
                1.0, 0.0,
            )
            row = {
                "layer": layer_idx, "block_size": block_size,
                "target_sparsity": 0.0, "mask_type": "dense",
                "local_blocks": 0, "landmark_method": "none",
                "num_landmarks": 0, "score_method": "none",
                "local_budget_overflow_rate": 0.0,
                "oracle_gap_mass": 0.0, "oracle_gap_l2": 0.0,
                **dq,
            }
            all_rows.append(row)

            # Block summaries for mean-pooled
            summaries = mean_pool_qk_blocks(q_full, k_full, block_size=block_size)
            ref_scores = reference_scores(
                summaries.query, summaries.key, scale_by_sqrt_dim=True,
            )
            nb = summaries.num_blocks
            valid_scores = valid_block_mask[:, :, :nb, :nb]

            # Landmark scores for each method
            landmark_score_cache: dict[tuple[str, str], torch.Tensor] = {}
            for lm_method in args.landmark_methods:
                q_lm = select_block_landmarks(
                    q_full, block_size=block_size,
                    method=lm_method, num_landmarks=args.num_landmarks,
                )
                k_lm = select_block_landmarks(
                    k_full, block_size=block_size,
                    method=lm_method, num_landmarks=args.num_landmarks,
                )
                for sc_method in args.score_methods:
                    lm_scores = compute_landmark_block_scores(
                        q_lm, k_lm,
                        score_method=sc_method,
                        valid_mask=valid_scores,
                    )
                    landmark_score_cache[(lm_method, sc_method)] = lm_scores

            for target_sp in args.target_sparsities:
                oracle_m = oracle_masses[target_sp]
                oracle_l = oracle_l2s[target_sp]

                # Random
                rand_mask = random_valid_block_mask(
                    valid_block_mask, target_sparsity=target_sp,
                )
                rq = compute_quality(
                    dense_out, dense_probs, q_full, k_full, v_full,
                    rand_mask, block_size, valid_block_mask, layer_idx, target_sp,
                    oracle_m, oracle_l,
                )
                row = {
                    "layer": layer_idx, "block_size": block_size,
                    "target_sparsity": target_sp, "mask_type": "random",
                    "local_blocks": 0, "landmark_method": "none",
                    "num_landmarks": 0, "score_method": "none",
                    "local_budget_overflow_rate": 0.0, **rq,
                }
                all_rows.append(row)

                # Mean-pooled score
                thresholds = thresholds_for_target_sparsity(
                    ref_scores, target_sp, valid_mask=valid_scores, per_query=True,
                )
                mp_mask = reference_mask(ref_scores, thresholds, valid_mask=valid_scores)
                mpq = compute_quality(
                    dense_out, dense_probs, q_full, k_full, v_full,
                    mp_mask, block_size, valid_block_mask, layer_idx, target_sp,
                    oracle_m, oracle_l,
                )
                row = {
                    "layer": layer_idx, "block_size": block_size,
                    "target_sparsity": target_sp, "mask_type": "mean_pooled_score",
                    "local_blocks": 0, "landmark_method": "none",
                    "num_landmarks": 0, "score_method": "none",
                    "local_budget_overflow_rate": 0.0, **mpq,
                }
                all_rows.append(row)

                # Oracle
                oq = compute_quality(
                    dense_out, dense_probs, q_full, k_full, v_full,
                    oracle_masks[target_sp], block_size, valid_block_mask,
                    layer_idx, target_sp, oracle_m, oracle_l,
                )
                row = {
                    "layer": layer_idx, "block_size": block_size,
                    "target_sparsity": target_sp, "mask_type": "oracle_block_mass",
                    "local_blocks": 0, "landmark_method": "none",
                    "num_landmarks": 0, "score_method": "none",
                    "local_budget_overflow_rate": 0.0, **oq,
                }
                all_rows.append(row)

                # Local window
                window_blocks = max(1, int(num_blocks * (1.0 - target_sp)))
                local_mask = local_window_block_mask(
                    num_blocks, num_blocks, window_blocks=window_blocks,
                    device=args.device,
                ).expand_as(valid_block_mask)
                lq = compute_quality(
                    dense_out, dense_probs, q_full, k_full, v_full,
                    local_mask, block_size, valid_block_mask, layer_idx, target_sp,
                    oracle_m, oracle_l,
                )
                row = {
                    "layer": layer_idx, "block_size": block_size,
                    "target_sparsity": target_sp, "mask_type": "local_window",
                    "local_blocks": 0, "landmark_method": "none",
                    "num_landmarks": 0, "score_method": "none",
                    "local_budget_overflow_rate": 0.0, **lq,
                }
                all_rows.append(row)

                # Landmark-only (no local guarantee)
                for lm_method in args.landmark_methods:
                    for sc_method in args.score_methods:
                        lm_scores = landmark_score_cache[(lm_method, sc_method)]
                        # Use local_plus_extra with local_blocks=0 as landmark-only
                        lm_mask, _ = local_plus_extra_mask(
                            lm_scores, target_sparsity=target_sp,
                            local_blocks=0, valid_mask=valid_scores,
                        )
                        if lm_mask.shape != valid_block_mask.shape:
                            padded = torch.zeros_like(valid_block_mask)
                            padded[:, :, :lm_mask.shape[2], :lm_mask.shape[3]] = lm_mask
                            lm_mask = padded
                        lmq = compute_quality(
                            dense_out, dense_probs, q_full, k_full, v_full,
                            lm_mask, block_size, valid_block_mask, layer_idx, target_sp,
                            oracle_m, oracle_l,
                        )
                        row = {
                            "layer": layer_idx, "block_size": block_size,
                            "target_sparsity": target_sp,
                            "mask_type": "landmark",
                            "local_blocks": 0,
                            "landmark_method": lm_method,
                            "num_landmarks": args.num_landmarks,
                            "score_method": sc_method,
                            "local_budget_overflow_rate": 0.0,
                            **lmq,
                        }
                        all_rows.append(row)

                # Local + landmark (AGLR-C v0)
                for local_blks in args.local_blocks:
                    for lm_method in args.landmark_methods:
                        for sc_method in args.score_methods:
                            lm_scores = landmark_score_cache[(lm_method, sc_method)]
                            aglr_result = aglr_local_plus_landmark_mask(
                                lm_scores, target_sparsity=target_sp,
                                local_blocks=local_blks, valid_mask=valid_scores,
                            )
                            aglr_mask = aglr_result.mask
                            if aglr_mask.shape != valid_block_mask.shape:
                                padded = torch.zeros_like(valid_block_mask)
                                padded[:, :, :aglr_mask.shape[2], :aglr_mask.shape[3]] = aglr_mask
                                aglr_mask = padded
                            aglrq = compute_quality(
                                dense_out, dense_probs, q_full, k_full, v_full,
                                aglr_mask, block_size, valid_block_mask,
                                layer_idx, target_sp, oracle_m, oracle_l,
                            )
                            row = {
                                "layer": layer_idx, "block_size": block_size,
                                "target_sparsity": target_sp,
                                "mask_type": "local_plus_landmark",
                                "local_blocks": local_blks,
                                "landmark_method": lm_method,
                                "num_landmarks": args.num_landmarks,
                                "score_method": sc_method,
                                "local_budget_overflow_rate": (
                                    aglr_result.local_budget_overflow_rate
                                ),
                                **aglrq,
                            }
                            all_rows.append(row)

                # AGLR adaptive
                for mass_target in args.adaptive_mass_targets:
                    for lm_method in args.landmark_methods:
                        for sc_method in args.score_methods:
                            lm_scores = landmark_score_cache[(lm_method, sc_method)]
                            for local_blks in args.local_blocks:
                                adapt_result = aglr_adaptive_mass_budget_mask(
                                    lm_scores, target_proxy_mass=mass_target,
                                    local_blocks=local_blks, valid_mask=valid_scores,
                                )
                                adapt_mask = adapt_result.mask
                                if adapt_mask.shape != valid_block_mask.shape:
                                    padded = torch.zeros_like(valid_block_mask)
                                    n2 = adapt_mask.shape[2]
                                    k2 = adapt_mask.shape[3]
                                    padded[:, :, :n2, :k2] = adapt_mask
                                    adapt_mask = padded
                                adaptq = compute_quality(
                                    dense_out, dense_probs, q_full, k_full, v_full,
                                    adapt_mask, block_size, valid_block_mask,
                                    layer_idx, target_sp, oracle_m, oracle_l,
                                )
                                row = {
                                    "layer": layer_idx, "block_size": block_size,
                                    "target_sparsity": target_sp,
                                    "mask_type": f"aglr_adaptive_m{mass_target:.2f}",
                                    "local_blocks": local_blks,
                                    "landmark_method": lm_method,
                                    "num_landmarks": args.num_landmarks,
                                    "score_method": sc_method,
                                    "local_budget_overflow_rate": (
                                        adapt_result.local_budget_overflow_rate
                                    ),
                                    **adaptq,
                                }
                                all_rows.append(row)

                # Print summary
                print(
                    f"  sp={target_sp:.2f}:"
                    f" mp={mpq['kept_attention_mass_mean']:.4f}"
                    f" oracle={oracle_m:.4f}"
                    f" rand={rq['kept_attention_mass_mean']:.4f}"
                )

    # Save CSV
    if all_rows:
        fields = list(all_rows[0].keys())
        with open(output_dir / "quality_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)

    # Find best configs
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
            w.writerows(viable[:30])

    # Summary by mask type
    mask_types = sorted(set(r["mask_type"] for r in all_rows))
    summary_rows = []
    for mt in mask_types:
        for sp in args.target_sparsities:
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
                    "mean_oracle_gap": sum(r["oracle_gap_mass"] for r in rows) / len(rows),
                    "quality_pass_count": sum(1 for r in rows if r["quality_pass"]),
                })
    if summary_rows:
        with open(output_dir / "summary_by_mask_type.csv", "w", newline="") as f:
            fields = list(summary_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary_rows)

    # Oracle gap CSV
    oracle_gap_rows = [
        r for r in all_rows if r["mask_type"] != "dense"
    ]
    if oracle_gap_rows:
        with open(output_dir / "oracle_gap.csv", "w", newline="") as f:
            fields = ["layer", "block_size", "target_sparsity", "mask_type",
                       "oracle_gap_mass", "oracle_gap_l2"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in oracle_gap_rows:
                w.writerow({k: r[k] for k in fields})

    # Diagnosis
    has_viable = len(viable) > 0
    best_config = viable[0] if viable else None

    def best_for_type(mt: str) -> dict[str, float | str] | None:
        rows = [r for r in all_rows if r["mask_type"] == mt and r["quality_pass"]]
        if not rows:
            return None
        best = min(rows, key=lambda r: r["output_l2_relative_mean"])
        return {
            "mask_type": mt,
            "kept_mass": best["kept_attention_mass_mean"],
            "cosine": best["output_cosine_mean"],
            "l2_rel": best["output_l2_relative_mean"],
            "work_frac": best["attention_tile_work_fraction"],
            "target_sparsity": best["target_sparsity"],
        }

    diagnosis = {
        "has_viable_indexer": has_viable,
        "best_indexer_config": {
            "mask_type": best_config["mask_type"],
            "block_size": best_config["block_size"],
            "target_sparsity": best_config["target_sparsity"],
            "local_blocks": best_config["local_blocks"],
            "landmark_method": best_config["landmark_method"],
            "score_method": best_config["score_method"],
            "kept_mass": best_config["kept_attention_mass_mean"],
            "cosine": best_config["output_cosine_mean"],
            "l2_rel": best_config["output_l2_relative_mean"],
            "work_frac": best_config["attention_tile_work_fraction"],
        } if best_config else None,
        "mean_pooled_best": best_for_type("mean_pooled_score"),
        "oracle_best": best_for_type("oracle_block_mass"),
    }
    with open(output_dir / "diagnosis_summary.json", "w") as f:
        json.dump(diagnosis, f, indent=2)

    # Print summary
    print()
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print(f"  Viable configs: {has_viable}")
    if best_config:
        print(f"  Best: {best_config['mask_type']} bs={best_config['block_size']} "
              f"sp={best_config['target_sparsity']:.2f}")
        print(f"    mass={best_config['kept_attention_mass_mean']:.4f} "
              f"cosine={best_config['output_cosine_mean']:.4f} "
              f"l2={best_config['output_l2_relative_mean']:.4f} "
              f"work={best_config['attention_tile_work_fraction']:.4f}")
    print(f"  Mean-pooled best: {diagnosis['mean_pooled_best']}")
    print(f"  Oracle best: {diagnosis['oracle_best']}")
    print()
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
