#!/usr/bin/env python3
"""Phase 9A/9B benchmark: PyTorch vs Triton AGLR-C CertiMask scoring.

Compares:
  1. PyTorch AGLR CertiMask scoring + interval (reference)
  2. Triton AGLR CertiMask scoring + interval
  3. FP32 AGLR reference sampled scoring (no quantization)

Reports true certificate fallback metrics from the partition certificate.

Usage (synthetic):
    python experiments/benchmark_aglr_triton.py \
        --batch-size 1 --num-heads 14 --seq-len 1024 \
        --head-dim 64 --block-size 8 --group-size 4 \
        --dtype float16 --device cuda --warmup 20 --iters 100 \
        --output-dir outputs/phase9a_triton_aglr_benchmark

Usage (real Qwen):
    python experiments/benchmark_aglr_triton.py \
        --use-real-qwen --model-name Qwen/Qwen2.5-0.5B-Instruct \
        --layers 8 12 16 20 --context-length 1024 \
        --block-size 8 --group-size 4 --dtype float16 --device cuda \
        --output-dir outputs/phase9a_triton_aglr_qwen_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from certimask.quantization import quantize_int8_per_group
from certimask.triton_aglr_kernels import triton_aglr_logsumexp_scoring
from certimask.triton_aglr_ops import (
    _expand_group_tensor,
    compute_fallback_metrics,
    triton_aglr_certimask_logsumexp_g4,
)


def _check_env() -> tuple[bool, bool]:
    """Return (cuda_ok, triton_ok)."""
    cuda_ok = torch.cuda.is_available()
    triton_ok = False
    if cuda_ok:
        try:
            import triton  # noqa: F401

            triton_ok = True
        except ImportError:
            pass
    return cuda_ok, triton_ok


def _make_synthetic(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic Q, K tensors."""
    gen = torch.Generator(device=device).manual_seed(42)
    q = torch.randn(batch, heads, seq_len, dim, dtype=dtype, device=device, generator=gen)
    k = torch.randn(batch, heads, seq_len, dim, dtype=dtype, device=device, generator=gen)
    return q, k


def _extract_qwen_qk(
    model_name: str,
    layer_idx: int,
    context_length: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract Q, K from a Qwen model layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.to(device).eval()

    text = (
        "The transformer architecture has revolutionized natural language processing "
        "by introducing self-attention mechanisms that allow models to weigh the "
        "importance of different parts of the input sequence."
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    while ids.shape[0] < context_length:
        ids = torch.cat([ids, ids])
    input_ids = ids[:context_length].unsqueeze(0).to(device)

    from certimask.block_summary import expand_kv_heads
    from certimask.hf_extraction import extract_qkv_from_qwen2

    qkv = extract_qkv_from_qwen2(model, input_ids, layer_index=layer_idx)
    q = qkv.query
    k = expand_kv_heads(qkv.key, qkv.num_query_heads)
    return q, k


def _cuda_timer(fn, warmup: int, iters: int) -> list[float]:
    """Time a callable using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _run_pytorch_certimask(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch AGLR CertiMask scoring (quantized/lower/upper)."""
    from certimask.aglr_certimask import _compute_sampled_dot_intervals
    from certimask.bounds import _get_group_per_coord_error
    from certimask.masking import make_block_causal_valid_mask
    from certimask.topk_certificate import logsumexp_interval

    batch, heads, seq_len, dim = q.shape
    num_blocks = seq_len // block_size
    device = q.device

    valid_mask = make_block_causal_valid_mask(
        num_blocks, num_blocks, device=device,
    ).expand(batch, heads, num_blocks, num_blocks)

    k_q = quantize_int8_per_group(k, group_size=group_size)
    k_tilde = k_q.dequantized.to(torch.float32)
    k_err = _get_group_per_coord_error(k_q, "analytic")

    _, lower_s, upper_s = _compute_sampled_dot_intervals(
        q, k_tilde, k_err, block_size=block_size,
        sample_pattern="both_diagonals", scale_by_sqrt_dim=True,
    )
    quant_s = (lower_s + upper_s) / 2.0

    lower, upper = logsumexp_interval(lower_s, upper_s, dim=-1)
    quantized = torch.logsumexp(quant_s, dim=-1)

    lower = lower.masked_fill(~valid_mask, torch.finfo(torch.float32).min)
    upper = upper.masked_fill(~valid_mask, torch.finfo(torch.float32).min)
    quantized = quantized.masked_fill(~valid_mask, torch.finfo(torch.float32).min)

    return quantized, lower, upper


def _run_fp32_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """FP32 AGLR reference scored (no quantization)."""
    from certimask.aglr_indexer import compute_antidiagonal_block_scores
    from certimask.masking import make_block_causal_valid_mask

    batch, heads, seq_len, _ = q.shape
    num_blocks = seq_len // block_size
    device = q.device

    valid_mask = make_block_causal_valid_mask(
        num_blocks, num_blocks, device=device,
    ).expand(batch, heads, num_blocks, num_blocks)

    return compute_antidiagonal_block_scores(
        q, k, block_size=block_size,
        sample_pattern="both_diagonals", aggregation="logsumexp",
        valid_mask=valid_mask, scale_by_sqrt_dim=True,
    )


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the full benchmark and return results."""
    cuda_ok, triton_ok = _check_env()
    if not cuda_ok:
        raise RuntimeError("CUDA not available")
    if not triton_ok:
        raise RuntimeError("Triton not installed")

    device = args.device
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    results: dict = {
        "config": vars(args),
        "cuda_device": torch.cuda.get_device_name(0),
        "triton_version": __import__("triton").__version__,
        "layers": {},
    }

    if args.use_real_qwen:
        layers = args.layers
        for layer_idx in layers:
            print(f"\n{'='*60}")
            print(f"Layer {layer_idx}")
            print(f"{'='*60}")

            q, k = _extract_qwen_qk(
                args.model_name, layer_idx, args.context_length, dtype, device,
            )
            layer_result = _benchmark_single(
                q, k, args.block_size, args.group_size, args.warmup, args.iters, device,
            )
            results["layers"][str(layer_idx)] = layer_result
            _print_layer_result(layer_idx, layer_result)
    else:
        q, k = _make_synthetic(
            args.batch_size, args.num_heads, args.seq_len, args.head_dim, dtype, device,
        )
        print(f"Synthetic: B={args.batch_size}, H={args.num_heads}, "
              f"L={args.seq_len}, D={args.head_dim}, dtype={args.dtype}")
        layer_result = _benchmark_single(
            q, k, args.block_size, args.group_size, args.warmup, args.iters, device,
        )
        results["layers"]["synthetic"] = layer_result
        _print_layer_result("synthetic", layer_result)

    # Aggregate
    all_layers = list(results["layers"].values())
    results["aggregate"] = {
        "pytorch_ms_median": _median([r["pytorch_certimask_ms_median"] for r in all_layers]),
        "triton_ms_median": _median([r["triton_certimask_ms_median"] for r in all_layers]),
        "fp32_ref_ms_median": _median([r["fp32_reference_ms_median"] for r in all_layers]),
        "speedup_vs_pytorch": _median([r["speedup_vs_pytorch_certimask"] for r in all_layers]),
        "speedup_vs_fp32_ref": _median([r["speedup_vs_reference"] for r in all_layers]),
        "max_quantized_diff": max(r["max_quantized_score_diff"] for r in all_layers),
        "max_lower_diff": max(r["max_lower_diff"] for r in all_layers),
        "max_upper_diff": max(r["max_upper_diff"] for r in all_layers),
        "all_exact_match": all(r["exact_match"] for r in all_layers),
        "total_mismatch": sum(r["mismatch_count"] for r in all_layers),
        "mean_fallback_rate": _median([r["fallback_rate"] for r in all_layers]),
        "mean_ambiguous_rate": _median([r["ambiguous_rate"] for r in all_layers]),
        "mean_row_certification_rate": _median([r["row_certification_rate"] for r in all_layers]),
        "mean_certified_keep_rate": _median([r["certified_keep_rate"] for r in all_layers]),
        "mean_certified_drop_rate": _median([r["certified_drop_rate"] for r in all_layers]),
    }

    return results


def _benchmark_single(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int,
    group_size: int,
    warmup: int,
    iters: int,
    device: str,
) -> dict:
    """Benchmark a single (Q, K) pair."""
    from certimask.masking import make_block_causal_valid_mask

    batch, heads, seq_len, dim = q.shape
    num_blocks = seq_len // block_size

    valid_mask = make_block_causal_valid_mask(
        num_blocks, num_blocks, device=device,
    ).expand(batch, heads, num_blocks, num_blocks)

    # --- FP32 reference ---
    fp32_ref_times = _cuda_timer(
        lambda: _run_fp32_reference(q, k, block_size), warmup, iters,
    )

    # --- PyTorch CertiMask ---
    pytorch_times = _cuda_timer(
        lambda: _run_pytorch_certimask(q, k, block_size, group_size), warmup, iters,
    )

    # --- Triton CertiMask ---
    # Pre-quantize once (not timed) to match PyTorch which also quantizes inside
    def triton_scoring() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_q = quantize_int8_per_group(k, group_size=group_size)
        key_scales_expanded = _expand_group_tensor(k_q.scale, group_size, dim).contiguous()
        key_is_zero_expanded = _expand_group_tensor(
            k_q.is_zero_group.to(torch.int8), group_size, dim,
        ).to(torch.bool).contiguous()
        return triton_aglr_logsumexp_scoring(
            q, k_q.values, key_scales_expanded, key_is_zero_expanded, valid_mask,
        )

    triton_times = _cuda_timer(triton_scoring, warmup, iters)

    # --- Accuracy comparison ---
    pytorch_q, pytorch_l, pytorch_u = _run_pytorch_certimask(q, k, block_size, group_size)
    triton_q, triton_l, triton_u = triton_scoring()

    valid = valid_mask
    max_q_diff = float((triton_q[valid] - pytorch_q[valid]).abs().max().item())
    max_l_diff = float((triton_l[valid] - pytorch_l[valid]).abs().max().item())
    max_u_diff = float((triton_u[valid] - pytorch_u[valid]).abs().max().item())

    # --- Exact match and fallback metrics via full pipeline ---
    # Use target_sparsity=0.5 as a representative test
    full_result = triton_aglr_certimask_logsumexp_g4(
        q, k, target_sparsity=0.5, local_blocks=0,
    )

    # Compute true fallback metrics from partition certificate
    fb_metrics = compute_fallback_metrics(full_result, valid_mask)

    pytorch_med = _median(pytorch_times)
    triton_med = _median(triton_times)
    fp32_med = _median(fp32_ref_times)

    return {
        "shape": {"B": batch, "H": heads, "L": seq_len, "D": dim},
        "pytorch_certimask_ms_median": pytorch_med,
        "triton_certimask_ms_median": triton_med,
        "fp32_reference_ms_median": fp32_med,
        "speedup_vs_pytorch_certimask": pytorch_med / triton_med if triton_med > 0 else 0,
        "speedup_vs_reference": fp32_med / triton_med if triton_med > 0 else 0,
        "max_quantized_score_diff": max_q_diff,
        "max_lower_diff": max_l_diff,
        "max_upper_diff": max_u_diff,
        "exact_match": full_result.exact_match,
        "mismatch_count": full_result.mismatch_count,
        "fallback_rate": fb_metrics["fallback_rate"],
        "ambiguous_rate": fb_metrics["ambiguous_rate"],
        "row_certification_rate": fb_metrics["row_certification_rate"],
        "certified_keep_rate": fb_metrics["certified_keep_rate"],
        "certified_drop_rate": fb_metrics["certified_drop_rate"],
    }


def _print_layer_result(layer, result: dict) -> None:
    """Pretty-print a single layer result."""
    print(f"  PyTorch CertiMask: {result['pytorch_certimask_ms_median']:.3f} ms")
    print(f"  Triton CertiMask:  {result['triton_certimask_ms_median']:.3f} ms")
    print(f"  FP32 Reference:    {result['fp32_reference_ms_median']:.3f} ms")
    print(f"  Speedup vs PyTorch: {result['speedup_vs_pytorch_certimask']:.2f}x")
    print(f"  Speedup vs FP32:    {result['speedup_vs_reference']:.2f}x")
    print(f"  Max quantized diff: {result['max_quantized_score_diff']:.6f}")
    print(f"  Max lower diff:     {result['max_lower_diff']:.6f}")
    print(f"  Max upper diff:     {result['max_upper_diff']:.6f}")
    print(f"  Exact match:        {result['exact_match']}")
    print(f"  Mismatch count:     {result['mismatch_count']}")
    print(f"  Fallback rate:      {result['fallback_rate']:.4f}")
    print(f"  Ambiguous rate:     {result['ambiguous_rate']:.4f}")
    print(f"  Row cert rate:      {result['row_certification_rate']:.4f}")
    print(f"  Cert keep rate:     {result['certified_keep_rate']:.4f}")
    print(f"  Cert drop rate:     {result['certified_drop_rate']:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 9A Triton AGLR benchmark")
    p.add_argument("--use-real-qwen", action="store_true")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16, 20])
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=14)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "float32", "bfloat16"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--output-dir", type=str,
                    default="outputs/phase9a_triton_aglr_benchmark")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 9A: Triton AGLR-C CertiMask Benchmark")
    print("=" * 60)

    results = run_benchmark(args)

    # Save
    with open(output_dir / "config.json", "w") as f:
        json.dump(results["config"], f, indent=2)

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # CSV for layers
    csv_lines = [
        "layer,pytorch_ms,triton_ms,fp32_ms,speedup_vs_pytorch,speedup_vs_fp32,"
        "max_q_diff,max_l_diff,max_u_diff,exact_match,mismatch_count,"
        "fallback_rate,ambiguous_rate,row_certification_rate,"
        "certified_keep_rate,certified_drop_rate",
    ]
    for layer_key, lr in results["layers"].items():
        csv_lines.append(
            f"{layer_key},{lr['pytorch_certimask_ms_median']:.4f},"
            f"{lr['triton_certimask_ms_median']:.4f},"
            f"{lr['fp32_reference_ms_median']:.4f},"
            f"{lr['speedup_vs_pytorch_certimask']:.4f},"
            f"{lr['speedup_vs_reference']:.4f},"
            f"{lr['max_quantized_score_diff']:.8f},"
            f"{lr['max_lower_diff']:.8f},"
            f"{lr['max_upper_diff']:.8f},"
            f"{lr['exact_match']},{lr['mismatch_count']},"
            f"{lr['fallback_rate']:.6f},{lr['ambiguous_rate']:.6f},"
            f"{lr['row_certification_rate']:.6f},"
            f"{lr['certified_keep_rate']:.6f},{lr['certified_drop_rate']:.6f}",
        )
    with open(output_dir / "benchmark_results.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    # Summary
    agg = results["aggregate"]
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"  PyTorch CertiMask median: {agg['pytorch_ms_median']:.3f} ms")
    print(f"  Triton CertiMask median:  {agg['triton_ms_median']:.3f} ms")
    print(f"  FP32 Reference median:    {agg['fp32_ref_ms_median']:.3f} ms")
    print(f"  Speedup vs PyTorch:       {agg['speedup_vs_pytorch']:.2f}x")
    print(f"  Speedup vs FP32:          {agg['speedup_vs_fp32_ref']:.2f}x")
    print(f"  Max quantized diff:       {agg['max_quantized_diff']:.8f}")
    print(f"  Max lower diff:           {agg['max_lower_diff']:.8f}")
    print(f"  Max upper diff:           {agg['max_upper_diff']:.8f}")
    print(f"  All exact match:          {agg['all_exact_match']}")
    print(f"  Total mismatch:           {agg['total_mismatch']}")
    print(f"  Mean fallback rate:       {agg['mean_fallback_rate']:.4f}")
    print(f"  Mean ambiguous rate:      {agg['mean_ambiguous_rate']:.4f}")
    print(f"  Mean row cert rate:       {agg['mean_row_certification_rate']:.4f}")
    print(f"  Mean cert keep rate:      {agg['mean_certified_keep_rate']:.4f}")
    print(f"  Mean cert drop rate:      {agg['mean_certified_drop_rate']:.4f}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
