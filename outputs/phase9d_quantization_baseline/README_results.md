# Phase 9D: Key Quantization Optimization and Dense Attention Baseline Check

## Summary

Benchmarked key quantization optimization strategies and measured dense attention
baseline to determine if AGLR-C + CertiMask is viable at L=1024.

**Critical finding: Key quantization is NOT the bottleneck.** The AGLR-C indexer
(`compute_antidiagonal_block_scores`) dominates the pipeline at ~6700ms, while
quantization is only ~5ms.

## Latency Breakdown (median ms)

| Mode | Median (ms) | Description |
|---|---|---|
| Mode A: Current full | 6702.37 | Full `triton_aglr_certimask_logsumexp_g4` |
| Mode B: Cached quant | 6698.19 | Pre-quantize K outside loop, rest inside |
| Mode C: Preallocated | 5.86 | Pre-compute reference + cached quant + kernel |
| Quantization only | 4.92 | Just `quantize_int8_per_group` |

**Key insight:** Mode A ≈ Mode B (difference ~4ms). Pre-computing quantization
saves almost nothing. Mode C is fast because it pre-computes FP32 reference scores
outside the timing loop.

## Dense Attention Baseline

| Method | Median (ms) |
|---|---|
| torch SDPA | 0.199 |
| Manual dense | 0.295 |

## Pipeline Proxy

| Metric | Value |
|---|---|
| Mean attention tile work fraction | 0.3765 |
| Dense SDPA baseline | 0.199 ms |
| Ideal sparse attention | 0.075 ms |
| Current total proxy (full pipeline) | 6702.45 ms |
| Cached total proxy (Mode C) | 5.94 ms |
| Current speedup proxy vs dense | 0.00x |
| Cached speedup proxy vs dense | 0.03x |

## Readiness Decision

**`not_viable_at_L1024`**

The AGLR-C indexer itself is the bottleneck, not quantization:
- Cached pipeline (Mode C): 5.86ms
- Dense SDPA: 0.20ms
- Overhead: **29x** the dense attention time

Even with pre-computed reference scores and cached quantization, the indexer
overhead (5.86ms) dwarfs the dense attention baseline (0.20ms).

## What This Means

1. **Quantization optimization is not the path forward.** Pre-computing K
   quantization saves ~4ms out of ~6700ms — negligible.

2. **The AGLR-C indexer is the bottleneck.** `compute_antidiagonal_block_scores`
   takes ~6700ms and cannot be cached across different K tensors.

3. **At L=1024, the pipeline is not viable.** The indexer overhead is 29x the
   dense attention time.

4. **Longer contexts may help.** At L=4096 or L=8192, the dense attention time
   grows quadratically while the indexer grows linearly in the number of blocks.
   The crossover point needs investigation.

5. **The cached path (Mode C) at 5.86ms is the realistic per-token cost** if
   K is pre-computed from prefill and reused across decode steps.

## Files

- `config.json` — Benchmark configuration
- `benchmark_results.json` — Full results
- `pipeline_proxy_summary.json` — Summary metrics
- `quantization_benchmark.csv` — Quantization modes comparison
- `dense_attention_baseline.csv` — Dense attention baseline
