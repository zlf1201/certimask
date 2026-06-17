# Phase 9E: Long-context Crossover Analysis

## Configuration
- Batch: 1, Heads: 14, D: 64
- Block size: 8, Group size: 4
- Sequence lengths: [1024, 2048, 4096, 8192]
- Warmup: 5, Iterations: 10
- Certificate: fused Triton (triton_certified_topk_mask_partition)

## Results Summary

| L | Dense FP16 | Dense FP32 | Online | Cached Q | Optimistic |
|---|---|---|---|---|---|
| 1024 | 0.05 | 0.18 | 7.15 | 1.83 | 1.07 |
| 2048 | 0.11 | 0.38 | 9.96 | 4.67 | 3.61 |
| 4096 | 0.26 | 1.25 | 25.35 | 19.98 | 13.97 |
| 8192 | 0.76 | 4.25 | 97.56 | 92.58 | 55.88 |

| L | Online Speedup | Cached Speedup | Optimistic Speedup |
|---|---|---|---|
| 1024 | 0.0063x | 0.0244x | 0.0414x |
| 2048 | 0.0114x | 0.0241x | 0.0311x |
| 4096 | 0.0103x | 0.0130x | 0.0186x |
| 8192 | 0.0078x | 0.0082x | 0.0136x |

## Scaling Exponents
- Dense SDPA (FP16): alpha = 1.34
- Dense SDPA (FP32): alpha = 1.54
- Online indexer: alpha = 1.27
- Triton scoring: alpha = 1.91

## Crossover Points
- Online: None
- Cached quantization: None
- Optimistic: None

## Readiness: not_viable_at_tested_lengths
No crossover found at tested lengths. Consider even longer contexts or fundamental redesign.
