# Phase 9C: Vectorized Top-k Reference Mask Construction

## Summary

Eliminated the `topk_reference_mask` bottleneck (322ms → 0.16ms) by replacing
the Python row-loop implementation with a fully vectorized PyTorch approach.

## Key Results

### Top-k Mask Optimization

| Implementation | Median (ms) | Speedup |
|---|---|---|
| Loop (original) | 321.46 | 1x |
| **Vectorized** | **0.156** | **2056x** |

- Mask exact match: **True**
- Mismatch count: **0**

### End-to-End Pipeline (with all Phase 9B + 9C optimizations)

| Stage | Median (ms) | Fraction |
|---|---|---|
| key_quantization | 4.81 | 74.5% |
| triton_score_interval_kernel | 0.89 | 13.8% |
| reference_fp32_aglr_score | 0.36 | 5.6% |
| topk_reference_mask (vectorized) | 0.16 | 2.4% |
| partition_certificate (fused Triton) | 0.09 | 1.4% |
| fallback_resolution | 0.14 | 2.2% |
| **TOTAL** | **6.44** | **100%** |

### Phase-over-Phase Improvement

| Phase | Total (ms) | Key Optimization |
|---|---|---|
| Phase 9A (baseline) | ~7100 | Triton scoring kernel |
| Phase 9B | 329 | Fused Triton certificate (73,527x) |
| **Phase 9C** | **6.44** | Vectorized top-k mask (2056x) |

**Overall speedup: ~1100x** from Phase 9A baseline.

## Implementation Details

### Vectorized Top-k Mask (`src/certimask/vectorized_topk.py`)

The key insight is that for `local_blocks=0` (23/24 layers), the top-k mask
construction reduces to a simple batched `torch.topk` + rank-based selection:

1. Compute per-row keep budget (vectorized)
2. Mask invalid/mandatory scores to -inf
3. `torch.topk(max_k)` across all rows simultaneously
4. Rank-based selection: keep entries where `rank < k_per_row`
5. One-hot encoding + OR aggregation to build dense mask

This eliminates:
- Triple-nested Python for-loop (1,792 iterations at B=1, H=14, Q=128)
- `.item()` CPU-GPU synchronization points (5,000+ per iteration)
- Per-row tensor allocations and kernel launches

### Mandatory Keep Mask Support

The vectorized implementation supports `mandatory_keep_mask` for `local_blocks > 0`:
- Mandatory blocks are always retained
- Extra budget = `k_per_row - mandatory_count`
- Top-k selection only among non-mandatory valid candidates

## Remaining Bottleneck

**key_quantization** at 4.81ms (74.5% of total) is now the largest stage.
This involves `quantize_int8_per_group` + `_expand_group_tensor`.

Potential optimizations:
- Fuse quantization into the Triton scoring kernel
- Pre-allocate and reuse quantization buffers
- Use INT8 key storage directly without dequantization

## Files

- `src/certimask/vectorized_topk.py` — Vectorized top-k mask implementation
- `src/certimask/triton_aglr_ops.py` — Updated with `topk_mask_mode` parameter
- `experiments/benchmark_topk_mask.py` — Standalone top-k mask benchmark
- `tests/test_vectorized_topk.py` — 12 tests for vectorized top-k
- `outputs/phase9c_topk_mask/` — Top-k mask benchmark results
- `outputs/phase9c_full_profile/` — Full pipeline profiling results
