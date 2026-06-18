# CertiMask Research Status

## What This Project Is

CertiMask is a **research prototype** for runtime-certified low-bit sparse-prefill
indexing. It validates that INT8-quantized scoring can produce block-sparse attention
masks identical to FP32 reference, using certified score intervals and top-k
partition certificates.

## Validated Results

| Claim | Status | Evidence |
|---|---|---|
| INT8 per-group K-only quantization bounds are correct | **Validated** | `tests/test_quantization.py`, `tests/test_score_bounds.py` |
| Score intervals cover FP32 reference scores | **Validated** | `tests/test_aglr_certimask.py`, `tests/test_aglr_antidiagonal.py` |
| Threshold certificate: `M_CertiMask == M_ref` | **Validated** | `tests/test_masking.py` (20-seed stress test) |
| AGLR-C v1 reference indexer quality (kept mass, cosine, L2) | **Validated** | Phase 7E full 24-layer scan |
| Top-k partition certificate: `M_CertiMask == M_AGLR_ref` | **Validated** | Phase 8A/8B, exact_match=True, mismatch=0 across all layers |
| Triton sampled scoring + interval matches PyTorch | **Validated** | `tests/test_triton_aglr_certimask.py`, max diff ~1e-6 |
| Fused Triton partition certificate matches PyTorch | **Validated** | `tests/test_triton_topk_certificate.py` |
| Vectorized top-k mask matches loop implementation | **Validated** | `tests/test_vectorized_topk.py` |
| CertiMask readiness for Triton prototype | **Validated** | Phase 8C work summary |

## Key Quantitative Results

| Metric | Value |
|---|---|
| CertiMask exact match | **Yes** (mismatch_count = 0 in all validated settings) |
| Mean AGLR-C attention tile work fraction | **0.3765** |
| Mean kept attention mass | **0.9113** |
| Mean cosine similarity (sparse vs dense output) | **0.9847** |
| Mean L2 relative error | **0.1055** |
| Mean FP score fallback fraction | **0.1915** |
| Triton scoring microbenchmark speedup vs PyTorch | **~1.5x** |
| Fused Triton certificate vs PyTorch certificate | **~73,000x** (6776ms → 0.09ms) |
| Vectorized top-k vs loop top-k | **~2000x** (321ms → 0.16ms) |

## Not Validated / Not Claimed

| Claim | Status |
|---|---|
| End-to-end prefill speedup | **Not demonstrated** |
| Sparse attention kernel | **Not implemented** |
| Low-bit-first deployable online path | **Not implemented** |
| Downstream task quality (LongBench, etc.) | **Not measured** |
| Crossover vs dense SDPA at any tested length (L=1024–8192) | **Not found** |

## Current Blockers

1. **Reference-first path is not deployable**: The current AGLR-C pipeline
   computes FP32 reference scores first, then certifies low-bit intervals.
   This requires the full O(N_blocks²) reference scoring, which is slower
   than FlashAttention SDPA at all tested lengths.

2. **Full-pair block scoring is quadratic**: `compute_antidiagonal_block_scores`
   scores all causal block pairs, scaling as O(N_blocks² × samples × D).
   This has a higher constant factor than FlashAttention.

3. **No crossover found**: At L=1024–8192, the AGLR-C + CertiMask pipeline
   is 1.4–73x slower than dense FP16 SDPA, depending on mode and length.

## Correct Framing

This project is a **correctness and certification research prototype**.
It validates that low-bit scoring can be certified to produce identical
block-sparse masks as FP32 reference. It does **not** demonstrate
end-to-end acceleration. The full-pair AGLR-C v1 indexer is not a
deployable speedup path against FlashAttention.
