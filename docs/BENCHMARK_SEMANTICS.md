# Benchmark Semantics

This document defines what each benchmark mode computes, whether it is
online-valid, and whether it represents a deployable path.

## Mode Definitions

### Reference-First Validation

- **What**: FP32 reference AGLR scores → FP32 reference mask → low-bit interval → certificate
- **Computes FP reference scores?** Yes
- **Computes reference mask?** Yes
- **Deployable online?** No — requires full FP32 reference scoring
- **Purpose**: Correctness validation. Proves that low-bit scoring produces
  the same mask as FP32 reference.

### Low-Bit-First Online (desired deployable path)

- **What**: Low-bit scores → low-bit top-k → interval certificate → FP fallback only on ambiguous boundary
- **Computes FP reference scores?** No
- **Computes reference mask?** No (low-bit mask is primary)
- **Deployable online?** Yes (if implemented)
- **Purpose**: The target deployment path. **Not yet implemented.**

### Mode 2A: Online Full with Quantization

- **What**: Full `triton_aglr_certimask_logsumexp_g4` pipeline
- **Includes**: FP32 reference scores, K quantization, Triton kernel, vectorized top-k, fused certificate
- **Deployable online?** No — includes FP32 reference scoring
- **Purpose**: Measures total pipeline cost with all components inside timing loop

### Mode 2B: Online Full with Cached Quantization

- **What**: K quantization outside timing loop; FP32 reference scores, vectorized top-k, Triton kernel, fused certificate inside
- **Computes FP reference scores?** Yes (inside timing)
- **Computes reference mask?** Yes (inside timing)
- **Deployable online?** No — still requires FP32 reference scoring
- **Purpose**: Measures cost savings from caching K quantization

### Mode 3: Optimistic Cached Indexer

- **What**: FP32 reference scores, vectorized top-k mask, K quantization all pre-computed outside timing; only Triton kernel + fused certificate inside
- **Computes FP reference scores?** No (pre-computed)
- **Computes reference mask?** No (pre-computed)
- **Deployable online?** No — assumes reference scores and mask are cached from prior computation
- **Purpose**: Measures lower-bound certificate overhead. Not a valid online cost.

### Dense SDPA Baseline

- **What**: `torch.nn.functional.scaled_dot_product_attention` with causal=True
- **Reports**: Both FP16 and FP32 timings
- **Purpose**: Baseline for crossover analysis

## Summary Table

| Mode | FP ref scores? | FP ref mask? | Deployable online? | Purpose |
|---|---|---|---|---|
| Reference-first validation | Yes | Yes | No | Correctness proof |
| Low-bit-first (desired) | No | No | Yes | **Not yet implemented** |
| Mode 2A (online full) | Yes | Yes | No | Total pipeline cost |
| Mode 2B (cached Q) | Yes | Yes | No | Quantization caching benefit |
| Mode 3 (optimistic) | Pre-computed | Pre-computed | No | Lower-bound certificate cost |
| Dense SDPA | N/A | N/A | N/A | Baseline |

## Crossover Analysis

The crossover benchmark compares Mode 2A/2B/3 against dense SDPA across
sequence lengths (L=1024–8192). The "speedup proxy" is:

```
total_proxy = indexer_cost + ideal_sparse_attention
speedup = dense_sdpa / total_proxy
```

where `ideal_sparse_attention = dense_sdpa × work_fraction` (work_fraction = 0.3765).

**No crossover was found at any tested length.** This means the current
reference-first AGLR-C pipeline is not competitive with FlashAttention
at L≤8192.
