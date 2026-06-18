# API Path Guide

This document describes the different code paths in CertiMask, their
purpose, and when to use each.

## Historical Baseline

**Threshold / mean-pooled path**

- `certified_threshold_mask` — threshold-based CertiMask
- `naive_quantized_mask` — naive quantized mask
- `reference_mask` — reference mask for comparison

These are early baselines superseded by AGLR-C. Kept for historical
reference and backward compatibility.

## Reference-First Validation Path

**Use for exactness validation. Not deployable.**

```
FP32 reference scores → FP32 reference mask → low-bit interval → certificate
```

Key APIs:
- `aglr_certimask_topk()` — full reference-first AGLR CertiMask
- `compute_antidiagonal_block_scores()` — FP32 reference block scoring
- `aglr_local_plus_landmark_mask()` — loop-based mask construction (slow)
- `certified_topk_mask()` — PyTorch loop-based partition certificate (slow)

This path computes full FP32 reference scores (O(N_blocks²)) before
certification. It proves correctness but cannot be used as a deployable
online path.

## Optimized Validation Path

**Triton scoring + vectorized top-k + fused certificate.
Still reference-first if FP scores are computed.**

```
FP32 reference scores → vectorized top-k → Triton scoring → fused Triton certificate
```

Key APIs:
- `triton_aglr_logsumexp_scoring()` — Triton scoring kernel
- `vectorized_topk_mask()` — optimized top-k mask construction
- `triton_certified_topk_mask_partition()` — fused Triton partition certificate
- `triton_aglr_certimask_logsumexp_g4()` — full pipeline wrapper

This path replaces the slow PyTorch loops with Triton kernels and
vectorized operations. It still computes FP32 reference scores, so it
is not deployable as a pure online path, but it is the fastest
validation path available.

## Desired Deployable Path

**Low-bit-first online CertiMask. Not implemented yet.**

```
Low-bit scores → low-bit top-k → interval certificate → FP fallback on ambiguous boundary
```

This path would eliminate the FP32 reference dependency entirely.
The low-bit scores would be primary, and FP fallback would only be
needed for ambiguous boundary tiles. **Not yet implemented.**

See `docs/ROADMAP.md` for implementation plan.

## Do Not Use for Optimized Benchmarks

The following APIs are slow and must not appear in benchmark hot paths:

| API | Why slow | Use instead |
|---|---|---|
| `certified_topk_mask()` | Python triple-nested loop over B×H×Q | `triton_certified_topk_mask_partition()` |
| `aglr_local_plus_landmark_mask()` | Python triple-nested loop over B×H×Q | `vectorized_topk_mask()` |
| `certified_threshold_mask()` | Historical baseline | `triton_aglr_certimask_logsumexp_g4()` |
| Full FP reference scores inside online path | O(N_blocks²) | Only for validation, not deployable |

## Benchmark Mode Summary

| Mode | FP ref scores? | Certificate | Deployable? | Purpose |
|---|---|---|---|---|
| Reference-first validation | Yes | PyTorch loop | No | Correctness proof |
| Optimized validation | Yes | Fused Triton | No | Fast validation |
| Low-bit-first (desired) | No | Fused Triton | Yes | **Not implemented** |
| Online full (Mode 2A) | Yes | Fused Triton | No | Total pipeline cost |
| Cached quantization (Mode 2B) | Yes | Fused Triton | No | Quantization caching |
| Optimistic cached (Mode 3) | Pre-computed | Fused Triton | No | Lower-bound certificate cost |

See `docs/BENCHMARK_SEMANTICS.md` for detailed mode definitions.
