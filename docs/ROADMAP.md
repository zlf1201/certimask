# CertiMask Roadmap

## Completed Work

### Phase 3–8: Correctness and Certification
- INT8 per-vector and per-group quantization with analytic error bounds
- Score interval computation (coordinate, groupwise, K-only per-group)
- Threshold certificate (three-way KEEP/DROP/AMBIGUOUS) — **removed from codebase**
- AGLR-C v1 reference indexer (antidiagonal sampled scoring)
- AGLR-C quality validation (24-layer full scan)
- Top-k partition certificate (threshold and partition modes)
- Full AGLR-C CertiMask pipeline with exact match guarantee

### Phase 9A–9E: Triton Prototype and Profiling
- Triton JIT kernel for sampled scoring + logsumexp interval
- Fused Triton partition certificate (73,527x over PyTorch)
- Vectorized top-k mask construction (2000x over loop)
- Latency decomposition profiling
- Key quantization analysis
- Dense SDPA baseline measurement
- Long-context crossover analysis (L=1024–8192)

## Current Status

**The project is a minimal cleaned correctness and certification research prototype.**

- All certificates produce exact-match masks (mismatch_count = 0)
- Triton microbenchmarks show significant speedups over PyTorch baselines
- No end-to-end prefill speedup has been demonstrated
- No crossover with FlashAttention SDPA found at L≤8192
- Historical threshold/mean-pooled paths have been removed
- Only current AGLR-C v1 validation and optimized Triton components remain

## Next Steps

### Priority 1: Low-Bit-First Online CertiMask

The current path is reference-first:
```
FP32 reference scores → FP32 reference mask → low-bit interval → certificate
```

The deployable path must be low-bit-first:
```
Low-bit scores → low-bit top-k → interval certificate → FP fallback on ambiguous boundary only
```

This eliminates the FP32 reference scoring dependency from the online path.

### Priority 2: Candidate-Pruned / Retrieval / Hierarchical Indexer

The full-pair AGLR-C v1 indexer scores all O(N_blocks²) causal block pairs.
This is the fundamental scaling bottleneck. Future work should explore:
- Candidate pruning (score only promising block pairs)
- Retrieval-based indexing (landmarks, clustering)
- Hierarchical coarse-to-fine scoring

### Priority 3: Sparse Attention Kernel

Only if the above two priorities produce a crossover with dense SDPA,
implement a sparse attention kernel to realize the end-to-end speedup.

## What Will Not Be Done

- No sparse attention kernel until crossover is demonstrated
- No end-to-end model benchmark until sparse attention is ready
- No LongBench or downstream task evaluation in this prototype
- No INT4 quantization (out of scope for this prototype)
