# CertiMask

Runtime-certified low-bit sparse-prefill indexer research prototype.

## What This Project Is

CertiMask validates that INT8-quantized scoring can produce block-sparse
attention masks **identical** to FP32 reference, using certified score
intervals and top-k partition certificates.

This is a **correctness and certification research prototype**.
It does not demonstrate end-to-end prefill speedup.

## Current Status

- **Correctness**: Validated. CertiMask exact match (mismatch=0) across all tested configurations.
- **AGLR-C reference quality**: Validated. Mean kept mass 0.91, cosine 0.98, L2 0.11.
- **Triton scoring/certificate prototype**: Validated. Fused Triton certificate achieves ~73,000x over PyTorch.
- **End-to-end prefill speedup**: **Not demonstrated.**
- **Crossover vs dense SDPA**: **None found** at L=1024–8192.
- **Sparse attention kernel**: **Not implemented.**

## Key Results

| Metric | Value |
|---|---|
| CertiMask exact match | Yes (mismatch=0 in all validated settings) |
| Mean AGLR-C attention work fraction | 0.3765 |
| Mean kept attention mass | 0.9113 |
| Mean cosine similarity | 0.9847 |
| Mean L2 relative error | 0.1055 |
| Triton scoring microbenchmark | ~1.5x vs PyTorch CertiMask path |
| Fused Triton certificate | ~73,000x vs PyTorch certificate |
| Vectorized top-k | ~2,000x vs loop top-k |
| Crossover vs dense SDPA (L≤8192) | **None found** |

## What Is Not Claimed

- **No end-to-end model speedup is claimed.**
- **No sparse attention kernel is implemented.**
- **No LongBench or downstream task accuracy is measured.**
- **Full-pair AGLR-C v1 is not deployable as a speedup path against FlashAttention.**
- The low-bit certificate validates a reference sparse indexer, not dense attention exactness.

## Removed Historical Paths

The following historical paths have been removed from this repository:

- **Threshold / mean-pooled CertiMask path** — superseded by AGLR-C v1 top-k certificate
- **Synthetic threshold experiments** — early-phase diagnostic scripts
- **HF diagnostic scans** — layer/group/quant strategy scan scripts
- **Oracle/local hybrid diagnostic scripts** — attention quality and oracle mask diagnostics
- **Threshold-path metrics and diagnostics** — `attention_quality.py`, `diagnostics.py`, `metrics.py`, `synthetic.py`
- **Archive experiment scripts** — all Phase 3–8 historical scripts

See `docs/results/PHASE_SUMMARY.md` for curated results from all phases.

## Architecture

### Current Validation Path: Reference-First AGLR-C + CertiMask

```
FP32 reference scores → FP32 reference mask → low-bit interval → certificate
```

Good for correctness validation. Not deployable because it requires
full FP32 reference scoring (O(N_blocks²)).

### Optimized Validation Path: Triton + Fused Certificate

```
FP32 reference scores → vectorized top-k → Triton scoring → fused Triton certificate
```

Fastest validation path. Uses Triton JIT kernels for scoring and
fused partition certificate (~73,000x over PyTorch).

### Desired Deployable Path: Low-Bit-First (future work)

```
Low-bit scores → low-bit top-k → interval certificate → FP fallback on ambiguous boundary
```

This eliminates the FP32 reference dependency. Not yet implemented.

## Repository Layout

```
src/certimask/     Core library: quantization, scoring, bounds, certificates,
                   Triton kernels, vectorized top-k
tests/             Test suite (focused on current path)
experiments/active/    Current benchmark entrypoints
docs/              Research status, benchmark semantics, API path guide, roadmap
```

`outputs/` contains benchmark results and is not tracked in git.

## Installation

```bash
pip install -e ".[dev]"          # Basic (torch + pytest + ruff + mypy)
pip install -e ".[dev,hf]"       # With HuggingFace support
pip install -e ".[dev,triton]"   # With Triton kernel support (requires CUDA)
```

## Development

```bash
pytest -q
ruff check .
mypy src
```

## Benchmark Semantics

| Mode | FP ref scores? | Deployable? | Purpose |
|---|---|---|---|
| Reference-first validation | Yes | No | Correctness proof |
| Optimized validation | Yes | No | Fast validation with fused Triton cert |
| Low-bit-first (desired) | No | Yes | **Not yet implemented** |

See `docs/BENCHMARK_SEMANTICS.md` for full definitions.

## Roadmap

1. **Low-bit-first online CertiMask** — remove FP reference dependency
2. **Candidate-pruned / retrieval / hierarchical indexer** — reduce O(N²) scoring
3. **Sparse attention kernel** — only if crossover is demonstrated

See `docs/ROADMAP.md` for details.

## Detailed Results

See `docs/RESEARCH_STATUS.md` for validated vs unvalidated claims.
See `docs/results/PHASE_SUMMARY.md` for curated phase-by-phase results.
