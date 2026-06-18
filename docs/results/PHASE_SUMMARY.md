# Phase Summary

Curated summary of key results from each phase. Detailed outputs are not
tracked in git; this file preserves the essential findings.

## Phase 3–6: Foundation

- INT8 per-vector quantization: analytic L2 bound = sqrt(d) × scale / 2
- INT8 per-group quantization: per-coordinate analytic bound = alpha_g / 2
- Threshold certificate: exact match between CertiMask and FP32 reference
- Real-model integration with Qwen2/Qwen2.5-0.5B-Instruct

## Phase 7A–7E: AGLR-C Indexer and Quality

- AGLR-C v1 antidiagonal sampled scoring with both_diagonals pattern
- Block size=8, 16 samples per tile (main + anti diagonal)
- Full 24-layer policy scan on Qwen2.5-0.5B-Instruct
- Mean kept mass: 0.9113, mean cosine: 0.9847, mean L2: 0.1055
- Mean attention tile work fraction: 0.3765
- CertiMask readiness: ready_for_triton_prototype

## Phase 8A–8C: Top-k Certificate

- Top-k partition certificate with threshold and partition ambiguity modes
- Partition mode: tighter (L_t > U_R_max for selected, U_r < L_T_min for rejected)
- Group size scan: gs=4 selected (tightest bounds)
- Full 24-layer CertiMask scan: exact_match=True, mismatch=0
- Mean fallback rate: 0.1563 (gs=4, partition mode)
- Mean row certification rate: 0.1522

## Phase 9A: Triton Prototype

- Triton JIT kernel for sampled scoring + logsumexp interval
- Fixed: block_size=8, group_size=4, D=64, both_diagonals (16 samples)
- Synthetic: 1.50x speedup over PyTorch CertiMask (8.88ms → 5.88ms)
- Real Qwen layers: ~1.51x speedup, exact_match=True
- Max score/interval diff: ~1e-6

## Phase 9B: Certificate Fusion

- PyTorch partition certificate: 6776ms (95.5% of total)
- Fused Triton certificate: 0.09ms (73,527x speedup)
- Decisions and ambiguous masks exactly match PyTorch
- Bottleneck shifts to topk_reference_mask (323ms)

## Phase 9C: Vectorized Top-k

- Loop top-k mask: 321ms
- Vectorized top-k mask: 0.16ms (2056x speedup)
- Optimized total with fused cert + vectorized topk: 6.44ms
- Remaining bottleneck: key_quantization at 4.81ms

## Phase 9D: Quantization and Dense Baseline

- Key quantization is NOT the bottleneck (~5ms out of ~6700ms total)
- The bottleneck is FP32 reference score computation
- Dense SDPA FP16: 0.199ms at L=1024
- Cached pipeline (pre-computed reference): 5.86ms
- Cached pipeline is 29x slower than dense SDPA

## Phase 9E: Long-Context Crossover

- All modes use fused Triton certificate (not slow PyTorch)
- Scaling exponents: dense FP16 α=1.34, online indexer α=1.27, Triton scoring α=1.91
- No crossover found at L=1024–8192
- At L=8192: dense FP16=0.76ms, optimistic cached=55.88ms (73x overhead)
- Readiness: not_viable_at_tested_lengths

## Key Takeaway

The AGLR-C v1 full-pair indexer is correct and certifiable, but not
competitive with FlashAttention SDPA at L≤8192. The path to deployment
requires low-bit-first online CertiMask and candidate-pruned indexers.
