# AGLR-C v2: Candidate-Pruned Indexer Design

## Motivation

AGLR-C v1 is a high-quality reference indexer that scores **all causal block
pairs** — O(N_blocks² × samples × D). It produces excellent masks (kept mass
0.91, cosine 0.98) but is too heavy for runtime: at L=8192 it is 73× slower
than dense FP16 SDPA.

AGLR-C v2 must be a **lightweight runtime indexer** that scores only a
**candidate subset** of block pairs, then applies the existing AGLR sampled
scoring and top-k certificate only on candidates.

## Architecture

```
AGLR-C v1 (teacher/reference):
  scores ALL causal block pairs
  high quality, too heavy for runtime

AGLR-C v2 (runtime indexer):
  cheap candidate generation → candidate-only AGLR scoring → top-k mask
  head/layer aware budget
  future: low-bit-first CertiMask
```

## Candidate Generation Modes

### 1. local_stride

The cheapest mode. Candidates include:
- **Local window**: nearest `local_blocks` causal blocks per query row
- **First/global block**: block 0 is always a candidate (global context)
- **Strided historical blocks**: every `stride`-th block

No dot products are used. Candidate fraction is controlled by
`local_blocks` and `stride`.

### 2. block_norm

Candidates are selected by a cheap norm proxy:
```
score(q_block, k_block) = ||q_block||_2 * ||k_block||_2
```

This requires computing per-block L2 norms (cheap) but the pairwise product
is O(N_blocks²) in the worst case. Marked as `uses_full_pair_proxy=True`.

Intended as a baseline, not necessarily the final direction.

### 3. coarse_to_fine (primary)

Two-level hierarchy:
1. **Coarse level**: group fine blocks (size 8) into coarse regions (size 64–128)
2. **Coarse candidate selection**: cheap selection at coarse level
3. **Expand**: map selected coarse regions to constituent fine blocks
4. **Fine scoring**: AGLR sampled scoring only on candidate fine tiles

Coarse selection uses:
- Local coarse region (always included)
- Strided coarse regions
- Coarse key norm top-k

This is the primary mode because it reduces the candidate space geometrically
without requiring full-pair computation at any level.

### 4. head_pattern

Fixed per-head routing based on head index:
- Some heads → local_stride (local-attention heads)
- Some heads → coarse_to_fine (global-attention heads)
- Some heads → block_norm (fallback)

No training or learned routing. Simple hash-based assignment.

## Related Work

| System | Key Idea | Relevance |
|---|---|---|
| XAttention | Antidiagonal pattern as attention proxy | Our AGLR-C v1 already uses antidiagonal sampling |
| FlexPrefill | Dynamic per-row budget with confidence | Adaptive budget for v2 |
| MInference | Head pattern routing (A-shape, vertical-slash) | head_pattern mode |
| Native Sparse Attention | Avoid full attention matrix materialization | coarse_to_fine philosophy |
| IndexCache / Kascade | Cross-layer candidate reuse | Future optimization |
| SeerAttention | Cheap gate before sparse attention | coarse gate in coarse_to_fine |
| SpargeAttn | Hardware-friendly coarse block sparse | Coarse block alignment |

## Success Criteria

### Keep candidate mode
- candidate_fraction ≤ 0.25
- teacher_selected_coverage ≥ 0.90
- total_indexer_ms ≤ dense_sdpa_fp16_ms × 2.0

### Strong target
- candidate_fraction ≤ 0.10
- teacher_selected_coverage ≥ 0.90
- total_indexer_ms ≤ dense_sdpa_fp16_ms

### Reject candidate mode
- candidate_fraction > 0.50
- or teacher_selected_coverage < 0.70
- or total_indexer_ms > dense_sdpa_fp16_ms × 10
- or uses_full_pair_scoring = true and latency remains high

## What Is NOT in Scope

- No sparse attention kernel
- No low-bit-first CertiMask (future phase)
- No LongBench / downstream evaluation
- No learned routing or training
- No end-to-end speedup claims
