# Cleanup Audit

Generated during Cleanup Phase 2.

## 1. Active Runtime APIs

These are used in the current optimized validation pipeline:

| API | Module | Purpose |
|---|---|---|
| `triton_aglr_certimask_logsumexp_g4` | `triton_aglr_ops` | Main Triton pipeline entry point |
| `triton_aglr_logsumexp_scoring` | `triton_aglr_kernels` | Triton scoring kernel |
| `triton_certified_topk_mask_partition` | `triton_topk_certificate` | Fused Triton partition certificate |
| `vectorized_topk_mask` | `vectorized_topk` | Optimized top-k mask construction |
| `compute_antidiagonal_block_scores` | `aglr_indexer` | FP32 reference block scoring |
| `quantize_int8_per_group` | `quantization` | INT8 per-group quantization |
| `make_block_causal_valid_mask` | `masking` | Causal valid mask builder |
| `compute_fallback_metrics` | `triton_aglr_ops` | Certificate fallback analysis |

## 2. Validation-Only APIs

Used for correctness validation, not in optimized benchmarks:

| API | Module | Purpose |
|---|---|---|
| `aglr_certimask_topk` | `aglr_certimask` | Full reference-first AGLR CertiMask |
| `compute_aglr_certimask_metrics` | `aglr_certimask` | Certification metrics |
| `compute_sampled_dot_intervals` | `aglr_certimask` | PyTorch dot interval computation |
| `logsumexp_interval` | `topk_certificate` | PyTorch logsumexp interval |
| `compute_attention_quality` | `attention_quality` | Attention quality metrics |
| `compute_benefit_proxy` | `attention_quality` | Benefit proxy metrics |

## 3. Historical APIs

Superseded by newer implementations. Kept for reference:

| API | Module | Purpose |
|---|---|---|
| `certified_topk_mask` | `topk_certificate` | PyTorch loop-based partition certificate (slow) |
| `aglr_local_plus_landmark_mask` | `aglr_indexer` | Loop-based mask construction (slow) |
| `aglr_adaptive_mass_budget_mask` | `aglr_indexer` | Adaptive mass budget mask |
| `certified_threshold_mask` | `masking` | Historical threshold certificate |
| `naive_quantized_mask` | `masking` | Naive quantized mask |
| `reference_mask` | `masking` | Reference mask |

## 4. Active Experiment Entrypoints

These scripts are current and should be run:

| Script | Purpose |
|---|---|
| `experiments/active/benchmark_aglr_crossover.py` | Long-context crossover analysis (Phase 9E) |
| `experiments/active/benchmark_aglr_triton.py` | PyTorch vs Triton scoring (Phase 9A) |
| `experiments/active/benchmark_aglr_triton_profile.py` | Triton latency decomposition (Phase 9B) |
| `experiments/active/benchmark_topk_mask.py` | Loop vs vectorized top-k (Phase 9C) |
| `experiments/active/benchmark_aglr_quantization_and_baseline.py` | Quantization + dense baseline (Phase 9D) |

## 5. Archived Experiment Candidates

Early-phase diagnostic and scan scripts. Moved to `experiments/archive/`:

| Script | Original Phase |
|---|---|
| `run_real_model_phase4.py` | Phase 4 real model diagnostics |
| `run_layer_scan.py` | Phase 5 layer scan |
| `run_hf_layer_scan.py` | Phase 5 HF layer scan |
| `run_attention_quality.py` | Phase 6A attention quality |
| `run_aglr_indexer_diagnostics.py` | Phase 7A indexer diagnostics |
| `run_aglr_antidiagonal_diagnostics.py` | Phase 7B antidiagonal diagnostics |
| `run_aglr_layerwise_policy_scan.py` | Phase 7C layerwise policy |
| `run_aglr_quality_work_frontier.py` | Phase 7D quality-work frontier |
| `run_aglr_full_layer_policy_scan.py` | Phase 7E full layer policy |
| `run_aglr_certimask_topk.py` | Phase 8A CertiMask top-k |
| `run_aglr_certimask_group_scan.py` | Phase 8B group scan |
| `run_aglr_certimask_work_summary.py` | Phase 8C work summary |
| `run_hf_diagnostics.py` | Phase 4 HF diagnostics |
| `run_hf_group_quant_scan.py` | Phase 5.5 group quant scan |
| `run_hf_konly_full_scan.py` | Phase 5.6 K-only scan |
| `run_hf_quant_strategy_scan.py` | Phase 5.5 quant strategy |
| `run_hf_threshold.py` | Phase 3 HF threshold |
| `run_synthetic_threshold.py` | Phase 3 synthetic threshold |
| `run_phase45b.py` | Phase 4.5b comparison |
| `run_local_hybrid_mask_diagnostics.py` | Phase 6A3 local hybrid |
| `run_sparse_mask_oracle_diagnostics.py` | Phase 6A oracle diagnostics |

## 6. Slow-Path APIs That Must Not Be Used in Optimized Benchmarks

| API | Why slow | Replacement |
|---|---|---|
| `certified_topk_mask` | Python triple-nested loop over B×H×Q | `triton_certified_topk_mask_partition` |
| `aglr_local_plus_landmark_mask` | Python triple-nested loop over B×H×Q | `vectorized_topk_mask` |
| `certified_threshold_mask` | Historical, not AGLR-C | `triton_aglr_certimask_logsumexp_g4` |

## 7. Public Exports That Should Remain

Core stable APIs (no change needed):

- `quantize_int8_per_group`, `quantize_int8_per_vector`
- `GroupQuantizedTensor`, `QuantizedTensor`
- `compute_score_bounds`, `compute_groupwise_score_bounds`, etc.
- `quantized_int8_scores`, `group_quantized_int8_scores`, `reference_scores`
- `make_block_causal_valid_mask`
- `AMBIGUOUS`, `DROP`, `INVALID`, `KEEP`
- `logsumexp_interval`
- `vectorized_topk_mask`, `VectorizedTopKMaskResult`
- `TritonAGLRCertiMaskResult`, `triton_aglr_certimask_logsumexp_g4`
- `triton_certified_topk_mask_partition`
- `triton_aglr_logsumexp_scoring`
- `compute_fallback_metrics`

## 8. Public Exports That Should Be Removed Later

Currently exported but are historical/internal. Keeping for backward compatibility
but marking as historical:

| Export | Reason |
|---|---|
| `certified_topk_mask` | Slow PyTorch loop; use `triton_certified_topk_mask_partition` |
| `aglr_local_plus_landmark_mask` | Slow loop; use `vectorized_topk_mask` |
| `aglr_adaptive_mass_budget_mask` | Experimental, not used in active benchmarks |
| `certified_threshold_mask` | Historical baseline |
| `naive_quantized_mask` | Historical baseline |
| `compute_row_subset_stats` | Diagnostic helper |
| `compute_diagnostic_quantiles` | Diagnostic helper |
| `compute_per_tile_diagnostics` | Diagnostic helper |
| `compute_refinement_decomposition` | Diagnostic helper |
