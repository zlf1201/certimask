# Minimal Refactor Audit

Generated for the repo-slimming refactor. Goal: keep only the current-best
validation and runtime building blocks; delete abandoned paths.

## 1. Keep: Core Source Modules

| Module | Purpose |
|---|---|
| `quantization.py` | INT8 per-vector and per-group quantization |
| `scoring.py` | FP32 reference, INT8 quantized, K-only scoring |
| `bounds.py` | Score error bounds and interval validation |
| `aglr_indexer.py` | AGLR-C v1 antidiagonal scoring + `_generate_sample_positions` |
| `aglr_certimask.py` | Reference-first AGLR CertiMask validation pipeline |
| `topk_certificate.py` | Reference PyTorch partition certificate (documented as slow) |
| `vectorized_topk.py` | Optimized vectorized top-k mask |
| `triton_aglr_kernels.py` | Triton JIT scoring kernel |
| `triton_aglr_ops.py` | High-level Triton ops + new optimized wrapper |
| `triton_topk_certificate.py` | Fused Triton partition certificate |
| `masking.py` | `make_block_causal_valid_mask` + cert decision/result types |
| `block_summary.py` | GQA head expansion + mean-pool block summaries |
| `hf_extraction.py` | Optional HF model Q/K extraction (suppress import) |

## 2. Keep: Current Docs

| Doc | Purpose |
|---|---|
| `README.md` | Updated to reflect minimal repo |
| `docs/API_PATHS.md` | API path guide |
| `docs/RESEARCH_STATUS.md` | Validated vs unvalidated claims |
| `docs/ROADMAP.md` | Next research steps |
| `docs/BENCHMARK_SEMANTICS.md` | Benchmark mode definitions |
| `docs/results/PHASE_SUMMARY.md` | Phase-by-phase results |
| `docs/dev/MINIMAL_REFACTOR_AUDIT.md` | This file |
| `docs/dev/CLEANUP_AUDIT.md` | Previous cleanup audit (kept) |

## 3. Delete: Abandoned Source Modules

| Module | Reason |
|---|---|
| `attention_quality.py` | Diagnostic-only; not used by current Triton path or kept tests |
| `diagnostics.py` | Threshold-path diagnostics; not used by current pipeline |
| `synthetic.py` | Synthetic data generation for old experiments |
| `metrics.py` | Threshold-path mask/bound metrics; depends on `CertiMaskResult` from threshold path |

## 4. Delete: Abandoned Tests

| Test | Reason |
|---|---|
| `test_attention_quality.py` | Tests removed `attention_quality.py` |
| `test_diagnostics.py` | Tests removed `diagnostics.py` |
| `test_synthetic.py` | Tests removed `synthetic.py` |
| `test_metrics.py` | Tests removed `metrics.py` |
| `test_masking.py` | Tests threshold certificate path (removed) |
| `test_oracle_mask_diagnostics.py` | Oracle diagnostic tests |
| `test_local_hybrid_masks.py` | Local hybrid mask tests |
| `test_aglr_certimask_group_scan.py` | Group scan diagnostic tests |
| `test_aglr_certimask_work_summary.py` | Work summary diagnostic tests |
| `test_aglr_full_layer_policy.py` | Full layer policy scan tests |
| `test_aglr_policy_selection.py` | Policy selection diagnostic tests |
| `test_aglr_quality_work_frontier.py` | Quality-work frontier diagnostic tests |
| `test_block_summary.py` | Block summary tests (mean-pool path) |
| `test_hf_extraction.py` | HF extraction tests (requires model) |
| `test_gqa_mapping.py` | GQA mapping tests |
| `test_konly_policy.py` | K-only policy tests |
| `test_layer_policy.py` | Layer policy tests |
| `test_quant_strategy.py` | Quantization strategy tests |
| `test_quantization_benchmark_utils.py` | Benchmark utility tests |
| `test_crossover_benchmark_utils.py` | Crossover benchmark utility tests |
| `test_experiment_entrypoints.py` | Archive experiment existence tests |
| `test_triton_profile_metrics.py` | Triton profile metrics tests |

## 5. Keep: Tests

| Test | Purpose |
|---|---|
| `test_quantization.py` | Core quantization correctness |
| `test_scoring.py` | Core scoring correctness |
| `test_score_bounds.py` | Score bounds validation |
| `test_group_quantization.py` | Group quantization tests |
| `test_groupwise_coordinate.py` | Groupwise coordinate bounds |
| `test_aglr_antidiagonal.py` | AGLR antidiagonal scoring (update: remove landmark mask test) |
| `test_aglr_indexer.py` | AGLR indexer (update: remove landmark/mass-budget tests) |
| `test_aglr_certimask.py` | Reference-first AGLR CertiMask |
| `test_topk_certificate.py` | Top-k certificate reference |
| `test_vectorized_topk.py` | Vectorized top-k |
| `test_triton_aglr_certimask.py` | Triton scoring validation |
| `test_triton_topk_certificate.py` | Triton certificate validation |
| `test_benchmark_semantics.py` | Benchmark semantics definitions |
| `test_api_path_annotations.py` | API path docstring warnings |

## 6. Delete: Archive Experiments

All files in `experiments/archive/` — deleted, not archived.

## 7. Keep: Active Experiments

| Script | Purpose |
|---|---|
| `experiments/active/benchmark_aglr_crossover.py` | Long-context crossover analysis |
| `experiments/active/benchmark_aglr_triton.py` | PyTorch vs Triton scoring |
| `experiments/active/benchmark_aglr_triton_profile.py` | Triton latency decomposition |
| `experiments/active/benchmark_topk_mask.py` | Loop vs vectorized top-k |
| `experiments/active/benchmark_aglr_quantization_and_baseline.py` | Quantization + dense baseline |

## 8. Rewrite: `__init__.py` Public API

Only export current-path symbols. Remove all threshold/diagnostic/historical exports.

## 9. Modify: `aglr_indexer.py`

- Keep: `compute_antidiagonal_block_scores`, `_generate_sample_positions`, `combine_aglr_scores`
- Keep: `aglr_local_plus_landmark_mask` (used by `triton_aglr_ops._compute_reference_mask` and active benchmarks)
- Keep: `AGLRMaskResult`, `BlockLandmarks` (used by `aglr_local_plus_landmark_mask`)
- Remove: `select_block_landmarks`, `compute_landmark_block_scores`, `aglr_adaptive_mass_budget_mask`

## 10. Modify: `masking.py`

- Keep: `make_block_causal_valid_mask`, `CertiMaskDecision`, `CertiMaskResult`, `_validate_mask_inputs`
- Remove: `certified_threshold_mask`, `naive_quantized_mask`, `reference_mask`, `thresholds_for_target_sparsity`

## 11. Modify: `triton_aglr_ops.py`

- Add: `triton_aglr_certimask_logsumexp_g4_optimized` using fused Triton certificate
- Keep: `triton_aglr_certimask_logsumexp_g4` as reference (uses PyTorch cert internally)
