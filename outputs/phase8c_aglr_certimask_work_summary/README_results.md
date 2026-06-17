# Phase 8C: AGLR-C + CertiMask Quality/Work Summary

## Configuration

- Group size: 4
- Ambiguity mode: partition
- c_lowbit values: [0.25, 0.5]

## Quality Summary

| Metric | Value |
|---|---|
| Mean kept mass | 0.9113 |
| Mean cosine | 0.9847 |
| Mean L2 | 0.1055 |
| Strict layers | 15 |
| Practical layers | 5 |
| Relaxed layers | 3 |
| Quality-only layers | 1 |
| Fallback quality layers | 0 |

## CertiMask Summary

| Metric | Value |
|---|---|
| All exact match | True |
| Total mismatch | 0 |
| Mean FP fallback | 0.1915 |
| Weighted FP fallback | 0.1915 |
| High fallback layers | [0, 1, 2] |
| Unsupported layers | [2] |

## Work Proxy

| Metric | Value |
|---|---|
| Mean attention work | 0.3765 |
| Mean indexer proxy (c=0.25) | 0.3936 |
| Mean indexer proxy (c=0.50) | 0.5957 |
| Attention savings proxy | 0.6235 |
| FP score savings proxy | 0.8085 |

## Readiness

**ready_for_triton_prototype**

Next step: Proceed to Triton kernel prototype for sparse attention + INT8 K scoring

## Per-Layer Runtime Paths

| Layer | Decision | Runtime Path | Attn Work | FP Fallback |
|---|---|---|---|---|
| 0 | conditional_relaxed | aglr_certimask_lowbit_high_fallback | 0.504 | 0.745 |
| 1 | conditional_relaxed | aglr_certimask_lowbit_high_fallback | 0.504 | 0.423 |
| 2 | go_practical | unsupported_aggregation_fp_reference | 0.504 | 1.000 |
| 3 | go_strict | aglr_certimask_lowbit | 0.256 | 0.090 |
| 4 | go_practical | aglr_certimask_lowbit | 0.504 | 0.168 |
| 5 | go_practical | aglr_certimask_lowbit | 0.504 | 0.173 |
| 6 | go_practical | aglr_certimask_lowbit | 0.504 | 0.142 |
| 7 | go_strict | aglr_certimask_lowbit | 0.256 | 0.069 |
| 8 | go_strict | aglr_certimask_lowbit | 0.256 | 0.215 |
| 9 | go_strict | aglr_certimask_lowbit | 0.256 | 0.093 |
| 10 | go_practical | aglr_certimask_lowbit | 0.504 | 0.157 |
| 11 | go_strict | aglr_certimask_lowbit | 0.307 | 0.094 |
| 12 | go_strict | aglr_certimask_lowbit | 0.357 | 0.092 |
| 13 | go_strict | aglr_certimask_lowbit | 0.256 | 0.082 |
| 14 | go_strict | aglr_certimask_lowbit | 0.307 | 0.102 |
| 15 | go_strict | aglr_certimask_lowbit | 0.307 | 0.092 |
| 16 | go_strict | aglr_certimask_lowbit | 0.256 | 0.077 |
| 17 | go_strict | aglr_certimask_lowbit | 0.307 | 0.116 |
| 18 | go_strict | aglr_certimask_lowbit | 0.256 | 0.092 |
| 19 | go_strict | aglr_certimask_lowbit | 0.256 | 0.087 |
| 20 | go_strict | aglr_certimask_lowbit | 0.357 | 0.134 |
| 21 | go_strict | aglr_certimask_lowbit | 0.307 | 0.121 |
| 22 | quality_only_high_work | aglr_certimask_lowbit_quality_only_high_attention_work | 0.707 | 0.108 |
| 23 | conditional_relaxed | aglr_certimask_lowbit | 0.504 | 0.125 |
