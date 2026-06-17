# Phase 7E: Full 24-Layer AGLR-C v1 Policy Scan

**Model:** Qwen/Qwen2.5-0.5B-Instruct
**Context length:** 1024
**Layers scanned:** 24 (0..23)

## Decision Distribution

| Decision | Count | Layers |
|---|---|---|
| go_strict | 15 | [3, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] |
| go_practical | 5 | [2, 4, 5, 6, 10] |
| conditional_relaxed | 3 | [0, 1, 23] |
| quality_only_high_work | 1 | [22] |
| fallback_quality | 0 | [] |

## Aggregate Metrics

- Mean selected work fraction: 0.3765
- Mean selected kept mass: 0.9113
- Mean selected cosine: 0.9847
- Mean selected L2: 0.1055
- Mean improvement over mean-pooled (mass): +0.0974
- Mean improvement over mean-pooled (L2): -0.2438

## CertiMask Readiness: ready_for_topk_certificate_design

## Per-Layer Policy

| Layer | Decision | BS | Sparsity | LB | Agg | Work | Mass | Cosine | L2 | Fallback |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | conditional_relaxed | 8 | 0.5 | 0 | logsumexp | 0.5039 | 0.8828 | 0.9910 | 0.0617 |  |
| 1 | conditional_relaxed | 8 | 0.5 | 0 | logsumexp | 0.5039 | 0.8646 | 0.9934 | 0.0756 |  |
| 2 | go_practical | 8 | 0.5 | 2 | topk_mean | 0.5040 | 0.9038 | 0.9907 | 0.1040 |  |
| 3 | go_strict | 8 | 0.75 | 0 | logsumexp | 0.2558 | 0.9032 | 0.9715 | 0.1649 |  |
| 4 | go_practical | 8 | 0.5 | 0 | logsumexp | 0.5039 | 0.9069 | 0.9887 | 0.0906 |  |
| 5 | go_practical | 8 | 0.5 | 0 | logsumexp | 0.5039 | 0.9114 | 0.9884 | 0.1120 |  |
| 6 | go_practical | 8 | 0.5 | 0 | logsumexp | 0.5039 | 0.9180 | 0.9867 | 0.1025 |  |
| 7 | go_strict | 8 | 0.75 | 0 | logsumexp | 0.2558 | 0.9485 | 0.9808 | 0.0847 |  |
| 8 | go_strict | 8 | 0.75 | 0 | logsumexp | 0.2558 | 0.9033 | 0.9813 | 0.0944 |  |
| 9 | go_strict | 8 | 0.75 | 0 | logsumexp | 0.2558 | 0.9106 | 0.9733 | 0.1217 |  |
| 10 | go_practical | 8 | 0.5 | 0 | logsumexp | 0.5039 | 0.9254 | 0.9917 | 0.0829 |  |
| 11 | go_strict | 8 | 0.7 | 0 | logsumexp | 0.3074 | 0.9177 | 0.9838 | 0.1192 |  |
| 12 | go_strict | 8 | 0.65 | 0 | logsumexp | 0.3574 | 0.9135 | 0.9900 | 0.0867 |  |
| 13 | go_strict | 8 | 0.75 | 0 | logsumexp | 0.2558 | 0.9118 | 0.9825 | 0.1031 |  |
| 14 | go_strict | 8 | 0.7 | 0 | logsumexp | 0.3074 | 0.9035 | 0.9872 | 0.1139 |  |
| 15 | go_strict | 8 | 0.7 | 0 | logsumexp | 0.3074 | 0.9133 | 0.9852 | 0.1048 |  |
| 16 | go_strict | 8 | 0.75 | 0 | logsumexp | 0.2558 | 0.9761 | 0.9821 | 0.0771 |  |
| 17 | go_strict | 8 | 0.7 | 0 | logsumexp | 0.3074 | 0.9172 | 0.9850 | 0.1028 |  |
| 18 | go_strict | 8 | 0.75 | 0 | logsumexp | 0.2558 | 0.9085 | 0.9635 | 0.1962 |  |
| 19 | go_strict | 8 | 0.75 | 0 | logsumexp | 0.2558 | 0.9246 | 0.9850 | 0.1003 |  |
| 20 | go_strict | 8 | 0.65 | 0 | logsumexp | 0.3574 | 0.9090 | 0.9917 | 0.0906 |  |
| 21 | go_strict | 8 | 0.7 | 0 | logsumexp | 0.3074 | 0.9062 | 0.9835 | 0.1117 |  |
| 22 | quality_only_high_work | 8 | 0.3 | 0 | logsumexp | 0.7070 | 0.9343 | 0.9924 | 0.0891 |  |
| 23 | conditional_relaxed | 8 | 0.5 | 0 | logsumexp | 0.5039 | 0.8575 | 0.9842 | 0.1413 |  |
