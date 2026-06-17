# Phase 8A: CertiMask Top-k Certificate for AGLR-C v1

**Model:** Qwen/Qwen2.5-0.5B-Instruct
**Context length:** 1024
**Group size:** 16
**Block size:** 8

## Results by Layer

| Layer | Sparsity | Exact | Row Cert | Ambiguous | Fallback | Interval Width |
|---|---|---|---|---|---|---|
| 3 | 0.75 | True | 0.1205 | 0.0884 | 0.0884 | 0.562071 |
| 8 | 0.75 | True | 0.1060 | 0.3856 | 0.3856 | 4.390377 |
| 12 | 0.65 | True | 0.1122 | 0.0841 | 0.0841 | 0.409376 |
| 13 | 0.75 | True | 0.1585 | 0.0671 | 0.0671 | 0.443521 |
| 16 | 0.75 | True | 0.1691 | 0.0601 | 0.0601 | 0.618675 |
| 20 | 0.65 | True | 0.0971 | 0.1182 | 0.1182 | 0.434272 |

## All Exact Match: True
## Total Mismatch: 0
## Mean Row Certification Rate: 0.1272
## Mean Ambiguous Rate: 0.1339
## Mean Fallback Rate: 0.1339

## Recommended Next Step: Proceed to Phase 8B: full 24-layer scan, group-size scan, or boundary optimization
