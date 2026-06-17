# Phase 8B: Group-Size Scan and Boundary Fallback Optimization

**Model:** Qwen/Qwen2.5-0.5B-Instruct
**Best group_size:** 4
**Best ambiguity_mode:** partition

## Group-Size Scan (partition mode, gs=16)

| Layer | gs=32 | gs=16 | gs=8 | gs=4 |
|---|---|---|---|---|
| 3 | 0.2356 | 0.1531 | 0.1110 | 0.0902 |
| 8 | 0.5542 | 0.5266 | 0.4428 | 0.2150 |
| 12 | 0.1710 | 0.1453 | 0.1263 | 0.0918 |
| 13 | 0.1267 | 0.1124 | 0.0957 | 0.0818 |
| 16 | 0.1156 | 0.1007 | 0.0892 | 0.0770 |
| 20 | 0.2713 | 0.2150 | 0.1755 | 0.1337 |

## Layer 8 Primary Cause: large_k_scales

## Full 24-Layer Results

- All exact match: True
- Total mismatch: 0
- Mean fallback rate: 0.1563
- Mean row certification rate: 0.1522

## Recommended Next Step: Phase 8C: full AGLR-C + CertiMask quality/work summary with per-layer certified vs fallback breakdown
