# Pipeline Mode Definitions

## Certificate
All modes use fused Triton partition certificate
(`triton_certified_topk_mask_partition`) instead of the slow PyTorch
loop-based `certified_topk_mask`. The fused certificate is ~73,000x faster.

## Mode 1: Dense SDPA Baseline
- `torch.nn.functional.scaled_dot_product_attention`
- Causal=True
- Reports both FP16 and FP32 timings

## Mode 2A: Online Full with Quantization
- FP32 reference AGLR scores (online)
- K quantization (online)
- Vectorized top-k mask (online)
- Triton scoring kernel
- Fused Triton certificate
- **Online valid**: all computation happens inside timing loop

## Mode 2B: Online Full with Cached Quantization
- K quantization pre-computed outside timing loop
- FP32 reference AGLR scores (online)
- Vectorized top-k mask (online)
- Triton scoring kernel (with cached K)
- Fused Triton certificate
- **Online valid**: reference scores and top-k still computed per-call

## Mode 3: Optimistic Cached Indexer
- Pre-computed outside: FP32 reference scores, vectorized top-k mask, K quantization
- Inside timing: Triton scoring kernel + fused certificate + fallback
- **NOT online valid**: assumes reference scores and mask are cached
- Measures: lower-bound certificate overhead

## Ideal Sparse Attention Proxy
- `ideal_sparse_ms = dense_sdpa_fp16_ms * work_fraction`
- `work_fraction = 0.3765` (from Phase 8C mean attention tile work fraction)
- Assumes: sparse attention compute scales linearly with kept tiles
