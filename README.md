# CertiMask

Certified masking for attention with INT8 quantization.

## Current Phase

Phase 9A: Triton prototype for AGLR-C CertiMask sampled scoring.

## Installation

```bash
# Basic (torch + pytest + ruff + mypy)
pip install -e ".[dev]"

# With Hugging Face support
pip install -e ".[dev,hf]"

# With Triton kernel support (requires CUDA GPU)
pip install -e ".[dev,triton]"
```

## Testing

```bash
pytest -q
ruff check .
mypy src
```

## INT8 Quantization Formula

Given a floating-point vector $x \in \mathbb{R}^d$:

**Scale:**
$$\alpha_x = \frac{\max_i |x_i|}{127}$$

**Quantized integer:**
$$x_q = \text{clip}(\text{round}(x / \alpha_x), -127, 127)$$

**Dequantized result:**
$$\tilde{x} = \alpha_x \cdot x_q$$

**Actual L2 error:**
$$\|x - \tilde{x}\|_2$$

**Analytic L2 bound:**
$$\sqrt{d} \cdot \frac{\alpha_x}{2}$$

## Block Score Computation

Given query blocks $Q \in \mathbb{R}^{B \times H \times Q \times d}$ and key blocks $K \in \mathbb{R}^{B \times H \times K \times d}$:

**Reference scores (FP32):**
$$s_{ab} = \frac{q_a^\top k_b}{\sqrt{d}}$$

**INT8 quantized scores:**
1. Quantize $q_a$ and $k_b$ per-vector to INT8
2. Compute integer dot product: $z_{ab} = q_{a,\text{int8}}^\top k_{b,\text{int8}}$
3. Dequantize: $\tilde{s}_{ab} = \frac{z_{ab} \cdot \alpha_a^q \cdot \alpha_b^k}{\sqrt{d}}$

## Score Error Bounds

For $q_a = \tilde{q}_a + \Delta q_a$ and $k_b = \tilde{k}_b + \Delta k_b$, the error bound is:

$$E_{ab} = \|\tilde{q}_a\|_2 \epsilon_b^k + \|\tilde{k}_b\|_2 \epsilon_a^q + \epsilon_a^q \epsilon_b^k$$

**Certificate types:**
- **actual**: Uses actual quantization error (oracle, for validation only)
- **analytic**: Uses analytic upper bound $\sqrt{d} \cdot \alpha / 2$ (deployable certificate)

## Threshold Masking

**Reference mask:** $M_{\text{ref}} = (s > \tau)$

**CertiMask three-way classification:**
- $L > \tau \Rightarrow \text{CERTAIN KEEP}$
- $U < \tau \Rightarrow \text{CERTAIN DROP}$
- $L \leq \tau \leq U \Rightarrow \text{AMBIGUOUS}$ (uses reference score fallback)

## Phase 4: Real Model Integration

### Extraction

Post-RoPE Q/K is extracted from Qwen2/Qwen2.5 models using forward hooks:
1. Hook captures hidden_states at the target decoder layer
2. Manually runs q_proj, k_proj
3. Applies RoPE using the model's own rotary embedding

### GQA Head Mapping

For grouped-query attention ($H_q > H_{kv}$):
- `expand_kv_heads` repeats each KV head `group_size = H_q // H_kv` times
- Maps: query heads 0..group_size-1 → kv head 0, etc.

### Block Summary

Current implementation uses simple mean-pooled block indexing:

> 当前使用简单 mean-pooled block indexer，只用于验证真实模型 Q/K 上的 CertiMask 数值可行性。

This is NOT equivalent to learned block-flash attention (BFLA) indexers.

### Threshold Strategy

> 目标稀疏率阈值由 reference score 生成，仅用于受控评测；当前尚不是可部署的在线阈值策略。

### Real-Model Experiment

```bash
python experiments/run_hf_threshold.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --layer-index 0 \
  --context-length 512 \
  --block-size 32 \
  --target-sparsities 0.70 0.80 0.85 0.90 \
  --certificate-types actual analytic \
  --device cpu \
  --dtype float32 \
  --output-dir outputs/phase4_qwen
```

### Certificate Tightness (rho)

$\rho = |s - \tilde{s}| / E$

- $\rho$ close to 1: certificate is tight
- $\rho$ much less than 1: certificate is conservative/loose

The primary metric for certificate utility is **refinement rate**, not rho alone.

### FP16 Fallback

> 已验证算法级边界回退逻辑；尚未实现只 gather ambiguous tile 的真实 FP16 refinement kernel。

## Phase 9A: Triton Prototype

Triton v0 accelerates the AGLR-C sampled scoring and interval computation path.
It does **not** implement sparse attention, and does **not** claim end-to-end speedup.

**Supported configuration:**
- `block_size = 8`, `group_size = 4`, `D = 64`
- `sample_pattern = both_diagonals` (16 samples per tile)
- `aggregation = logsumexp`
- K-only per-group INT8 quantization
- Q in FP16 or FP32

**Not supported in Triton v0:**
- `topk_mean` aggregation (falls back to PyTorch)
- `block_size != 8`, `group_size != 4`, `D != 64`
- Sparse attention kernel
- End-to-end latency benchmarking

### Quick Start (requires CUDA + triton)

```bash
# Run tests (skips cleanly on CPU)
pytest tests/test_triton_aglr_certimask.py -v

# Synthetic benchmark
python experiments/benchmark_aglr_triton.py \
  --batch-size 1 --num-heads 14 --seq-len 1024 \
  --head-dim 64 --dtype float16 --device cuda \
  --warmup 20 --iters 100 \
  --output-dir outputs/phase9a_triton_aglr_benchmark

# Real Qwen smoke test
python experiments/benchmark_aglr_triton.py \
  --use-real-qwen --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --layers 8 12 16 20 --context-length 1024 \
  --dtype float16 --device cuda \
  --output-dir outputs/phase9a_triton_aglr_qwen_smoke
```

## Phase 9B: Triton CertiMask Profiling and Fused Certificate

Phase 9B profiles the Triton CertiMask full wrapper, identifies bottlenecks,
and implements a fused Triton partition certificate kernel.

### Key Findings

**Latency decomposition (B=1, H=14, L=1024, D=64, fp16, RTX 5090):**

| Stage | Median (ms) | Fraction |
|---|---|---|
| key_quantization | 4.92 | 0.1% |
| triton_score_interval_kernel | 0.89 | 0.0% |
| reference_fp32_aglr_score | 0.32 | 0.0% |
| topk_reference_mask | 322.79 | 4.5% |
| **partition_certificate_pytorch** | **6776.28** | **95.5%** |
| partition_certificate_fused_triton | 0.09 | 0.0% |
| fallback_resolution | 0.14 | 0.0% |

**The PyTorch partition certificate was the bottleneck at 95.5% of total time.**

**Fused Triton certificate:**
- Speedup: **73,527x** (6776 ms → 0.09 ms)
- Decisions exact match: **True**
- Ambiguous mask exact match: **True**

**With fused certificate, optimized total: 329 ms** (down from 7096 ms).
Remaining bottleneck: `topk_reference_mask` at 323 ms (98% of optimized total).

### Real-Qwen Fallback Metrics

| Layer | Fallback Rate | Row Cert Rate | Cert Keep | Cert Drop |
|---|---|---|---|---|
| 8 | 0.2621 | 0.1205 | 0.5122 | 0.4878 |
| 12 | 0.0908 | 0.1635 | 0.5060 | 0.4940 |
| 16 | 0.1009 | 0.1680 | 0.5028 | 0.4972 |
| 20 | 0.1420 | 0.1161 | 0.5055 | 0.4945 |
| **Mean** | **0.1214** | **0.1420** | **0.5057** | **0.4943** |

### Quick Start

```bash
# Run tests
pytest tests/test_triton_topk_certificate.py tests/test_triton_profile_metrics.py -v

# Latency profiling
python experiments/benchmark_aglr_triton_profile.py \
  --batch-size 1 --num-heads 14 --seq-len 1024 \
  --head-dim 64 --dtype float16 --device cuda \
  --warmup 5 --iters 10 \
  --output-dir outputs/phase9b_triton_profile

# Benchmark with fallback metrics
python experiments/benchmark_aglr_triton.py \
  --use-real-qwen --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --layers 8 12 16 20 --context-length 1024 \
  --dtype float16 --device cuda \
  --output-dir outputs/phase9b_triton_aglr_qwen_smoke
```

### Important Notes

- This is a **microbenchmark** for AGLR-C CertiMask scoring + interval + certificate
- It does **not** include sparse attention kernel speedup
- It does **not** claim end-to-end prefill speedup
- Fallback metrics come from the partition certificate, not from score equality
- The fused Triton certificate produces decisions and ambiguous masks identical to PyTorch

## Not Yet Implemented

- Sparse attention kernel
- INT4 quantization
- Top-K / Top-p sampling
- Softmax mass certificate
- Long-context performance benchmarks
- End-to-end acceleration claims
- Triton v1: topk_mean aggregation, variable block/group sizes
