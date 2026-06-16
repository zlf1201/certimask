# CertiMask

Certified masking for attention with INT8 quantization.

## Current Phase

Phase 4: Real Hugging Face model integration with Qwen2/Qwen2.5 single-layer Q/K analysis.

## Installation

```bash
# Basic (torch + pytest + ruff + mypy)
pip install -e ".[dev]"

# With Hugging Face support
pip install -e ".[dev,hf]"
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

## Not Yet Implemented

- Triton/CUDA kernels
- INT4 quantization
- Top-K / Top-p sampling
- Softmax mass certificate
- Sparse attention kernel
- All-layer experiments
- Long-context performance benchmarks
- End-to-end acceleration claims
