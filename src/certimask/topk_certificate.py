"""Top-k partition certificate for block-sparse attention.

Certifies that a low-bit (INT8 per-group K-only) scoring produces the same
top-k block selection as the FP32 reference, using score interval analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Decision codes for per-tile certification status.
DROP = 0
KEEP = 1
AMBIGUOUS = 2
INVALID = 3


@dataclass
class TopKCertificateResult:
    """Result of top-k partition certification.

    Attributes:
        certified_mask: Final boolean block mask [B, H, Q_blk, K_blk].
            Always equals selected_reference_mask.
        decisions: Per-tile decision codes [B, H, Q_blk, K_blk].
            0=DROP, 1=KEEP, 2=AMBIGUOUS, 3=INVALID.
        selected_reference_mask: FP reference top-k boolean mask.
        ambiguous: Boolean mask of ambiguous (boundary) tiles.
        fallback_mask: Boolean mask of tiles that required FP fallback.
        row_certified: Per-row certification status [B, H, Q_blk].
        lower_scores: Lower bound block scores [B, H, Q_blk, K_blk].
        upper_scores: Upper bound block scores [B, H, Q_blk, K_blk].
        quantized_scores: Quantized midpoint block scores.
        margin_to_boundary: Per-tile margin to partition boundary [B, H, Q_blk, K_blk].
            For selected tiles: S_b - max_{r in R} S_r.
            For rejected tiles: min_{t in T} S_t - S_b.
            Only valid for tiles where margin >= 0.
    """

    certified_mask: torch.Tensor
    decisions: torch.Tensor
    selected_reference_mask: torch.Tensor
    ambiguous: torch.Tensor
    fallback_mask: torch.Tensor
    row_certified: torch.Tensor
    lower_scores: torch.Tensor
    upper_scores: torch.Tensor
    quantized_scores: torch.Tensor
    margin_to_boundary: torch.Tensor


def logsumexp_interval(
    lower_samples: torch.Tensor,
    upper_samples: torch.Tensor,
    *,
    dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute logsumexp score interval from per-sample dot intervals.

    Given z_m in [L_m, U_m] for each sample m, the logsumexp aggregate
    S = logsumexp(z_m) satisfies:
        S in [logsumexp(L_m), logsumexp(U_m)]

    This follows from the monotonicity of logsumexp in each argument.

    Args:
        lower_samples: Lower bound per sample, shape [..., P].
        upper_samples: Upper bound per sample, shape [..., P].
        dim: Reduction dimension.

    Returns:
        (lower, upper) logsumexp interval tensors.

    Raises:
        ValueError: If shapes don't match or dim is invalid.
    """
    lower_samples = lower_samples.to(torch.float32)
    upper_samples = upper_samples.to(torch.float32)

    if lower_samples.shape != upper_samples.shape:
        raise ValueError(
            f"Shape mismatch: lower {lower_samples.shape} vs upper "
            f"{upper_samples.shape}"
        )

    lower = torch.logsumexp(lower_samples, dim=dim)
    upper = torch.logsumexp(upper_samples, dim=dim)

    # Numerical guard: ensure lower <= upper
    upper = torch.maximum(lower, upper)

    return lower, upper


def certified_topk_mask(
    reference_scores: torch.Tensor,
    lower_scores: torch.Tensor,
    upper_scores: torch.Tensor,
    *,
    k_per_row: torch.Tensor,
    valid_mask: torch.Tensor,
    ambiguity_mode: str = "partition",
) -> TopKCertificateResult:
    """Certify top-k partition using score intervals.

    For each query row, the FP reference selects the top-k scoring blocks.
    This function checks whether the low-bit score intervals support the
    same partition, and identifies boundary (ambiguous) candidates when
    certification is not possible.

    Two ambiguity modes:
    - "threshold": A tile is ambiguous if its interval overlaps the k-th
      reference score threshold. Conservative.
    - "partition": A tile is ambiguous unless it can be certified by the
      partition-aware check: L_t > U_R^max for selected, U_r < L_T^min
      for rejected. Tighter than threshold mode.

    Args:
        reference_scores: FP32 block scores [B, H, Q_blk, K_blk].
        lower_scores: Lower bound of score intervals [B, H, Q_blk, K_blk].
        upper_scores: Upper bound of score intervals [B, H, Q_blk, K_blk].
        k_per_row: Number of blocks to keep per query row [B, H, Q_blk].
        valid_mask: Boolean causal valid mask [B, H, Q_blk, K_blk].
        ambiguity_mode: "threshold" or "partition".

    Returns:
        TopKCertificateResult with certification status and masks.

    Raises:
        ValueError: If shapes are inconsistent or ambiguity_mode is invalid.
    """
    if ambiguity_mode not in ("threshold", "partition"):
        raise ValueError(
            f"ambiguity_mode must be 'threshold' or 'partition', "
            f"got '{ambiguity_mode}'"
        )
    if reference_scores.shape != lower_scores.shape:
        raise ValueError("Shape mismatch between reference and lower scores")
    if reference_scores.shape != upper_scores.shape:
        raise ValueError("Shape mismatch between reference and upper scores")
    if reference_scores.shape != valid_mask.shape:
        raise ValueError("Shape mismatch between scores and valid_mask")

    batch, heads, q_blk, k_blk = reference_scores.shape
    device = reference_scores.device

    # Initialize outputs
    decisions = torch.full(
        (batch, heads, q_blk, k_blk), INVALID, dtype=torch.long, device=device,
    )
    selected_mask = torch.zeros(
        batch, heads, q_blk, k_blk, dtype=torch.bool, device=device,
    )
    ambiguous = torch.zeros(
        batch, heads, q_blk, k_blk, dtype=torch.bool, device=device,
    )
    fallback_mask = torch.zeros(
        batch, heads, q_blk, k_blk, dtype=torch.bool, device=device,
    )
    row_certified = torch.zeros(
        batch, heads, q_blk, dtype=torch.bool, device=device,
    )
    margin = torch.zeros(
        batch, heads, q_blk, k_blk, dtype=torch.float32, device=device,
    )

    for b in range(batch):
        for h in range(heads):
            for q in range(q_blk):
                k_keep = int(k_per_row[b, h, q].item())
                valid_k = valid_mask[b, h, q]  # [K_blk]
                n_valid = int(valid_k.sum().item())

                if n_valid == 0:
                    row_certified[b, h, q] = True
                    continue

                k_keep = min(k_keep, n_valid)

                # Step 1: FP reference top-k selection
                ref_row = reference_scores[b, h, q].clone()
                ref_row[~valid_k] = float("-inf")
                _, topk_idx = ref_row.topk(k_keep)
                selected_mask[b, h, q, topk_idx] = True

                # Build T and R masks
                t_mask = torch.zeros(k_blk, dtype=torch.bool, device=device)
                t_mask[topk_idx] = True
                t_mask = t_mask & valid_k
                r_mask = valid_k & ~t_mask

                # Compute margin_to_boundary for each valid candidate
                ref_valid = ref_row.clone()
                ref_valid[~valid_k] = float("-inf")

                if t_mask.any() and r_mask.any():
                    min_t_score = ref_valid[t_mask].min().item()
                    max_r_score = ref_valid[r_mask].max().item()
                    for idx in range(k_blk):
                        if not valid_k[idx]:
                            continue
                        s = ref_valid[idx].item()
                        if t_mask[idx]:
                            margin[b, h, q, idx] = s - max_r_score
                        else:
                            margin[b, h, q, idx] = min_t_score - s
                else:
                    # All selected or all rejected -> margin is +inf
                    for idx in range(k_blk):
                        if valid_k[idx]:
                            margin[b, h, q, idx] = float("inf")

                # Step 2: Certification check
                sel_lower = lower_scores[b, h, q, topk_idx]
                min_sel_lower = sel_lower.min().item()

                if r_mask.any():
                    unsel_upper = upper_scores[b, h, q, r_mask]
                    max_unsel_upper = unsel_upper.max().item()
                else:
                    max_unsel_upper = float("-inf")

                if min_sel_lower > max_unsel_upper:
                    # Fully certified
                    row_certified[b, h, q] = True
                    decisions[b, h, q, t_mask] = KEEP
                    decisions[b, h, q, r_mask] = DROP
                else:
                    # Not fully certified: apply ambiguity mode
                    if ambiguity_mode == "threshold":
                        # Phase 8A: threshold overlap mode
                        tau = ref_row.topk(k_keep)[0][-1].item()
                        lower_row = lower_scores[b, h, q]
                        upper_row = upper_scores[b, h, q]
                        is_boundary = (
                            valid_k
                            & (lower_row <= tau)
                            & (upper_row >= tau)
                        )
                        ambiguous[b, h, q, is_boundary] = True
                        fallback_mask[b, h, q, is_boundary] = True
                        definite_keep = valid_k & ~is_boundary & (lower_row > tau)
                        definite_drop = valid_k & ~is_boundary & (upper_row < tau)
                        decisions[b, h, q, definite_keep] = KEEP
                        decisions[b, h, q, definite_drop] = DROP
                        decisions[b, h, q, is_boundary] = AMBIGUOUS

                    else:
                        # Phase 8B: partition-aware mode (tighter)
                        # L_T^min = min lower of selected, U_R^max = max upper of rejected
                        lt_min = lower_scores[b, h, q, t_mask].min().item()
                        ur_max = (upper_scores[b, h, q, r_mask].max().item()
                                  if r_mask.any() else float("-inf"))

                        lower_row = lower_scores[b, h, q]
                        upper_row = upper_scores[b, h, q]

                        # Certified KEEP: selected t with L_t > U_R^max
                        cert_keep = t_mask & (lower_row > ur_max)
                        # Certified DROP: rejected r with U_r < L_T^min
                        cert_drop = r_mask & (upper_row < lt_min)

                        # Ambiguous: everything else that's valid and not certified
                        is_ambig = valid_k & ~cert_keep & ~cert_drop

                        decisions[b, h, q, cert_keep] = KEEP
                        decisions[b, h, q, cert_drop] = DROP
                        decisions[b, h, q, is_ambig] = AMBIGUOUS
                        ambiguous[b, h, q, is_ambig] = True
                        fallback_mask[b, h, q, is_ambig] = True

    # Final certified mask: boundary candidates use FP reference
    certified_mask = selected_mask.clone()

    return TopKCertificateResult(
        certified_mask=certified_mask,
        decisions=decisions,
        selected_reference_mask=selected_mask,
        ambiguous=ambiguous,
        fallback_mask=fallback_mask,
        row_certified=row_certified,
        lower_scores=lower_scores,
        upper_scores=upper_scores,
        quantized_scores=(lower_scores + upper_scores) / 2.0,
        margin_to_boundary=margin,
    )
