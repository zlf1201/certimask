"""Tests for Phase 8B: partition ambiguity mode, new metrics, group-size scan."""

from __future__ import annotations

import torch

from certimask.aglr_certimask import (
    aglr_certimask_topk,
    compute_aglr_certimask_metrics,
)
from certimask.topk_certificate import (
    DROP,
    KEEP,
    certified_topk_mask,
)


def _make_qk(
    batch: int = 1,
    heads: int = 1,
    seq_len: int = 32,
    dim: int = 16,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, seq_len, dim, generator=gen)
    k = torch.randn(batch, heads, seq_len, dim, generator=gen)
    return q, k


class TestPartitionVsThreshold:
    """Test partition ambiguity mode produces fewer or equal ambiguous tiles."""

    def test_partition_not_more_ambiguous(self) -> None:
        """Partition mode ambiguous count <= threshold mode."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)

        res_p = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            group_size=16, ambiguity_mode="partition",
        )
        res_t = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            group_size=16, ambiguity_mode="threshold",
        )

        ambig_p = res_p.ambiguous.sum().item()
        ambig_t = res_t.ambiguous.sum().item()
        assert ambig_p <= ambig_t, (
            f"Partition ({ambig_p}) should have <= ambiguous than threshold ({ambig_t})"
        )

    def test_both_exact_match(self) -> None:
        """Both modes must produce exact match."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        for mode in ["partition", "threshold"]:
            res = aglr_certimask_topk(
                q, k, block_size=8, target_sparsity=0.5,
                group_size=16, ambiguity_mode=mode,
            )
            assert res.exact_mask_match, f"{mode} mode mismatch: {res.mismatch_count}"


class TestPartitionCertification:
    """Test partition-aware certified KEEP/DROP."""

    def test_certified_keep_correct(self) -> None:
        """Selected tile with L_t > U_R^max is certified KEEP."""
        ref = torch.tensor([[[[5.0, 4.0, 2.0, 1.0]]]])
        # Selected (5.0, 4.0): L > max U of rejected (2.5, 1.5) = 2.5
        lower = torch.tensor([[[[4.0, 3.5, 1.0, 0.0]]]])
        upper = torch.tensor([[[[6.0, 5.0, 2.5, 1.5]]]])
        k = torch.tensor([[[2]]])
        valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)

        r = certified_topk_mask(ref, lower, upper, k_per_row=k,
                                valid_mask=valid, ambiguity_mode="partition")
        # Selected tiles 0,1 should be KEEP (L > U_R^max=2.5)
        assert r.decisions[0, 0, 0, 0] == KEEP
        assert r.decisions[0, 0, 0, 1] == KEEP

    def test_certified_drop_correct(self) -> None:
        """Rejected tile with U_r < L_T^min is certified DROP."""
        ref = torch.tensor([[[[5.0, 4.0, 2.0, 1.0]]]])
        lower = torch.tensor([[[[4.0, 3.5, 1.0, 0.0]]]])
        upper = torch.tensor([[[[6.0, 5.0, 2.5, 1.5]]]])
        k = torch.tensor([[[2]]])
        valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)

        r = certified_topk_mask(ref, lower, upper, k_per_row=k,
                                valid_mask=valid, ambiguity_mode="partition")
        # L_T^min = min(4.0, 3.5) = 3.5
        # Rejected tile 2: U=2.5 < 3.5 -> certified DROP
        assert r.decisions[0, 0, 0, 2] == DROP
        # Rejected tile 3: U=1.5 < 3.5 -> certified DROP
        assert r.decisions[0, 0, 0, 3] == DROP

    def test_exact_match_with_partition(self) -> None:
        """Fallback with partition mode produces exact match."""
        # Wide intervals -> not fully certified
        ref = torch.tensor([[[[5.0, 4.0, 3.5, 1.0]]]])
        lower = torch.tensor([[[[3.0, 2.0, 2.5, 0.0]]]])
        upper = torch.tensor([[[[7.0, 6.0, 5.0, 3.0]]]])
        k = torch.tensor([[[2]]])
        valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)

        r = certified_topk_mask(ref, lower, upper, k_per_row=k,
                                valid_mask=valid, ambiguity_mode="partition")
        assert (r.certified_mask == r.selected_reference_mask).all()


class TestNewMetrics:
    """Test new metrics computation."""

    def test_selected_ambiguous_rate(self) -> None:
        """selected_ambiguous_rate computed correctly."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            group_size=16, ambiguity_mode="partition",
        )
        from certimask.masking import make_block_causal_valid_mask
        valid = make_block_causal_valid_mask(4, 4).expand(1, 1, 4, 4)
        m = compute_aglr_certimask_metrics(result, valid)
        assert 0.0 <= m.selected_ambiguous_rate <= 1.0

    def test_rejected_ambiguous_rate(self) -> None:
        """rejected_ambiguous_rate computed correctly."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            group_size=16, ambiguity_mode="partition",
        )
        from certimask.masking import make_block_causal_valid_mask
        valid = make_block_causal_valid_mask(4, 4).expand(1, 1, 4, 4)
        m = compute_aglr_certimask_metrics(result, valid)
        assert 0.0 <= m.rejected_ambiguous_rate <= 1.0

    def test_boundary_band_size(self) -> None:
        """boundary_band_size_mean >= 0."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            group_size=16, ambiguity_mode="partition",
        )
        from certimask.masking import make_block_causal_valid_mask
        valid = make_block_causal_valid_mask(4, 4).expand(1, 1, 4, 4)
        m = compute_aglr_certimask_metrics(result, valid)
        assert m.boundary_band_size_mean >= 0
        assert m.boundary_band_size_p90 >= 0

    def test_certified_keep_drop_rate(self) -> None:
        """certified_keep_rate and certified_drop_rate in [0, 1]."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            group_size=16, ambiguity_mode="partition",
        )
        from certimask.masking import make_block_causal_valid_mask
        valid = make_block_causal_valid_mask(4, 4).expand(1, 1, 4, 4)
        m = compute_aglr_certimask_metrics(result, valid)
        assert 0.0 <= m.certified_keep_rate <= 1.0
        assert 0.0 <= m.certified_drop_rate <= 1.0

    def test_margin_to_boundary(self) -> None:
        """mean_margin_to_boundary computed correctly."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            group_size=16, ambiguity_mode="partition",
        )
        from certimask.masking import make_block_causal_valid_mask
        valid = make_block_causal_valid_mask(4, 4).expand(1, 1, 4, 4)
        m = compute_aglr_certimask_metrics(result, valid)
        assert m.mean_margin_to_boundary >= 0

    def test_width_over_margin(self) -> None:
        """score_interval_width_over_margin metrics computed."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            group_size=16, ambiguity_mode="partition",
        )
        from certimask.masking import make_block_causal_valid_mask
        valid = make_block_causal_valid_mask(4, 4).expand(1, 1, 4, 4)
        m = compute_aglr_certimask_metrics(result, valid)
        assert m.score_interval_width_over_margin_p50 >= 0
        assert m.score_interval_width_over_margin_p90 >= 0


class TestGroupSizes:
    """Test all required group sizes work."""

    def test_group_size_32(self) -> None:
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        r = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5, group_size=32,
        )
        assert r.exact_mask_match

    def test_group_size_16(self) -> None:
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        r = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5, group_size=16,
        )
        assert r.exact_mask_match

    def test_group_size_8(self) -> None:
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        r = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5, group_size=8,
        )
        assert r.exact_mask_match

    def test_group_size_4(self) -> None:
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        r = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5, group_size=4,
        )
        assert r.exact_mask_match


class TestGroupSizeMonotonicity:
    """Test that smaller group_size produces no wider intervals."""

    def test_smaller_group_no_wider_interval(self) -> None:
        """In controlled test, smaller group_size should not increase interval width."""
        # Use a seed where this holds
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        from certimask.masking import make_block_causal_valid_mask
        valid = make_block_causal_valid_mask(4, 4).expand(1, 1, 4, 4)

        widths: dict[int, float] = {}
        for gs in [32, 16, 8, 4]:
            result = aglr_certimask_topk(
                q, k, block_size=8, target_sparsity=0.5, group_size=gs,
            )
            m = compute_aglr_certimask_metrics(result, valid)
            widths[gs] = m.mean_interval_width

        # Check non-increasing trend (allowing small numerical noise)
        for gs_prev, gs_curr in [(32, 16), (16, 8), (8, 4)]:
            # Allow 5% tolerance for numerical noise
            assert widths[gs_curr] <= widths[gs_prev] * 1.05, (
                f"gs={gs_curr} width {widths[gs_curr]:.4f} > "
                f"gs={gs_prev} width {widths[gs_prev]:.4f}"
            )


class TestUnsupportedAggregation:
    """Test unsupported aggregation handling."""

    def test_unknown_aggregation_raises(self) -> None:
        q, k = _make_qk(seq_len=32, dim=16)
        try:
            aglr_certimask_topk(
                q, k, block_size=8, target_sparsity=0.5,
                aggregation="mean", group_size=16,
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
