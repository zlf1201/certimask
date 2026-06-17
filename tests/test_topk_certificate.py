"""Tests for top-k partition certificate."""

from __future__ import annotations

import torch

from certimask.topk_certificate import (
    DROP,
    INVALID,
    KEEP,
    TopKCertificateResult,
    certified_topk_mask,
    logsumexp_interval,
)


class TestLogsumexpInterval:
    """Test logsumexp interval computation."""

    def test_fp_inside_interval(self) -> None:
        """FP logsumexp must lie inside the interval."""
        lower = torch.tensor([[1.0, 2.0, 0.5]])
        upper = torch.tensor([[1.5, 2.5, 1.0]])
        lo, hi = logsumexp_interval(lower, upper)

        fp_val = torch.logsumexp(
            (lower + upper) / 2, dim=-1,
        )
        assert (fp_val >= lo - 1e-5).all()
        assert (fp_val <= hi + 1e-5).all()

    def test_lower_leq_upper(self) -> None:
        """Lower bound must be <= upper bound."""
        lower = torch.randn(3, 5, 7)
        upper = lower + torch.rand(3, 5, 7).abs()
        lo, hi = logsumexp_interval(lower, upper)
        assert (lo <= hi + 1e-6).all()

    def test_interval_width_nonneg(self) -> None:
        """Interval width must be non-negative."""
        lower = torch.randn(2, 4, 6)
        upper = lower + torch.rand(2, 4, 6).abs() * 2
        lo, hi = logsumexp_interval(lower, upper)
        assert ((hi - lo) >= -1e-6).all()

    def test_multiple_samples(self) -> None:
        """Handles multiple sample dimensions."""
        lower = torch.randn(1, 1, 4, 4, 16)
        upper = lower + torch.rand(1, 1, 4, 4, 16).abs()
        lo, hi = logsumexp_interval(lower, upper, dim=-1)
        assert lo.shape == (1, 1, 4, 4)
        assert (lo <= hi + 1e-6).all()

    def test_very_negative_scores(self) -> None:
        """Handles very negative (invalid) scores safely."""
        lower = torch.tensor([[-1e6, -1e6, 1.0]])
        upper = torch.tensor([[-1e6, -1e6, 1.5]])
        lo, hi = logsumexp_interval(lower, upper)
        # The -1e6 values should be negligible
        assert torch.isfinite(lo).all()
        assert torch.isfinite(hi).all()
        # Interval should be narrow (dominated by the 1.0-1.5 entry)
        assert (hi - lo).abs() < 1.0

    def test_single_sample(self) -> None:
        """Single sample: interval equals the sample interval."""
        lower = torch.tensor([[3.0]])
        upper = torch.tensor([[4.0]])
        lo, hi = logsumexp_interval(lower, upper)
        assert torch.allclose(lo, lower)
        assert torch.allclose(hi, upper)


class TestCertifiedTopKMask:
    """Test top-k partition certification."""

    def _make_result(
        self,
        reference_scores: torch.Tensor,
        lower_scores: torch.Tensor,
        upper_scores: torch.Tensor,
        k_per_row: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> TopKCertificateResult:
        return certified_topk_mask(
            reference_scores, lower_scores, upper_scores,
            k_per_row=k_per_row, valid_mask=valid_mask,
        )

    def test_certified_when_no_overlap(self) -> None:
        """Certified when selected lower > unselected upper."""
        # 4 blocks, select top 2
        ref = torch.tensor([[[[5.0, 4.0, 2.0, 1.0]]]])
        # Tight intervals: selected [5,6] [4,5], unselected [1,2] [0,1]
        lower = torch.tensor([[[[5.0, 4.0, 1.0, 0.0]]]])
        upper = torch.tensor([[[[6.0, 5.0, 2.0, 1.0]]]])
        k = torch.tensor([[[2]]])
        valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)

        r = self._make_result(ref, lower, upper, k, valid)
        assert r.row_certified[0, 0, 0]
        assert (r.decisions[0, 0, 0] == torch.tensor([KEEP, KEEP, DROP, DROP])).all()

    def test_uncertified_when_overlap(self) -> None:
        """Uncertified when intervals overlap the boundary."""
        # 4 blocks, select top 2
        ref = torch.tensor([[[[5.0, 4.0, 3.5, 1.0]]]])
        # Wide intervals: selected overlap with unselected
        lower = torch.tensor([[[[4.0, 3.0, 3.0, 0.0]]]])
        upper = torch.tensor([[[[6.0, 5.0, 4.5, 2.0]]]])
        k = torch.tensor([[[2]]])
        valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)

        r = self._make_result(ref, lower, upper, k, valid)
        assert not r.row_certified[0, 0, 0]
        # Blocks 1 and 2 have intervals overlapping tau=4.0
        # Block 1: [3,5] overlaps 4 -> boundary
        # Block 2: [3,4.5] overlaps 4 -> boundary
        assert r.ambiguous[0, 0, 0, 1]
        assert r.ambiguous[0, 0, 0, 2]

    def test_exact_match_after_fallback(self) -> None:
        """certified_mask must always equal selected_reference_mask."""
        ref = torch.tensor([[[[5.0, 4.0, 3.5, 1.0]]]])
        lower = torch.tensor([[[[4.0, 3.0, 3.0, 0.0]]]])
        upper = torch.tensor([[[[6.0, 5.0, 4.5, 2.0]]]])
        k = torch.tensor([[[2]]])
        valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)

        r = self._make_result(ref, lower, upper, k, valid)
        assert (r.certified_mask == r.selected_reference_mask).all()

    def test_invalid_tiles_remain_invalid(self) -> None:
        """Invalid tiles must have INVALID decision."""
        ref = torch.tensor([[[[5.0, 4.0, 2.0, 1.0]]]])
        lower = torch.tensor([[[[5.0, 4.0, 1.0, 0.0]]]])
        upper = torch.tensor([[[[6.0, 5.0, 2.0, 1.0]]]])
        k = torch.tensor([[[2]]])
        valid = torch.tensor([[[[True, True, True, False]]]])

        r = self._make_result(ref, lower, upper, k, valid)
        assert r.decisions[0, 0, 0, 3] == INVALID

    def test_k_equals_one(self) -> None:
        """Works when k=1."""
        ref = torch.tensor([[[[5.0, 4.0, 2.0, 1.0]]]])
        lower = torch.tensor([[[[5.0, 3.0, 1.0, 0.0]]]])
        upper = torch.tensor([[[[6.0, 4.0, 2.0, 1.0]]]])
        k = torch.tensor([[[1]]])
        valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)

        r = self._make_result(ref, lower, upper, k, valid)
        assert r.row_certified[0, 0, 0]
        assert r.certified_mask[0, 0, 0, 0]  # top-1 selected
        assert not r.certified_mask[0, 0, 0, 1]

    def test_k_equals_all_valid(self) -> None:
        """Works when k equals number of valid tiles."""
        ref = torch.tensor([[[[5.0, 4.0, 2.0, 1.0]]]])
        lower = torch.tensor([[[[5.0, 4.0, 2.0, 1.0]]]])
        upper = torch.tensor([[[[5.0, 4.0, 2.0, 1.0]]]])
        k = torch.tensor([[[4]]])
        valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)

        r = self._make_result(ref, lower, upper, k, valid)
        assert r.row_certified[0, 0, 0]
        assert r.certified_mask.all()  # all valid selected

    def test_ties_handled_deterministically(self) -> None:
        """Tied scores produce deterministic selection."""
        ref = torch.tensor([[[[5.0, 5.0, 5.0, 1.0]]]])
        lower = torch.tensor([[[[4.5, 4.5, 4.5, 0.0]]]])
        upper = torch.tensor([[[[5.5, 5.5, 5.5, 1.5]]]])
        k = torch.tensor([[[2]]])
        valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)

        r = self._make_result(ref, lower, upper, k, valid)
        # Should select exactly 2 blocks
        assert r.certified_mask[0, 0, 0].sum() == 2

    def test_row_certification_rate(self) -> None:
        """row_certification_rate computed correctly."""
        # 2 rows: one certified, one not
        ref = torch.tensor([[[[5.0, 4.0, 2.0, 1.0],
                               [5.0, 4.0, 3.5, 1.0]]]])
        lower = torch.tensor([[[[5.0, 4.0, 1.0, 0.0],
                                 [4.0, 3.0, 3.0, 0.0]]]])
        upper = torch.tensor([[[[6.0, 5.0, 2.0, 1.0],
                                 [6.0, 5.0, 4.5, 2.0]]]])
        k = torch.tensor([[[2, 2]]])
        valid = torch.ones(1, 1, 2, 4, dtype=torch.bool)

        r = self._make_result(ref, lower, upper, k, valid)
        rate = r.row_certified.float().mean().item()
        assert rate == 0.5

    def test_ambiguous_rate(self) -> None:
        """ambiguous_rate computed correctly."""
        # 4 blocks, select top 2. tau=4.0
        # Block 0: [5.0, 6.0] -> L=5.0 > tau -> KEEP (not boundary)
        # Block 1: [3.0, 5.0] -> overlaps tau -> boundary
        # Block 2: [3.0, 4.5] -> overlaps tau -> boundary
        # Block 3: [0.0, 1.0] -> U=1.0 < tau -> DROP (not boundary)
        ref = torch.tensor([[[[5.0, 4.0, 3.5, 1.0]]]])
        lower = torch.tensor([[[[5.0, 3.0, 3.0, 0.0]]]])
        upper = torch.tensor([[[[6.0, 5.0, 4.5, 1.0]]]])
        k = torch.tensor([[[2]]])
        valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)

        r = self._make_result(ref, lower, upper, k, valid)
        # 2 out of 4 valid tiles are ambiguous (blocks 1 and 2)
        amb_count = r.ambiguous.sum().item()
        assert amb_count == 2

    def test_empty_valid_row(self) -> None:
        """Row with no valid blocks is trivially certified."""
        ref = torch.zeros(1, 1, 1, 4)
        lower = torch.zeros(1, 1, 1, 4)
        upper = torch.zeros(1, 1, 1, 4)
        k = torch.tensor([[[2]]])
        valid = torch.zeros(1, 1, 1, 4, dtype=torch.bool)

        r = self._make_result(ref, lower, upper, k, valid)
        assert r.row_certified[0, 0, 0]
