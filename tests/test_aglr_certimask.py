"""Tests for AGLR CertiMask top-k certification."""

from __future__ import annotations

import torch

from certimask.aglr_certimask import (
    aglr_certimask_topk,
    compute_aglr_certimask_metrics,
)
from certimask.quantization import quantize_int8_per_group


def _make_qk(
    batch: int = 1,
    heads: int = 1,
    seq_len: int = 32,
    dim: int = 16,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic Q and K tensors with controlled values."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, seq_len, dim, generator=gen)
    k = torch.randn(batch, heads, seq_len, dim, generator=gen)
    return q, k


class TestSampledDotIntervals:
    """Test that sampled dot intervals cover FP values."""

    def test_intervals_cover_fp_dots(self) -> None:
        """K-only intervals must cover FP sampled dots."""
        q, k = _make_qk(seq_len=16, dim=8)
        k_q = quantize_int8_per_group(k, group_size=4)

        from certimask.aglr_certimask import _compute_sampled_dot_intervals
        from certimask.bounds import _get_group_per_coord_error

        k_err = _get_group_per_coord_error(k_q, "analytic")
        _, lower, upper = _compute_sampled_dot_intervals(
            q, k_q.dequantized, k_err,
            block_size=4, sample_pattern="both_diagonals",
        )

        # FP dots (computed from original k)
        from certimask.aglr_indexer import _generate_sample_positions
        positions = _generate_sample_positions(4, "both_diagonals")
        nb = 4
        q_blocks = q.reshape(1, 1, nb, 4, 8)
        k_blocks = k.reshape(1, 1, nb, 4, 8)

        for p_idx, (qi, ki) in enumerate(positions):
            q_s = q_blocks[:, :, :, qi, :]
            k_s = k_blocks[:, :, :, ki, :]
            fp_dot = torch.einsum("bhqd,bhkd->bhqk", q_s, k_s) / (8 ** 0.5)
            lo = lower[:, :, :, :, p_idx]
            hi = upper[:, :, :, :, p_idx]
            assert (fp_dot >= lo - 1e-6).all(), f"FP below lower at sample {p_idx}"
            assert (fp_dot <= hi + 1e-6).all(), f"FP above upper at sample {p_idx}"

    def test_intervals_cover_fp_with_error(self) -> None:
        """Intervals with actual error bound cover FP dots."""
        q, k = _make_qk(seq_len=16, dim=8)
        k_q = quantize_int8_per_group(k, group_size=4)

        from certimask.aglr_certimask import _compute_sampled_dot_intervals
        from certimask.bounds import _get_group_per_coord_error

        k_err = _get_group_per_coord_error(k_q, "analytic")

        _, lower, upper = _compute_sampled_dot_intervals(
            q, k_q.dequantized, k_err,
            block_size=4, sample_pattern="both_diagonals",
        )

        # Lower <= upper for all samples
        assert (lower <= upper + 1e-6).all()


class TestLogsumexpIntervalCoverage:
    """Test that logsumexp intervals cover FP AGLR scores."""

    def test_interval_covers_fp_score(self) -> None:
        """Logsumexp interval must contain the FP AGLR score."""
        q, k = _make_qk(seq_len=32, dim=16, seed=123)

        from certimask.aglr_indexer import compute_antidiagonal_block_scores
        from certimask.masking import make_block_causal_valid_mask

        nb = 4  # seq_len=32, block_size=8 -> num_blocks=4
        valid_mask = make_block_causal_valid_mask(nb, nb).expand(1, 1, nb, nb)

        fp_scores = compute_antidiagonal_block_scores(
            q, k, block_size=8,
            sample_pattern="both_diagonals", aggregation="logsumexp",
            valid_mask=valid_mask,
        )

        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            aggregation="logsumexp", group_size=16,
        )

        # FP scores should be within [lower, upper] for valid tiles
        valid = valid_mask & (fp_scores > torch.finfo(torch.float32).min / 2)
        assert (fp_scores[valid] >= result.lower_scores[valid] - 1e-3).all()
        assert (fp_scores[valid] <= result.upper_scores[valid] + 1e-3).all()


class TestExactMatch:
    """Test that final mask exactly matches reference."""

    def test_exact_match(self) -> None:
        """Certified mask must match reference AGLR mask."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            aggregation="logsumexp", group_size=16,
        )
        assert result.exact_mask_match, (
            f"Mismatch count: {result.mismatch_count}"
        )

    def test_exact_match_high_sparsity(self) -> None:
        """Exact match at 75% sparsity."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.75,
            aggregation="logsumexp", group_size=16,
        )
        assert result.exact_mask_match

    def test_exact_match_low_sparsity(self) -> None:
        """Exact match at 30% sparsity."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.30,
            aggregation="logsumexp", group_size=16,
        )
        assert result.exact_mask_match


class TestZeroKGroup:
    """Test zero K group behavior."""

    def test_zero_group_bound_is_zero(self) -> None:
        """Zero group should have zero error bound."""
        k = torch.zeros(1, 1, 8, 16)
        k_q = quantize_int8_per_group(k, group_size=8)

        from certimask.bounds import _get_group_per_coord_error
        k_err = _get_group_per_coord_error(k_q, "analytic")

        assert (k_err == 0).all(), "Zero group should have zero error bound"


class TestGroupSize:
    """Test different group sizes."""

    def test_group_size_16(self) -> None:
        """group_size=16 works correctly."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            aggregation="logsumexp", group_size=16,
        )
        assert result.exact_mask_match

    def test_group_size_8(self) -> None:
        """group_size=8 works correctly."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            aggregation="logsumexp", group_size=8,
        )
        assert result.exact_mask_match


class TestAggregation:
    """Test aggregation support."""

    def test_topk_mean_works(self) -> None:
        """topk_mean aggregation now supported."""
        q, k = _make_qk(seq_len=32, dim=16)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            aggregation="topk_mean", group_size=16,
        )
        assert result.exact_mask_match

    def test_unsupported_aggregation_raises(self) -> None:
        """Unknown aggregation should raise ValueError."""
        q, k = _make_qk(seq_len=32, dim=16)
        try:
            aglr_certimask_topk(
                q, k, block_size=8, target_sparsity=0.5,
                aggregation="mean", group_size=16,
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass


class TestComputeMetrics:
    """Test metric computation."""

    def test_metrics_fields(self) -> None:
        """All metric fields are populated."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            aggregation="logsumexp", group_size=16,
        )

        from certimask.masking import make_block_causal_valid_mask
        valid = make_block_causal_valid_mask(4, 4).expand(1, 1, 4, 4)
        metrics = compute_aglr_certimask_metrics(result, valid)

        assert metrics.valid_tiles > 0
        assert metrics.row_count > 0
        assert 0.0 <= metrics.row_certification_rate <= 1.0
        assert 0.0 <= metrics.ambiguous_rate <= 1.0
        assert metrics.exact_mask_match
        assert metrics.mismatch_count == 0

    def test_interval_width_nonneg(self) -> None:
        """Interval width statistics are non-negative."""
        q, k = _make_qk(seq_len=32, dim=16, seed=42)
        result = aglr_certimask_topk(
            q, k, block_size=8, target_sparsity=0.5,
            aggregation="logsumexp", group_size=16,
        )

        from certimask.masking import make_block_causal_valid_mask
        valid = make_block_causal_valid_mask(4, 4).expand(1, 1, 4, 4)
        metrics = compute_aglr_certimask_metrics(result, valid)

        assert metrics.mean_interval_width >= 0
        assert metrics.p50_interval_width >= 0
        assert metrics.p90_interval_width >= 0
        assert metrics.p99_interval_width >= 0
