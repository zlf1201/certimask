"""Tests for synthetic data generation."""

from __future__ import annotations

import pytest
import torch

from certimask.synthetic import generate_synthetic_summaries


class TestGenerateSyntheticSummaries:
    """Test synthetic data generation."""

    @pytest.mark.parametrize(
        "distribution",
        ["normal", "small_variance", "large_variance", "sparse",
         "outlier", "correlated", "near_zero"],
    )
    def test_shape_and_dtype(self, distribution: str) -> None:
        q, k = generate_synthetic_summaries(
            batch_size=2,
            num_heads=4,
            num_query_blocks=8,
            num_key_blocks=12,
            head_dim=16,
            distribution=distribution,
            seed=42,
            device="cpu",
            dtype=torch.float32,
        )
        assert q.shape == (2, 4, 8, 16)
        assert k.shape == (2, 4, 12, 16)
        assert q.dtype == torch.float32
        assert k.dtype == torch.float32

    def test_deterministic(self) -> None:
        kwargs = dict(
            batch_size=1,
            num_heads=2,
            num_query_blocks=4,
            num_key_blocks=4,
            head_dim=16,
            distribution="normal",
            seed=123,
            device="cpu",
            dtype=torch.float32,
        )
        q1, k1 = generate_synthetic_summaries(**kwargs)
        q2, k2 = generate_synthetic_summaries(**kwargs)
        assert torch.equal(q1, q2)
        assert torch.equal(k1, k2)

    def test_different_seeds_differ(self) -> None:
        kwargs = dict(
            batch_size=1,
            num_heads=2,
            num_query_blocks=4,
            num_key_blocks=4,
            head_dim=16,
            distribution="normal",
            device="cpu",
            dtype=torch.float32,
        )
        q1, k1 = generate_synthetic_summaries(seed=0, **kwargs)
        q2, k2 = generate_synthetic_summaries(seed=1, **kwargs)
        assert not torch.equal(q1, q2)

    def test_finite_values(self) -> None:
        for dist in [
            "normal", "small_variance", "large_variance", "sparse",
            "outlier", "correlated", "near_zero",
        ]:
            q, k = generate_synthetic_summaries(
                batch_size=1,
                num_heads=2,
                num_query_blocks=4,
                num_key_blocks=4,
                head_dim=16,
                distribution=dist,
                seed=42,
                device="cpu",
                dtype=torch.float32,
            )
            assert torch.isfinite(q).all(), f"Non-finite in {dist} query"
            assert torch.isfinite(k).all(), f"Non-finite in {dist} key"

    def test_sparse_mostly_zero(self) -> None:
        q, k = generate_synthetic_summaries(
            batch_size=1,
            num_heads=1,
            num_query_blocks=4,
            num_key_blocks=4,
            head_dim=64,
            distribution="sparse",
            seed=42,
            device="cpu",
            dtype=torch.float32,
        )
        q_zero_frac = (q == 0).float().mean().item()
        k_zero_frac = (k == 0).float().mean().item()
        # ~80% zero
        assert q_zero_frac > 0.5
        assert k_zero_frac > 0.5

    def test_correlated_keys(self) -> None:
        q, k = generate_synthetic_summaries(
            batch_size=1,
            num_heads=1,
            num_query_blocks=4,
            num_key_blocks=4,
            head_dim=32,
            distribution="correlated",
            seed=42,
            device="cpu",
            dtype=torch.float32,
        )
        # First 4 key blocks should be close to first 4 query blocks
        for i in range(4):
            diff = (k[0, 0, i] - q[0, 0, i]).abs().max().item()
            assert diff < 1.0, f"Block {i} diff too large: {diff}"

    def test_unknown_distribution_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown distribution"):
            generate_synthetic_summaries(
                batch_size=1,
                num_heads=1,
                num_query_blocks=4,
                num_key_blocks=4,
                head_dim=16,
                distribution="invalid",
                seed=42,
                device="cpu",
                dtype=torch.float32,
            )

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            generate_synthetic_summaries(
                batch_size=0, num_heads=1, num_query_blocks=4,
                num_key_blocks=4, head_dim=16, distribution="normal",
                seed=42, device="cpu", dtype=torch.float32,
            )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_output_dtype(self, dtype: torch.dtype) -> None:
        q, k = generate_synthetic_summaries(
            batch_size=1,
            num_heads=1,
            num_query_blocks=4,
            num_key_blocks=4,
            head_dim=16,
            distribution="normal",
            seed=42,
            device="cpu",
            dtype=dtype,
        )
        assert q.dtype == dtype
        assert k.dtype == dtype
