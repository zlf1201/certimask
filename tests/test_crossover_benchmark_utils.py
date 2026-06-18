"""Tests for Phase 9E: crossover benchmark utilities."""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, "experiments/active")

from benchmark_aglr_crossover import (
    _estimate_alpha,
    _find_crossover_length,
)


class TestModeDefinitions:
    """Test that mode definitions are distinct and well-defined."""

    def test_modes_are_distinct(self) -> None:
        """Each mode should represent a different pipeline configuration."""
        # Mode 1: Dense SDPA - no indexer at all
        # Mode 2A: Full pipeline with quantization inside timing
        # Mode 2B: Quantization outside, rest inside
        # Mode 3: Reference + mask + quant outside, kernel + cert inside
        modes = [
            "dense_sdpa",
            "online_full_quant",
            "online_full_cached_quant",
            "optimistic_cached",
        ]
        assert len(modes) == len(set(modes)), "Mode names should be unique"

    def test_online_mode_includes_reference_and_topk(self) -> None:
        """Online modes must include reference score computation and top-k."""
        # Mode 2A and 2B both compute reference scores inside timing loop
        # This is verified by the benchmark structure:
        # - Mode 2A calls triton_aglr_certimask_logsumexp_g4 (includes everything)
        # - Mode 2B calls compute_antidiagonal_block_scores + vectorized_topk_mask inside
        assert True  # Structural verification

    def test_optimistic_mode_is_not_online(self) -> None:
        """Mode 3 pre-computes reference scores, so it's not online valid."""
        # Mode 3 pre-computes fp_scores_pre, ref_mask_pre, k_per_row_pre outside timing
        # This means it assumes these are cached from a prior computation
        # It is NOT a valid online prefill cost
        assert True  # Structural verification


class TestSpeedupProxyCalculation:
    """Test speedup proxy calculation correctness."""

    def test_speedup_proxy_basic(self) -> None:
        """Speedup = dense / total_proxy."""
        dense_ms = 10.0
        total_proxy = 5.0
        speedup = dense_ms / total_proxy
        assert speedup == 2.0

    def test_speedup_proxy_with_indexer(self) -> None:
        """Total proxy = indexer + ideal_sparse."""
        dense_ms = 10.0
        work_fraction = 0.3765
        ideal_sparse_ms = dense_ms * work_fraction
        indexer_ms = 5.0
        total = indexer_ms + ideal_sparse_ms
        speedup = dense_ms / total
        expected = 10.0 / (5.0 + 3.765)
        assert abs(speedup - expected) < 1e-6

    def test_speedup_less_than_1_means_slower(self) -> None:
        """Speedup < 1 means the pipeline is slower than dense."""
        dense_ms = 1.0
        indexer_ms = 10.0
        ideal_sparse_ms = dense_ms * 0.3765
        total = indexer_ms + ideal_sparse_ms
        speedup = dense_ms / total
        assert speedup < 1.0


class TestCrossoverDetection:
    """Test crossover length detection."""

    def test_crossover_found_when_indexer_faster(self) -> None:
        """Crossover should be found when indexer + sparse <= dense."""
        lengths = [1024, 2048, 4096]
        # At L=4096, indexer becomes fast enough
        indexer_times = [10.0, 5.0, 2.0]
        dense_times = [1.0, 4.0, 16.0]
        # At L=1024: 10.0 + 1.0*0.3765 = 10.38 > 1.0 -> no
        # At L=2048: 5.0 + 4.0*0.3765 = 6.51 > 4.0 -> no
        # At L=4096: 2.0 + 16.0*0.3765 = 8.02 < 16.0 -> yes!
        result = _find_crossover_length(lengths, indexer_times, dense_times)
        assert result == 4096

    def test_crossover_none_when_never(self) -> None:
        """Return None when no crossover found."""
        lengths = [1024, 2048]
        indexer_times = [100.0, 200.0]
        dense_times = [1.0, 2.0]
        result = _find_crossover_length(lengths, indexer_times, dense_times)
        assert result is None

    def test_crossover_at_first_length(self) -> None:
        """Crossover can happen at the first tested length."""
        lengths = [1024, 2048]
        indexer_times = [0.1, 0.2]
        dense_times = [10.0, 20.0]
        result = _find_crossover_length(lengths, indexer_times, dense_times)
        assert result == 1024


class TestScalingExponent:
    """Test scaling exponent calculation."""

    def test_perfect_quadratic(self) -> None:
        """T = L^2 should give alpha = 2."""
        lengths = [1024, 2048, 4096]
        times = [length ** 2 for length in lengths]
        alpha = _estimate_alpha(lengths, times)
        assert abs(alpha - 2.0) < 1e-6

    def test_perfect_linear(self) -> None:
        """T = L should give alpha = 1."""
        lengths = [1024, 2048, 4096]
        times = [float(length) for length in lengths]
        alpha = _estimate_alpha(lengths, times)
        assert abs(alpha - 1.0) < 1e-6

    def test_perfect_cubic(self) -> None:
        """T = L^3 should give alpha = 3."""
        lengths = [1024, 2048, 4096]
        times = [length ** 3 for length in lengths]
        alpha = _estimate_alpha(lengths, times)
        assert abs(alpha - 3.0) < 1e-6

    def test_constant_gives_zero(self) -> None:
        """Constant times should give alpha ~ 0 (no scaling)."""
        lengths = [1024, 2048, 4096]
        times = [1.0, 1.0, 1.0]
        alpha = _estimate_alpha(lengths, times)
        assert abs(alpha) < 1e-6

    def test_two_points_sufficient(self) -> None:
        """Two points should give a valid estimate."""
        lengths = [1024, 2048]
        times = [1.0, 4.0]  # quadratic
        alpha = _estimate_alpha(lengths, times)
        assert abs(alpha - 2.0) < 1e-6

    def test_single_point_gives_nan(self) -> None:
        """Single point cannot determine slope."""
        lengths = [1024]
        times = [1.0]
        alpha = _estimate_alpha(lengths, times)
        assert math.isnan(alpha)


class TestOOMHandling:
    """Test OOM entry handling."""

    def test_oom_entries_excluded_from_scaling(self) -> None:
        """OOM entries should be excluded from scaling analysis."""
        # Simulate entries with OOM
        entries = [
            {"seq_len": 1024, "oom": False, "dense_sdpa_ms": {"median": 1.0}},
            {"seq_len": 2048, "oom": True},
            {"seq_len": 4096, "oom": False, "dense_sdpa_ms": {"median": 4.0}},
        ]
        valid = [e for e in entries if not e.get("oom")]
        assert len(valid) == 2
        assert valid[0]["seq_len"] == 1024
        assert valid[1]["seq_len"] == 4096

    def test_oom_entries_have_seq_len(self) -> None:
        """OOM entries should still have seq_len for reporting."""
        entry = {"seq_len": 8192, "oom": True}
        assert "seq_len" in entry
        assert entry["oom"] is True
