"""Tests to enforce correct benchmark semantics and prevent misuse.

These tests verify that benchmark modes are correctly labeled and that
the optimized path uses the fused Triton certificate (not the slow PyTorch one).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_CROSSOVER_SCRIPT = (
    Path(__file__).parent.parent / "experiments" / "active"
    / "benchmark_aglr_crossover.py"
)


def _read_crossover() -> str:
    return _CROSSOVER_SCRIPT.read_text()


class TestBenchmarkModeLabels:
    """Verify benchmark modes have correct semantic labels."""

    def test_reference_first_is_non_deployable(self) -> None:
        """Reference-first validation mode must be labeled non-deployable."""
        semantics_path = (
            Path(__file__).parent.parent / "docs" / "BENCHMARK_SEMANTICS.md"
        )
        if semantics_path.exists():
            content = semantics_path.read_text()
            assert (
                "Deployable online?** No" in content
                or "Deployable online?: No" in content
            )

    def test_optimistic_cached_is_non_online(self) -> None:
        """Optimistic cached indexer mode must be labeled non-online."""
        semantics_path = (
            Path(__file__).parent.parent / "docs" / "BENCHMARK_SEMANTICS.md"
        )
        if semantics_path.exists():
            content = semantics_path.read_text()
            assert "optimistic" in content.lower()
            assert (
                "not a valid online" in content.lower()
                or "not online" in content.lower()
            )


class TestOptimizedPathUsesFusedCertificate:
    """Verify optimized benchmark uses fused Triton certificate."""

    def test_crossover_benchmark_uses_fused_cert(self) -> None:
        """Crossover benchmark must use triton_certified_topk_mask_partition."""
        content = _read_crossover()
        assert "triton_certified_topk_mask_partition" in content
        assert "from certimask.triton_topk_certificate import" in content

    def test_crossover_no_slow_cert_in_hot_path(self) -> None:
        """Crossover benchmark should not use certified_topk_mask in timing."""
        content = _read_crossover()
        lines = content.split("\n")
        in_benchmark_func = False
        for line in lines:
            if "def _run_online_full" in line or "def _run_mode" in line:
                in_benchmark_func = True
            elif line.startswith("def ") and in_benchmark_func:
                in_benchmark_func = False
            if (
                in_benchmark_func
                and "certified_topk_mask(" in line
                and "triton_certified_topk_mask_partition" not in line
            ):
                pytest.fail(
                    "certified_topk_mask used in benchmark hot path. "
                    "Use triton_certified_topk_mask_partition instead."
                )


class TestDenseBaselineReportsBothDtypes:
    """Dense baseline should report both FP16 and FP32 timings."""

    def test_crossover_reports_fp16_and_fp32(self) -> None:
        content = _read_crossover()
        assert "dense_sdpa_fp16" in content
        assert "dense_sdpa_fp32" in content


class TestBenchmarkSummaryFields:
    """Benchmark summaries must include semantic flags."""

    def test_crossover_summary_includes_crossover_points(self) -> None:
        content = _read_crossover()
        assert "first_online_crossover_length" in content
        assert "first_cached_quant_crossover_length" in content
        assert "first_optimistic_crossover_length" in content

    def test_crossover_summary_includes_readiness(self) -> None:
        content = _read_crossover()
        assert "ready_for_sparse_attention_kernel" in content


class TestLowBitFirstPathNotRequired:
    """Verify that not having a low-bit-first path doesn't break tests."""

    def test_no_lowbit_first_module_yet(self) -> None:
        """Low-bit-first is future work; absence should not cause errors."""
        from contextlib import suppress

        with suppress(ImportError):
            from certimask import triton_aglr_certimask_logsumexp_g4  # noqa: F401


class TestOutputsNotTracked:
    """Outputs should not be test dependencies."""

    def test_no_output_imports_in_tests(self) -> None:
        """Tests should not import from outputs/ directory."""
        import re

        test_dir = Path(__file__).parent
        import_pattern = re.compile(
            r"^\s*(?:from|import)\s+outputs\b", re.MULTILINE
        )
        for test_file in test_dir.glob("test_*.py"):
            content = test_file.read_text()
            match = import_pattern.search(content)
            assert match is None, (
                f"{test_file.name} has import from outputs/: {match.group()!r}"
            )
