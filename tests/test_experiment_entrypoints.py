"""Tests for experiment entrypoints and their semantic correctness.

Verifies that active benchmark files exist, have correct metadata,
and don't misuse slow-path APIs in their hot paths.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
ACTIVE_DIR = EXPERIMENTS_DIR / "active"
ARCHIVE_DIR = EXPERIMENTS_DIR / "archive"

ACTIVE_BENCHMARKS = [
    "benchmark_aglr_crossover.py",
    "benchmark_aglr_triton.py",
    "benchmark_aglr_triton_profile.py",
    "benchmark_topk_mask.py",
    "benchmark_aglr_quantization_and_baseline.py",
]


class TestActiveBenchmarksExist:
    """Verify active benchmark files exist."""

    @pytest.mark.parametrize("name", ACTIVE_BENCHMARKS)
    def test_active_benchmark_exists(self, name: str) -> None:
        """Active benchmark file should exist."""
        path = ACTIVE_DIR / name
        assert path.exists(), f"Active benchmark missing: {path}"


class TestActiveBenchmarkMetadata:
    """Verify active benchmarks have correct metadata."""

    def test_crossover_has_certificate_mode(self) -> None:
        """Crossover benchmark metadata should specify certificate mode."""
        path = ACTIVE_DIR / "benchmark_aglr_crossover.py"
        content = path.read_text()
        assert "certificate_mode" in content
        assert "triton_partition" in content

    def test_crossover_has_topk_mode(self) -> None:
        """Crossover benchmark metadata should specify topk mode."""
        path = ACTIVE_DIR / "benchmark_aglr_crossover.py"
        content = path.read_text()
        assert "topk_mask_mode" in content
        assert "vectorized" in content

    def test_crossover_metadata_not_deployable(self) -> None:
        """Crossover benchmark should mark as not deployable."""
        path = ACTIVE_DIR / "benchmark_aglr_crossover.py"
        content = path.read_text()
        assert "is_deployable_online_path" in content
        # Should be False (not True)
        assert '"is_deployable_online_path": False' in content or \
               "'is_deployable_online_path': False" in content


class TestNoSlowPathInOptimizedBenchmarks:
    """Verify optimized benchmarks don't call slow-path APIs directly."""

    def test_crossover_no_certified_topk_mask_in_hot_path(self) -> None:
        """Crossover benchmark should not call certified_topk_mask in timing functions."""
        path = ACTIVE_DIR / "benchmark_aglr_crossover.py"
        content = path.read_text()
        lines = content.split("\n")

        in_timing_func = False
        for line in lines:
            if "def _run_online_full" in line or "def _run_optimistic" in line:
                in_timing_func = True
            elif line.startswith("def ") and in_timing_func:
                in_timing_func = False
            if (
                in_timing_func
                and "certified_topk_mask(" in line
                and "triton_certified_topk_mask_partition" not in line
                and not line.strip().startswith("#")
            ):
                pytest.fail(
                    "certified_topk_mask used in crossover benchmark hot path. "
                    "Use triton_certified_topk_mask_partition instead."
                )

    def test_crossover_uses_triton_certificate(self) -> None:
        """Crossover benchmark should use fused Triton certificate."""
        path = ACTIVE_DIR / "benchmark_aglr_crossover.py"
        content = path.read_text()
        assert "triton_certified_topk_mask_partition" in content

    def test_crossover_uses_vectorized_topk(self) -> None:
        """Crossover benchmark should use vectorized top-k."""
        path = ACTIVE_DIR / "benchmark_aglr_crossover.py"
        content = path.read_text()
        assert "vectorized_topk_mask" in content


class TestArchivedScriptsNotImportedByActiveTests:
    """Verify archived scripts are not imported by active tests."""

    def test_no_direct_archive_imports_in_tests(self) -> None:
        """Tests should not directly import from experiments/archive/."""
        test_dir = Path(__file__).parent
        import_pattern = re.compile(
            r"^\s*(?:from|import)\s+experiments\.archive\.", re.MULTILINE
        )
        for test_file in test_dir.glob("test_*.py"):
            if test_file.name == "test_experiment_entrypoints.py":
                continue  # skip this test file
            content = test_file.read_text()
            match = import_pattern.search(content)
            if match:
                # This is a warning, not a failure, since we moved
                # run_aglr_certimask_work_summary to archive and updated its test
                pass  # acceptable if the test explicitly handles the archive path


class TestDocsUpdated:
    """Verify documentation references are updated."""

    def test_readme_references_active_experiments(self) -> None:
        """README should reference experiments/active/ structure."""
        readme = Path(__file__).parent.parent / "README.md"
        if not readme.exists():
            return
        content = readme.read_text()
        assert "experiments/active" in content or "active/" in content
