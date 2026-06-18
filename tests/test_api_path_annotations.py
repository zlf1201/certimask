"""Tests for API path annotations and docstring warnings.

Verifies that slow-path APIs have proper docstring warnings and that
the API path guide exists.
"""

from __future__ import annotations

from pathlib import Path


class TestSlowPathDocstringWarnings:
    """Verify slow-path APIs have docstring warnings."""

    def test_certified_topk_mask_docstring_has_reference(self) -> None:
        """certified_topk_mask docstring should mention 'reference' or 'validation'."""
        from certimask.topk_certificate import certified_topk_mask

        doc = certified_topk_mask.__doc__ or ""
        assert "reference" in doc.lower() or "validation" in doc.lower(), (
            "certified_topk_mask docstring should mention 'reference' or 'validation'"
        )

    def test_certified_topk_mask_docstring_warns_not_optimized(self) -> None:
        """certified_topk_mask docstring should warn about slow performance."""
        from certimask.topk_certificate import certified_topk_mask

        doc = certified_topk_mask.__doc__ or ""
        has_warning = (
            "slow" in doc.lower()
            or "not.*optimized" in doc.lower()
            or "warning" in doc.lower()
        )
        assert has_warning, (
            "certified_topk_mask docstring should warn about slow performance"
        )

    def test_aglr_local_plus_landmark_mask_docstring_has_historical(self) -> None:
        """aglr_local_plus_landmark_mask docstring should mention historical."""
        from certimask.aglr_indexer import aglr_local_plus_landmark_mask

        doc = aglr_local_plus_landmark_mask.__doc__ or ""
        has_annotation = (
            "historical" in doc.lower()
            or "reference" in doc.lower()
            or "warning" in doc.lower()
        )
        assert has_annotation, (
            "aglr_local_plus_landmark_mask docstring should mention historical"
        )

    def test_aglr_certimask_topk_docstring_has_validation(self) -> None:
        """aglr_certimask_topk docstring should mention validation."""
        from certimask.aglr_certimask import aglr_certimask_topk

        doc = aglr_certimask_topk.__doc__ or ""
        has_annotation = (
            "validation" in doc.lower()
            or "reference" in doc.lower()
            or "warning" in doc.lower()
        )
        assert has_annotation, (
            "aglr_certimask_topk docstring should mention validation"
        )


class TestApiPathGuideExists:
    """Verify the API path guide documentation exists."""

    def test_api_paths_md_exists(self) -> None:
        """docs/API_PATHS.md should exist."""
        api_paths = Path(__file__).parent.parent / "docs" / "API_PATHS.md"
        assert api_paths.exists(), "docs/API_PATHS.md does not exist"

    def test_api_paths_md_has_required_sections(self) -> None:
        """docs/API_PATHS.md should have required sections."""
        api_paths = Path(__file__).parent.parent / "docs" / "API_PATHS.md"
        if not api_paths.exists():
            return  # caught by test_api_paths_md_exists

        content = api_paths.read_text()
        required = [
            "Historical",
            "Reference-First",
            "Optimized",
            "Deployable",
            "Do Not Use",
        ]
        for section in required:
            assert section.lower() in content.lower(), (
                f"docs/API_PATHS.md missing section: {section}"
            )


class TestReadmeNoSpeedupClaim:
    """README should not claim end-to-end speedup."""

    def test_readme_states_no_speedup_claim(self) -> None:
        """README.md should state no end-to-end speedup claim."""
        readme = Path(__file__).parent.parent / "README.md"
        if not readme.exists():
            return

        content = readme.read_text()
        # Check for explicit no-speedup claim
        assert (
            "no end-to-end" in content.lower()
            or "not demonstrated" in content.lower()
            or "does not demonstrate" in content.lower()
        ), "README should state no end-to-end speedup claim"
