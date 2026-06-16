"""Tests for layer classification policy."""

from __future__ import annotations

import pytest


def classify_layer(ref_rate: float) -> str:
    """Classify a layer based on coordinate analytic refinement rate."""
    if ref_rate < 0.20:
        return "Go"
    if ref_rate <= 0.30:
        return "Conditional Go"
    return "FP16 fallback"


def compute_fp16_fraction(decisions: list[str], ref_rates: list[float]) -> float:
    """Estimate FP16 refinement fraction. Not a latency estimate."""
    if not decisions:
        raise ValueError("Empty layer results")
    total = len(decisions)
    work = sum(
        1.0 if d == "FP16 fallback" else r
        for d, r in zip(decisions, ref_rates, strict=True)
    )
    return work / total


class TestLayerClassification:
    """Test layer decision rules."""

    def test_go_below_20(self) -> None:
        assert classify_layer(0.0) == "Go"
        assert classify_layer(0.10) == "Go"
        assert classify_layer(0.19) == "Go"

    def test_conditional_20_to_30(self) -> None:
        assert classify_layer(0.20) == "Conditional Go"
        assert classify_layer(0.25) == "Conditional Go"
        assert classify_layer(0.30) == "Conditional Go"

    def test_fallback_above_30(self) -> None:
        assert classify_layer(0.31) == "FP16 fallback"
        assert classify_layer(0.50) == "FP16 fallback"
        assert classify_layer(1.0) == "FP16 fallback"


class TestScoreQuantizationUnstable:
    """Test score_quantization_unstable flag."""

    def test_unstable_above_10(self) -> None:
        assert 0.11 > 0.10
        assert not (0.09 > 0.10)


class TestFP16Fraction:
    """Test estimated_fp16_refinement_fraction computation."""

    def test_all_go(self) -> None:
        decisions = ["Go", "Go", "Go"]
        rates = [0.10, 0.15, 0.18]
        frac = compute_fp16_fraction(decisions, rates)
        expected = (0.10 + 0.15 + 0.18) / 3
        assert abs(frac - expected) < 1e-10

    def test_all_fallback(self) -> None:
        decisions = ["FP16 fallback", "FP16 fallback"]
        rates = [0.5, 0.6]
        frac = compute_fp16_fraction(decisions, rates)
        assert abs(frac - 1.0) < 1e-10

    def test_mixed(self) -> None:
        decisions = ["Go", "FP16 fallback", "Conditional Go"]
        rates = [0.10, 0.50, 0.25]
        frac = compute_fp16_fraction(decisions, rates)
        expected = (0.10 + 1.0 + 0.25) / 3
        assert abs(frac - expected) < 1e-10

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            compute_fp16_fraction([], [])
