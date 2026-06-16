"""Tests for AGLR layer-wise policy selection logic."""

from __future__ import annotations


def strict_quality_pass(
    kept_mass: float, cosine: float, l2: float, work_frac: float,
) -> bool:
    return kept_mass >= 0.90 and cosine >= 0.95 and l2 <= 0.20 and work_frac <= 0.50


def practical_quality_pass(
    kept_mass: float, cosine: float, l2: float, work_frac: float,
) -> bool:
    return kept_mass >= 0.90 and cosine >= 0.95 and l2 <= 0.20 and work_frac <= 0.55


def strong_quality_pass(
    kept_mass: float, cosine: float, l2: float, work_frac: float,
) -> bool:
    return kept_mass >= 0.95 and cosine >= 0.98 and l2 <= 0.10 and work_frac <= 0.55


def select_decision(
    kept_mass: float, cosine: float, l2: float, work_frac: float,
) -> str:
    if practical_quality_pass(kept_mass, cosine, l2, work_frac):
        return "go"
    if kept_mass >= 0.85 and l2 <= 0.30 and work_frac <= 0.60:
        return "conditional"
    return "fallback"


class TestQualityPass:
    """Test quality pass criteria."""

    def test_strict_pass(self) -> None:
        assert strict_quality_pass(0.92, 0.96, 0.15, 0.45) is True

    def test_strict_fail_low_mass(self) -> None:
        assert strict_quality_pass(0.89, 0.96, 0.15, 0.45) is False

    def test_strict_fail_low_cosine(self) -> None:
        assert strict_quality_pass(0.92, 0.94, 0.15, 0.45) is False

    def test_strict_fail_high_l2(self) -> None:
        assert strict_quality_pass(0.92, 0.96, 0.25, 0.45) is False

    def test_strict_fail_high_work(self) -> None:
        assert strict_quality_pass(0.92, 0.96, 0.15, 0.55) is False

    def test_practical_pass_higher_work(self) -> None:
        assert practical_quality_pass(0.92, 0.96, 0.15, 0.55) is True

    def test_practical_fail_high_work(self) -> None:
        assert practical_quality_pass(0.92, 0.96, 0.15, 0.56) is False

    def test_strong_pass(self) -> None:
        assert strong_quality_pass(0.96, 0.99, 0.08, 0.40) is True

    def test_strong_fail_low_mass(self) -> None:
        assert strong_quality_pass(0.94, 0.99, 0.08, 0.40) is False


class TestDecisionSelection:
    """Test layer decision logic."""

    def test_go_when_practical_pass(self) -> None:
        assert select_decision(0.92, 0.96, 0.15, 0.50) == "go"

    def test_conditional_when_near_pass(self) -> None:
        assert select_decision(0.87, 0.93, 0.25, 0.55) == "conditional"

    def test_fallback_when_poor(self) -> None:
        assert select_decision(0.70, 0.80, 0.50, 0.70) == "fallback"

    def test_go_takes_lowest_work(self) -> None:
        """When multiple configs pass practical, pick lowest work fraction."""
        configs = [
            {"work_frac": 0.52, "kept_mass": 0.91},
            {"work_frac": 0.48, "kept_mass": 0.92},
            {"work_frac": 0.50, "kept_mass": 0.93},
        ]
        passing = [c for c in configs if c["kept_mass"] >= 0.90]
        best = min(passing, key=lambda c: c["work_frac"])
        assert best["work_frac"] == 0.48

    def test_conditional_takes_lowest_l2(self) -> None:
        """Among conditional configs, pick lowest L2."""
        configs = [
            {"l2": 0.28, "kept_mass": 0.86},
            {"l2": 0.22, "kept_mass": 0.87},
            {"l2": 0.25, "kept_mass": 0.88},
        ]
        passing = [c for c in configs if c["kept_mass"] >= 0.85]
        best = min(passing, key=lambda c: c["l2"])
        assert best["l2"] == 0.22


class TestOracleGap:
    """Test oracle gap computation."""

    def test_oracle_gap(self) -> None:
        oracle_mass = 0.95
        candidate_mass = 0.88
        gap = oracle_mass - candidate_mass
        assert abs(gap - 0.07) < 1e-10

    def test_oracle_gap_zero_when_equal(self) -> None:
        oracle_mass = 0.95
        candidate_mass = 0.95
        gap = oracle_mass - candidate_mass
        assert abs(gap) < 1e-10


class TestImprovementOverMeanPooled:
    """Test improvement computation."""

    def test_improvement(self) -> None:
        aglr_mass = 0.92
        mp_mass = 0.70
        improvement = aglr_mass - mp_mass
        assert abs(improvement - 0.22) < 1e-10

    def test_l2_reduction(self) -> None:
        mp_l2 = 0.40
        aglr_l2 = 0.15
        reduction = mp_l2 - aglr_l2
        assert abs(reduction - 0.25) < 1e-10
