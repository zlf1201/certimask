"""Tests for quality-work frontier and Pareto frontier computation."""

from __future__ import annotations


def _dominates(
    a: dict[str, float | int | str | bool],
    b: dict[str, float | int | str | bool],
) -> bool:
    """Check if a dominates b."""
    aw = a["attention_tile_work_fraction"]
    bw = b["attention_tile_work_fraction"]
    am = a["kept_attention_mass_mean"]
    bm = b["kept_attention_mass_mean"]
    al = a["output_l2_relative_mean"]
    bl = b["output_l2_relative_mean"]
    return (aw <= bw and am >= bm and al <= bl
            and (aw < bw or am > bm or al < bl))


def compute_pareto_frontier(
    results: list[dict[str, float | int | str | bool]],
) -> list[dict[str, float | int | str | bool]]:
    """Compute Pareto frontier: minimize work_fraction, maximize kept_mass, minimize l2."""
    frontier: list[dict[str, float | int | str | bool]] = []
    for candidate in results:
        dominated = False
        to_remove: list[int] = []
        for i, existing in enumerate(frontier):
            if _dominates(existing, candidate):
                dominated = True
                break
            if _dominates(candidate, existing):
                to_remove.append(i)
        if not dominated:
            for i in reversed(to_remove):
                frontier.pop(i)
            frontier.append(candidate)
    return frontier


def select_policy(
    results: list[dict[str, float | int | str | bool]],
) -> dict[str, float | int | str | bool]:
    """Select best policy per layer following priority rules."""
    for pass_type, key in [
        ("strict_pass", "go_strict"),
        ("practical_pass", "go_practical"),
        ("relaxed_pass", "conditional_relaxed"),
        ("quality_only_pass", "quality_only_high_work"),
    ]:
        passing = [r for r in results if r[pass_type]]
        if passing:
            best = min(passing, key=lambda r: r["attention_tile_work_fraction"])
            best["decision"] = key
            best["fallback_reason"] = ""
            return best

    best = min(results, key=lambda r: r["output_l2_relative_mean"])
    best["decision"] = "fallback_quality"
    has_quality = any(r["quality_only_pass"] for r in results)
    has_work = any(r["attention_tile_work_fraction"] <= 0.55 for r in results)
    if not has_quality and not has_work:
        best["fallback_reason"] = "both_quality_and_work_failure"
    elif not has_quality:
        best["fallback_reason"] = "quality_failure"
    else:
        best["fallback_reason"] = "work_failure"
    return best


def _make_result(
    mass: float, l2: float, work: float, cosine: float = 0.99,
) -> dict[str, float | int | str | bool]:
    strict = mass >= 0.90 and cosine >= 0.95 and l2 <= 0.20 and work <= 0.50
    practical = mass >= 0.90 and cosine >= 0.95 and l2 <= 0.20 and work <= 0.55
    relaxed = mass >= 0.85 and cosine >= 0.95 and l2 <= 0.20 and work <= 0.55
    quality_only = mass >= 0.90 and cosine >= 0.95 and l2 <= 0.20
    return {
        "kept_attention_mass_mean": mass,
        "output_cosine_mean": cosine,
        "output_l2_relative_mean": l2,
        "attention_tile_work_fraction": work,
        "strict_pass": strict,
        "practical_pass": practical,
        "relaxed_pass": relaxed,
        "quality_only_pass": quality_only,
    }


class TestParetoFrontier:
    """Test Pareto frontier computation."""

    def test_dominated_removed(self) -> None:
        # A dominates B: A has lower work, same mass, same l2
        a = _make_result(0.92, 0.10, 0.45)
        b = _make_result(0.92, 0.10, 0.55)
        frontier = compute_pareto_frontier([a, b])
        assert len(frontier) == 1
        assert frontier[0]["attention_tile_work_fraction"] == 0.45

    def test_non_dominated_kept(self) -> None:
        # A has lower work but lower mass
        a = _make_result(0.88, 0.10, 0.40)
        b = _make_result(0.95, 0.10, 0.55)
        frontier = compute_pareto_frontier([a, b])
        assert len(frontier) == 2

    def test_single_config(self) -> None:
        a = _make_result(0.92, 0.10, 0.45)
        frontier = compute_pareto_frontier([a])
        assert len(frontier) == 1

    def test_empty(self) -> None:
        frontier = compute_pareto_frontier([])
        assert len(frontier) == 0


class TestPolicySelection:
    """Test policy selection logic."""

    def test_strict_pass_preferred(self) -> None:
        results = [
            _make_result(0.92, 0.10, 0.52),  # practical only
            _make_result(0.92, 0.10, 0.48),  # strict
        ]
        policy = select_policy(results)
        assert policy["decision"] == "go_strict"
        assert policy["attention_tile_work_fraction"] == 0.48

    def test_practical_when_no_strict(self) -> None:
        results = [
            _make_result(0.92, 0.10, 0.52),  # practical
            _make_result(0.92, 0.10, 0.58),  # relaxed only
        ]
        policy = select_policy(results)
        assert policy["decision"] == "go_practical"

    def test_relaxed_when_no_practical(self) -> None:
        results = [
            _make_result(0.87, 0.10, 0.50),  # relaxed
            _make_result(0.87, 0.10, 0.58),  # quality_only
        ]
        policy = select_policy(results)
        assert policy["decision"] == "conditional_relaxed"

    def test_quality_only_when_no_relaxed(self) -> None:
        results = [
            _make_result(0.92, 0.10, 0.60),  # quality_only (work too high)
        ]
        policy = select_policy(results)
        assert policy["decision"] == "quality_only_high_work"

    def test_fallback_quality(self) -> None:
        results = [
            _make_result(0.70, 0.50, 0.70),  # nothing passes
        ]
        policy = select_policy(results)
        assert policy["decision"] == "fallback_quality"

    def test_fallback_reason_quality_failure(self) -> None:
        results = [_make_result(0.70, 0.50, 0.40)]
        policy = select_policy(results)
        assert policy["fallback_reason"] == "quality_failure"

    def test_fallback_reason_work_failure(self) -> None:
        # Quality passes (quality_only=True) but work too high for relaxed
        results = [_make_result(0.92, 0.10, 0.60)]
        policy = select_policy(results)
        assert policy["decision"] == "quality_only_high_work"


class TestCosineInOutput:
    """Test that cosine is always present in results."""

    def test_cosine_field_exists(self) -> None:
        result = _make_result(0.92, 0.10, 0.45, cosine=0.98)
        assert "output_cosine_mean" in result
        assert result["output_cosine_mean"] == 0.98

    def test_cosine_in_quality_pass(self) -> None:
        # Low cosine should fail quality pass
        result = _make_result(0.92, 0.10, 0.45, cosine=0.90)
        assert result["strict_pass"] is False
        assert result["practical_pass"] is False
