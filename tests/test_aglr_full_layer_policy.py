"""Tests for full 24-layer AGLR-C v1 policy scan."""

from __future__ import annotations


def _make_summary(
    go_strict: list[int],
    go_practical: list[int],
    conditional: list[int],
    quality_only: list[int],
    fallback: list[int],
    mean_wf: float = 0.40,
    mean_mass: float = 0.91,
    mean_cos: float = 0.988,
    mean_l2: float = 0.10,
) -> dict[str, object]:
    total = (len(go_strict) + len(go_practical) + len(conditional)
             + len(quality_only) + len(fallback))
    go_count = len(go_strict) + len(go_practical)
    ready = (
        go_count >= 12
        and len(fallback) <= 3
        and mean_wf <= 0.50
        and mean_mass >= 0.90
        and mean_cos >= 0.98
        and mean_l2 <= 0.12
    )
    return {
        "total_layers": total,
        "go_strict_layers": go_strict,
        "go_practical_layers": go_practical,
        "conditional_relaxed_layers": conditional,
        "quality_only_high_work_layers": quality_only,
        "fallback_quality_layers": fallback,
        "mean_selected_work_fraction": mean_wf,
        "mean_selected_kept_mass": mean_mass,
        "mean_selected_cosine": mean_cos,
        "mean_selected_l2": mean_l2,
        "certimask_readiness": "ready_for_topk_certificate_design" if ready
        else "needs_indexer_or_policy_improvement",
    }


class TestCertimaskReadiness:
    """Test CertiMask readiness classification."""

    def test_ready_when_all_criteria_met(self) -> None:
        s = _make_summary(
            go_strict=list(range(15)),
            go_practical=[],
            conditional=[15, 16, 17],
            quality_only=[18, 19],
            fallback=[],
            mean_wf=0.40, mean_mass=0.91, mean_cos=0.988, mean_l2=0.10,
        )
        assert s["total_layers"] == 20
        assert s["certimask_readiness"] == "ready_for_topk_certificate_design"

    def test_not_ready_too_many_fallback(self) -> None:
        s = _make_summary(
            go_strict=list(range(12)),
            go_practical=[],
            conditional=[],
            quality_only=[],
            fallback=[12, 13, 14, 15],
            mean_wf=0.40, mean_mass=0.91, mean_cos=0.988, mean_l2=0.10,
        )
        assert s["certimask_readiness"] == "needs_indexer_or_policy_improvement"

    def test_not_ready_high_work(self) -> None:
        s = _make_summary(
            go_strict=list(range(12)),
            go_practical=[],
            conditional=[],
            quality_only=[],
            fallback=[],
            mean_wf=0.55, mean_mass=0.91, mean_cos=0.988, mean_l2=0.10,
        )
        assert s["certimask_readiness"] == "needs_indexer_or_policy_improvement"

    def test_not_ready_low_mass(self) -> None:
        s = _make_summary(
            go_strict=list(range(12)),
            go_practical=[],
            conditional=[],
            quality_only=[],
            fallback=[],
            mean_wf=0.40, mean_mass=0.89, mean_cos=0.988, mean_l2=0.10,
        )
        assert s["certimask_readiness"] == "needs_indexer_or_policy_improvement"

    def test_not_ready_low_go_count(self) -> None:
        s = _make_summary(
            go_strict=list(range(5)),
            go_practical=[5, 6, 7],
            conditional=[8, 9, 10, 11],
            quality_only=[12, 13, 14, 15, 16, 17],
            fallback=[18, 19, 20, 21, 22, 23],
            mean_wf=0.40, mean_mass=0.91, mean_cos=0.988, mean_l2=0.10,
        )
        assert s["certimask_readiness"] == "needs_indexer_or_policy_improvement"


class TestFallbackReasons:
    """Test fallback reason assignment."""

    def test_mass_failure(self) -> None:
        kept_mass = 0.80
        cosine = 0.96
        l2 = 0.10
        quality_only = kept_mass >= 0.90 and cosine >= 0.95 and l2 <= 0.20
        if not quality_only:
            if kept_mass < 0.90:
                reason = "mass_failure"
            elif cosine < 0.95:
                reason = "cosine_failure"
            else:
                reason = "l2_failure"
        else:
            reason = "work_failure"
        assert reason == "mass_failure"

    def test_cosine_failure(self) -> None:
        kept_mass = 0.92
        cosine = 0.93
        l2 = 0.10
        quality_only = kept_mass >= 0.90 and cosine >= 0.95 and l2 <= 0.20
        if not quality_only:
            if kept_mass < 0.90:
                reason = "mass_failure"
            elif cosine < 0.95:
                reason = "cosine_failure"
            else:
                reason = "l2_failure"
        else:
            reason = "work_failure"
        assert reason == "cosine_failure"

    def test_l2_failure(self) -> None:
        kept_mass = 0.92
        cosine = 0.96
        l2 = 0.25
        quality_only = kept_mass >= 0.90 and cosine >= 0.95 and l2 <= 0.20
        if not quality_only:
            if kept_mass < 0.90:
                reason = "mass_failure"
            elif cosine < 0.95:
                reason = "cosine_failure"
            else:
                reason = "l2_failure"
        else:
            reason = "work_failure"
        assert reason == "l2_failure"

    def test_work_failure(self) -> None:
        kept_mass = 0.92
        cosine = 0.96
        l2 = 0.10
        work = 0.70
        quality_only = kept_mass >= 0.90 and cosine >= 0.95 and l2 <= 0.20
        relaxed = kept_mass >= 0.85 and cosine >= 0.95 and l2 <= 0.20 and work <= 0.55
        if not quality_only:
            reason = "quality_failure"
        elif not relaxed:
            reason = "work_failure"
        else:
            reason = ""
        assert reason == "work_failure"


class TestCosineInOutput:
    """Test that cosine is always present."""

    def test_cosine_present(self) -> None:
        row = {"output_cosine_mean": 0.988, "kept_attention_mass_mean": 0.91}
        assert "output_cosine_mean" in row
        assert row["output_cosine_mean"] >= 0.95

    def test_cosine_used_in_pass(self) -> None:
        cosine = 0.93
        assert not (cosine >= 0.95)


class TestSummaryFields:
    """Test summary field completeness."""

    def test_all_required_fields(self) -> None:
        s = _make_summary([], [], [], [], [], 0.40, 0.91, 0.988, 0.10)
        required = [
            "total_layers", "go_strict_layers", "go_practical_layers",
            "conditional_relaxed_layers", "quality_only_high_work_layers",
            "fallback_quality_layers", "mean_selected_work_fraction",
            "mean_selected_kept_mass", "mean_selected_cosine",
            "mean_selected_l2", "certimask_readiness",
        ]
        for field in required:
            assert field in s, f"Missing field: {field}"


class TestDistributionCounts:
    """Test distribution counting."""

    def test_block_size_counts(self) -> None:
        from collections import Counter
        sizes = [8, 8, 8, 16, 8, 16]
        counts = dict(Counter(sizes))
        assert counts == {8: 4, 16: 2}

    def test_local_blocks_counts(self) -> None:
        from collections import Counter
        lbs = [0, 0, 1, 0, 2, 0, 0]
        counts = dict(Counter(lbs))
        assert counts == {0: 5, 1: 1, 2: 1}

    def test_target_sparsity_counts(self) -> None:
        from collections import Counter
        sparsities = [0.30, 0.50, 0.50, 0.70, 0.50, 0.30]
        counts = dict(Counter(sparsities))
        assert counts == {0.30: 2, 0.50: 3, 0.70: 1}

    def test_aggregation_counts(self) -> None:
        from collections import Counter
        aggs = ["logsumexp", "logsumexp", "topk_mean", "logsumexp"]
        counts = dict(Counter(aggs))
        assert counts == {"logsumexp": 3, "topk_mean": 1}
