"""Tests for Phase 8C: AGLR-C + CertiMask Quality/Work Summary."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper: build synthetic Phase 7E policy CSV
# ---------------------------------------------------------------------------
def _write_policy_csv(path: Path) -> None:
    rows = [
        {
            "layer": 0, "decision": "conditional_relaxed",
            "selected_block_size": 8, "selected_target_sparsity": 0.5,
            "selected_local_blocks": 0, "selected_aggregation": "logsumexp",
            "selected_work_fraction": 0.504, "selected_kept_mass": 0.883,
            "selected_cosine": 0.991, "selected_l2": 0.062,
        },
        {
            "layer": 1, "decision": "conditional_relaxed",
            "selected_block_size": 8, "selected_target_sparsity": 0.5,
            "selected_local_blocks": 0, "selected_aggregation": "logsumexp",
            "selected_work_fraction": 0.504, "selected_kept_mass": 0.865,
            "selected_cosine": 0.993, "selected_l2": 0.076,
        },
        {
            "layer": 2, "decision": "go_practical",
            "selected_block_size": 8, "selected_target_sparsity": 0.5,
            "selected_local_blocks": 2, "selected_aggregation": "topk_mean",
            "selected_work_fraction": 0.504, "selected_kept_mass": 0.904,
            "selected_cosine": 0.991, "selected_l2": 0.104,
        },
        {
            "layer": 3, "decision": "go_strict",
            "selected_block_size": 8, "selected_target_sparsity": 0.75,
            "selected_local_blocks": 0, "selected_aggregation": "logsumexp",
            "selected_work_fraction": 0.256, "selected_kept_mass": 0.903,
            "selected_cosine": 0.971, "selected_l2": 0.165,
        },
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Helper: build synthetic Phase 8B certimask CSV
# ---------------------------------------------------------------------------
def _write_certimask_csv(path: Path, *, layer2_unsupported: bool = True) -> None:
    rows = [
        {
            "layer": 0, "exact_match": True, "mismatch_count": 0,
            "fallback_rate": 0.745, "row_certification_rate": 0.018,
            "certified_keep_rate": 0.273, "certified_drop_rate": 0.236,
            "mean_interval_width": 6.608, "p90_interval_width": 12.171,
            "p99_interval_width": 20.938, "unsupported_aggregation": False,
            "valid_tiles": 115584, "selected_tiles": 58240,
        },
        {
            "layer": 1, "exact_match": True, "mismatch_count": 0,
            "fallback_rate": 0.423, "row_certification_rate": 0.068,
            "certified_keep_rate": 0.585, "certified_drop_rate": 0.569,
            "mean_interval_width": 1.841, "p90_interval_width": 5.448,
            "p99_interval_width": 6.132, "unsupported_aggregation": False,
            "valid_tiles": 115584, "selected_tiles": 58240,
        },
        {
            "layer": 2, "exact_match": True, "mismatch_count": 0,
            "fallback_rate": "", "row_certification_rate": "",
            "certified_keep_rate": "", "certified_drop_rate": "",
            "mean_interval_width": "", "p90_interval_width": "",
            "p99_interval_width": "",
            "unsupported_aggregation": layer2_unsupported,
            "valid_tiles": "", "selected_tiles": "",
        },
        {
            "layer": 3, "exact_match": True, "mismatch_count": 0,
            "fallback_rate": 0.090, "row_certification_rate": 0.193,
            "certified_keep_rate": 0.838, "certified_drop_rate": 0.935,
            "mean_interval_width": 0.338, "p90_interval_width": 0.444,
            "p99_interval_width": 0.581, "unsupported_aggregation": False,
            "valid_tiles": 115584, "selected_tiles": 29568,
        },
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
def _import_summary():
    """Import the summary builder from the experiment module."""
    import importlib
    import sys
    # Ensure project root is on sys.path for experiments.* imports
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    mod = importlib.import_module(
        "experiments.archive.run_aglr_certimask_work_summary"
    )
    return mod


class TestUnsupportedFallback:
    """Test that unsupported aggregation is counted as fp fallback = 1.0."""

    def test_unsupported_layer_fp_fallback_is_one(self) -> None:
        mod = _import_summary()
        with tempfile.TemporaryDirectory() as td:
            policy_path = Path(td) / "policy.csv"
            cm_path = Path(td) / "certimask.csv"
            _write_policy_csv(policy_path)
            _write_certimask_csv(cm_path, layer2_unsupported=True)

            policy_rows = mod.read_csv(policy_path)
            cm_rows = mod.read_csv(cm_path)
            merged = mod.merge_layers(policy_rows, cm_rows)

            layer2 = [r for r in merged if r["layer"] == 2][0]
            assert layer2["indexer_fp_fallback_fraction"] == 1.0
            assert layer2["certimask_supported"] is False


class TestSupportedFallback:
    """Test that supported layer uses fallback_rate as fp fallback fraction."""

    def test_supported_layer_uses_fallback_rate(self) -> None:
        mod = _import_summary()
        with tempfile.TemporaryDirectory() as td:
            policy_path = Path(td) / "policy.csv"
            cm_path = Path(td) / "certimask.csv"
            _write_policy_csv(policy_path)
            _write_certimask_csv(cm_path)

            policy_rows = mod.read_csv(policy_path)
            cm_rows = mod.read_csv(cm_path)
            merged = mod.merge_layers(policy_rows, cm_rows)

            layer3 = [r for r in merged if r["layer"] == 3][0]
            assert layer3["certimask_supported"] is True
            assert abs(layer3["indexer_fp_fallback_fraction"] - 0.090) < 1e-6


class TestMeanAttentionWork:
    """Test mean attention work computation."""

    def test_mean_attention_work(self) -> None:
        mod = _import_summary()
        with tempfile.TemporaryDirectory() as td:
            policy_path = Path(td) / "policy.csv"
            cm_path = Path(td) / "certimask.csv"
            _write_policy_csv(policy_path)
            _write_certimask_csv(cm_path)

            policy_rows = mod.read_csv(policy_path)
            cm_rows = mod.read_csv(cm_path)
            merged = mod.merge_layers(policy_rows, cm_rows)
            summary = mod.compute_summary(merged)

            expected = (0.504 + 0.504 + 0.504 + 0.256) / 4
            assert abs(summary["mean_attention_tile_work_fraction"] - expected) < 1e-6


class TestMeanFpFallback:
    """Test mean fp fallback computation."""

    def test_mean_fp_fallback(self) -> None:
        mod = _import_summary()
        with tempfile.TemporaryDirectory() as td:
            policy_path = Path(td) / "policy.csv"
            cm_path = Path(td) / "certimask.csv"
            _write_policy_csv(policy_path)
            _write_certimask_csv(cm_path)

            policy_rows = mod.read_csv(policy_path)
            cm_rows = mod.read_csv(cm_path)
            merged = mod.merge_layers(policy_rows, cm_rows)
            summary = mod.compute_summary(merged)

            # layer0=0.745, layer1=0.423, layer2=1.0 (unsupported), layer3=0.090
            expected = (0.745 + 0.423 + 1.0 + 0.090) / 4
            assert abs(summary["mean_fp_score_fallback_fraction"] - expected) < 1e-6


class TestWeightedMeanFpFallback:
    """Test weighted mean fp fallback computation."""

    def test_weighted_mean_fp_fallback(self) -> None:
        mod = _import_summary()
        with tempfile.TemporaryDirectory() as td:
            policy_path = Path(td) / "policy.csv"
            cm_path = Path(td) / "certimask.csv"
            _write_policy_csv(policy_path)
            _write_certimask_csv(cm_path)

            policy_rows = mod.read_csv(policy_path)
            cm_rows = mod.read_csv(cm_path)
            merged = mod.merge_layers(policy_rows, cm_rows)
            summary = mod.compute_summary(merged)

            # valid_tiles: layer0=115584, layer1=115584, layer2=115584, layer3=115584
            # All equal weight, so weighted mean == arithmetic mean
            expected = (0.745 + 0.423 + 1.0 + 0.090) / 4
            assert abs(summary["weighted_mean_fp_score_fallback_fraction"] - expected) < 1e-3


class TestIndexerProxy:
    """Test indexer proxy computation."""

    def test_indexer_proxy_c025(self) -> None:
        mod = _import_summary()
        with tempfile.TemporaryDirectory() as td:
            policy_path = Path(td) / "policy.csv"
            cm_path = Path(td) / "certimask.csv"
            _write_policy_csv(policy_path)
            _write_certimask_csv(cm_path)

            policy_rows = mod.read_csv(policy_path)
            cm_rows = mod.read_csv(cm_path)
            merged = mod.merge_layers(policy_rows, cm_rows)

            for row in merged:
                fp_frac = row["indexer_fp_fallback_fraction"]
                expected = 0.25 * (1 - fp_frac) + 1.0 * fp_frac
                assert abs(row["indexer_proxy_c025"] - expected) < 1e-6

    def test_indexer_proxy_c050(self) -> None:
        mod = _import_summary()
        with tempfile.TemporaryDirectory() as td:
            policy_path = Path(td) / "policy.csv"
            cm_path = Path(td) / "certimask.csv"
            _write_policy_csv(policy_path)
            _write_certimask_csv(cm_path)

            policy_rows = mod.read_csv(policy_path)
            cm_rows = mod.read_csv(cm_path)
            merged = mod.merge_layers(policy_rows, cm_rows)

            for row in merged:
                fp_frac = row["indexer_fp_fallback_fraction"]
                expected = 0.50 * (1 - fp_frac) + 1.0 * fp_frac
                assert abs(row["indexer_proxy_c050"] - expected) < 1e-6


class TestHighFallbackDetection:
    """Test high fallback layer detection."""

    def test_high_fallback_layers(self) -> None:
        mod = _import_summary()
        with tempfile.TemporaryDirectory() as td:
            policy_path = Path(td) / "policy.csv"
            cm_path = Path(td) / "certimask.csv"
            _write_policy_csv(policy_path)
            _write_certimask_csv(cm_path)

            policy_rows = mod.read_csv(policy_path)
            cm_rows = mod.read_csv(cm_path)
            merged = mod.merge_layers(policy_rows, cm_rows)
            summary = mod.compute_summary(merged)

            # layer0=0.745 > 0.30, layer1=0.423 > 0.30
            assert 0 in summary["high_fallback_layers"]
            assert 1 in summary["high_fallback_layers"]
            assert 3 not in summary["high_fallback_layers"]


class TestReadiness:
    """Test readiness decision logic."""

    def test_ready_when_all_criteria_met(self) -> None:
        mod = _import_summary()
        # Build a merged list that meets all criteria
        merged = []
        for layer in range(24):
            merged.append({
                "layer": layer,
                "aglr_decision": "go_strict",
                "certimask_supported": True,
                "aggregation": "logsumexp",
                "target_sparsity": 0.75,
                "attention_work_fraction": 0.256,
                "kept_mass": 0.91,
                "cosine": 0.985,
                "l2": 0.10,
                "certimask_exact_match": True,
                "certimask_mismatch_count": 0,
                "certimask_fallback_rate": 0.10,
                "indexer_fp_fallback_fraction": 0.10,
                "indexer_proxy_c025": 0.325,
                "indexer_proxy_c050": 0.55,
                "runtime_path": "aglr_certimask_lowbit",
            })
        summary = mod.compute_summary(merged)
        assert summary["readiness"] == "ready_for_triton_prototype"

    def test_needs_fallback_cleanup(self) -> None:
        mod = _import_summary()
        merged = []
        for layer in range(24):
            merged.append({
                "layer": layer,
                "aglr_decision": "go_strict",
                "certimask_supported": True,
                "aggregation": "logsumexp",
                "target_sparsity": 0.75,
                "attention_work_fraction": 0.256,
                "kept_mass": 0.91,
                "cosine": 0.985,
                "l2": 0.10,
                "certimask_exact_match": True,
                "certimask_mismatch_count": 0,
                "certimask_fallback_rate": 0.40,  # high fallback
                "indexer_fp_fallback_fraction": 0.40,
                "indexer_proxy_c025": 0.55,
                "indexer_proxy_c050": 0.70,
                "runtime_path": "aglr_certimask_lowbit_high_fallback",
            })
        summary = mod.compute_summary(merged)
        assert summary["readiness"] == "needs_fallback_policy_cleanup"

    def test_needs_indexer_improvement(self) -> None:
        mod = _import_summary()
        merged = []
        for layer in range(24):
            merged.append({
                "layer": layer,
                "aglr_decision": "conditional_relaxed",
                "certimask_supported": True,
                "aggregation": "logsumexp",
                "target_sparsity": 0.5,
                "attention_work_fraction": 0.504,
                "kept_mass": 0.85,  # below 0.90
                "cosine": 0.99,
                "l2": 0.06,
                "certimask_exact_match": True,
                "certimask_mismatch_count": 0,
                "certimask_fallback_rate": 0.10,
                "indexer_fp_fallback_fraction": 0.10,
                "indexer_proxy_c025": 0.325,
                "indexer_proxy_c050": 0.55,
                "runtime_path": "aglr_certimask_lowbit",
            })
        summary = mod.compute_summary(merged)
        assert summary["readiness"] == "needs_indexer_improvement"


class TestRuntimePath:
    """Test runtime path assignment."""

    def test_unsupported_gets_fp_reference(self) -> None:
        mod = _import_summary()
        row = {
            "layer": 2, "aglr_decision": "go_practical",
            "certimask_supported": False, "aggregation": "topk_mean",
            "certimask_fallback_rate": 1.0,
            "attention_work_fraction": 0.504,
        }
        path = mod.assign_runtime_path(row)
        assert path == "unsupported_aggregation_fp_reference"

    def test_high_fallback_gets_high_fallback_path(self) -> None:
        mod = _import_summary()
        row = {
            "layer": 0, "aglr_decision": "conditional_relaxed",
            "certimask_supported": True, "aggregation": "logsumexp",
            "certimask_fallback_rate": 0.745,
            "attention_work_fraction": 0.504,
        }
        path = mod.assign_runtime_path(row)
        assert path == "aglr_certimask_lowbit_high_fallback"

    def test_normal_gets_lowbit_path(self) -> None:
        mod = _import_summary()
        row = {
            "layer": 3, "aglr_decision": "go_strict",
            "certimask_supported": True, "aggregation": "logsumexp",
            "certimask_fallback_rate": 0.090,
            "attention_work_fraction": 0.256,
        }
        path = mod.assign_runtime_path(row)
        assert path == "aglr_certimask_lowbit"

    def test_quality_only_high_work(self) -> None:
        mod = _import_summary()
        row = {
            "layer": 22, "aglr_decision": "quality_only_high_work",
            "certimask_supported": True, "aggregation": "logsumexp",
            "certimask_fallback_rate": 0.108,
            "attention_work_fraction": 0.707,
        }
        path = mod.assign_runtime_path(row)
        assert path == "aglr_certimask_lowbit_quality_only_high_attention_work"


class TestFinalLayerPolicyFields:
    """Test that final layer policy has all required fields."""

    def test_required_fields_present(self) -> None:
        mod = _import_summary()
        with tempfile.TemporaryDirectory() as td:
            policy_path = Path(td) / "policy.csv"
            cm_path = Path(td) / "certimask.csv"
            _write_policy_csv(policy_path)
            _write_certimask_csv(cm_path)

            policy_rows = mod.read_csv(policy_path)
            cm_rows = mod.read_csv(cm_path)
            merged = mod.merge_layers(policy_rows, cm_rows)

            required = [
                "layer", "aglr_decision", "certimask_supported", "aggregation",
                "target_sparsity", "attention_work_fraction", "kept_mass",
                "cosine", "l2", "certimask_exact_match", "certimask_mismatch_count",
                "certimask_fallback_rate", "indexer_fp_fallback_fraction",
                "runtime_path",
            ]
            for row in merged:
                for field in required:
                    assert field in row, f"Missing field {field} in layer {row['layer']}"
