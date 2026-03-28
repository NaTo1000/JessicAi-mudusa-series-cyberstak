"""Integration tests for ThoughtProcessor (end-to-end pipeline)."""

import pytest
from nonganon_tesseract.thought_processor import ThoughtProcessor


def _make_data(n=30, dims=3):
    return [[float((i * j + 1) % 10) for j in range(1, dims + 1)] for i in range(n)]


class TestThoughtProcessor:
    def test_process_returns_required_keys(self):
        tp = ThoughtProcessor(n_outcomes=3, seed=0)
        result = tp.process(_make_data())
        for key in ("decision", "inference_state", "cluster_summary",
                    "diagnostics", "vault_manifest"):
            assert key in result

    def test_decision_is_normalised(self):
        tp = ThoughtProcessor(n_outcomes=3, seed=1)
        result = tp.process(_make_data())
        total = sum(result["decision"])
        assert abs(total - 1.0) < 1e-9

    def test_inference_state_correct_length(self):
        tp = ThoughtProcessor(n_outcomes=4, seed=2)
        result = tp.process(_make_data())
        assert len(result["inference_state"]) == 4

    def test_cluster_summary_has_meta_centroid(self):
        tp = ThoughtProcessor(n_outcomes=3, seed=3)
        result = tp.process(_make_data())
        assert result["cluster_summary"]["meta_centroid"] is not None

    def test_vault_manifest_has_result_entry(self):
        tp = ThoughtProcessor(n_outcomes=3, seed=4)
        result = tp.process(_make_data())
        assert "result" in result["vault_manifest"]
        assert result["vault_manifest"]["result"]["sealed"] is True

    def test_vault_passphrase_protected(self):
        tp = ThoughtProcessor(
            n_outcomes=3, seed=5, vault_passphrase="secret"
        )
        result = tp.process(_make_data())
        assert result["vault_manifest"]["result"]["passphrase_protected"] is True

    def test_topology_returns_string(self):
        tp = ThoughtProcessor()
        topo = tp.topology()
        assert isinstance(topo, str)
        assert "TesseractOctagon" in topo

    def test_reproducible_with_seed(self):
        data = _make_data(30)
        tp1 = ThoughtProcessor(n_outcomes=3, seed=99)
        r1 = tp1.process(data)
        tp2 = ThoughtProcessor(n_outcomes=3, seed=99)
        r2 = tp2.process(data)
        assert r1["cluster_summary"]["meta_centroid"] == r2["cluster_summary"]["meta_centroid"]

    def test_diagnostics_has_twinbrain_info(self):
        tp = ThoughtProcessor(n_outcomes=3, seed=0)
        result = tp.process(_make_data())
        diag = result["diagnostics"]
        assert "brain_a" in diag
        assert "brain_b" in diag

    def test_repr(self):
        tp = ThoughtProcessor(n_outcomes=5, alpha=0.3)
        r = repr(tp)
        assert "ThoughtProcessor" in r
        assert "n_outcomes=5" in r
