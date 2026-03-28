"""Tests for QuantumInferenceEngine."""

import math
import pytest
from nonganon_tesseract.quantum_inference import QuantumInferenceEngine


class TestQuantumInferenceEngine:
    def test_init_uniform_prior(self):
        engine = QuantumInferenceEngine(n_outcomes=4)
        state = engine.state
        assert len(state) == 4
        assert all(abs(v - 0.25) < 1e-9 for v in state)

    def test_init_custom_seed_normalised(self):
        engine = QuantumInferenceEngine(n_outcomes=3, seed_state=[2.0, 2.0, 2.0])
        assert all(abs(v - 1 / 3) < 1e-9 for v in engine.state)

    def test_init_invalid_n_outcomes(self):
        with pytest.raises(ValueError):
            QuantumInferenceEngine(n_outcomes=1)

    def test_init_invalid_alpha(self):
        with pytest.raises(ValueError):
            QuantumInferenceEngine(alpha=1.5)

    def test_init_seed_wrong_length(self):
        with pytest.raises(ValueError):
            QuantumInferenceEngine(n_outcomes=3, seed_state=[0.5, 0.5])

    def test_update_returns_normalised_distribution(self):
        engine = QuantumInferenceEngine(n_outcomes=3, alpha=0.5)
        new_state = engine.update([0.0, 1.0, 0.0])
        assert abs(sum(new_state) - 1.0) < 1e-9

    def test_update_negative_evidence_raises(self):
        engine = QuantumInferenceEngine(n_outcomes=3)
        with pytest.raises(ValueError):
            engine.update([-0.1, 0.5, 0.6])

    def test_update_wrong_length_raises(self):
        engine = QuantumInferenceEngine(n_outcomes=3)
        with pytest.raises(ValueError):
            engine.update([0.5, 0.5])

    def test_update_increments_steps(self):
        engine = QuantumInferenceEngine(n_outcomes=2)
        engine.update([1.0, 0.0])
        engine.update([0.5, 0.5])
        assert engine._steps == 2

    def test_most_probable_outcome(self):
        engine = QuantumInferenceEngine(n_outcomes=4, alpha=1.0)
        engine.update([0.0, 0.0, 1.0, 0.0])
        assert engine.most_probable_outcome() == 2

    def test_entropy_uniform_is_maximum(self):
        engine = QuantumInferenceEngine(n_outcomes=4)
        h = engine.entropy()
        assert abs(h - math.log(4)) < 1e-6

    def test_entropy_certain_is_zero(self):
        engine = QuantumInferenceEngine(n_outcomes=4, alpha=1.0)
        engine.update([0.0, 0.0, 0.0, 1.0])
        h = engine.entropy()
        assert h < 1e-9

    def test_fingerprint_is_hex(self):
        engine = QuantumInferenceEngine(n_outcomes=3)
        fp = engine.fingerprint()
        assert len(fp) == 64
        int(fp, 16)

    def test_infer_clusters_updates_state(self):
        engine = QuantumInferenceEngine(n_outcomes=4, alpha=0.5)
        original = engine.state[:]
        summary = {
            "clusters": [
                {"centroid": [1.0, 2.0], "members": 10},
                {"centroid": [3.0, 4.0], "members": 20},
                {"centroid": [5.0, 6.0], "members": 5},
            ]
        }
        new_state = engine.infer_clusters(summary)
        assert new_state != original

    def test_infer_clusters_empty_summary(self):
        engine = QuantumInferenceEngine(n_outcomes=3)
        original = engine.state[:]
        result = engine.infer_clusters({"clusters": []})
        assert result == original
