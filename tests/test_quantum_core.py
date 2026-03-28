"""Tests for the Quantum Inference Engine and Quad-Brain coordinator."""

import pytest

from quantum_quad_brain.quantum_core.inference_engine import (
    QuantumInferenceEngine,
    QuantumState,
    InferenceResult,
)
from quantum_quad_brain.quantum_core.quad_brain import QuantumQuadBrain, BrainConfig


# ---------------------------------------------------------------------------
# QuantumState
# ---------------------------------------------------------------------------

class TestQuantumState:
    def test_uniform_superposition(self):
        state = QuantumState(n_qubits=2)
        probs = state.probabilities
        assert len(probs) == 4
        assert all(abs(p - 0.25) < 1e-9 for p in probs)

    def test_probabilities_sum_to_one(self):
        state = QuantumState(n_qubits=4)
        state.apply_hadamard(0)
        state.apply_phase(1, 1.57)
        assert sum(state.probabilities) == pytest.approx(1.0, abs=1e-6)

    def test_measure_returns_valid_index(self):
        state = QuantumState(n_qubits=3)
        outcome = state.measure()
        assert 0 <= outcome < 8


# ---------------------------------------------------------------------------
# QuantumInferenceEngine
# ---------------------------------------------------------------------------

class TestQuantumInferenceEngine:
    def test_infer_returns_result(self):
        engine = QuantumInferenceEngine(n_qubits=4, shots=16, seed=42)
        result = engine.infer(b"hello world")
        assert isinstance(result, InferenceResult)

    def test_confidence_in_range(self):
        engine = QuantumInferenceEngine(n_qubits=4, shots=64, seed=0)
        result = engine.infer(b"test data")
        assert 0.0 <= result.confidence <= 1.0

    def test_predicted_class_valid(self):
        engine = QuantumInferenceEngine(n_qubits=4, shots=32, seed=7)
        result = engine.infer(b"abc")
        assert 0 <= result.predicted_class < 2 ** 4

    def test_latency_positive(self):
        engine = QuantumInferenceEngine(n_qubits=4, shots=16, seed=1)
        result = engine.infer(b"x")
        assert result.latency_ms >= 0.0

    def test_deterministic_with_seed(self):
        e1 = QuantumInferenceEngine(n_qubits=4, shots=128, seed=99)
        e2 = QuantumInferenceEngine(n_qubits=4, shots=128, seed=99)
        r1 = e1.infer(b"same_input")
        r2 = e2.infer(b"same_input")
        assert r1.predicted_class == r2.predicted_class

    def test_invalid_qubits_raises(self):
        with pytest.raises(ValueError, match="n_qubits"):
            QuantumInferenceEngine(n_qubits=1)

    def test_avg_latency_accumulates(self):
        engine = QuantumInferenceEngine(n_qubits=4, shots=8, seed=0)
        engine.infer(b"a")
        engine.infer(b"b")
        assert engine.avg_latency_ms > 0.0

    def test_engine_summary(self):
        engine = QuantumInferenceEngine(n_qubits=4, shots=16, seed=0)
        engine.infer(b"data")
        s = engine.engine_summary()
        assert s["inferences_run"] == 1
        assert "avg_latency_ms" in s


# ---------------------------------------------------------------------------
# QuantumQuadBrain
# ---------------------------------------------------------------------------

def _make_quad_brain() -> QuantumQuadBrain:
    configs = [
        BrainConfig("alpha",  n_qubits=4, shots=8,  seed=1),
        BrainConfig("beta",   n_qubits=4, shots=8,  seed=2),
        BrainConfig("gamma",  n_qubits=4, shots=8,  seed=3),
        BrainConfig("delta",  n_qubits=4, shots=8,  seed=4),
    ]
    return QuantumQuadBrain(brain_configs=configs, n_cm4_nodes=2, dram_capacity_gb=0.001)


class TestQuantumQuadBrain:
    def test_infer_default_brain(self):
        qb = _make_quad_brain()
        result = qb.infer(b"payload")
        assert isinstance(result, InferenceResult)

    def test_infer_named_brain(self):
        qb = _make_quad_brain()
        result = qb.infer(b"data", brain_id="alpha")
        assert result is not None

    def test_infer_unknown_brain_raises(self):
        qb = _make_quad_brain()
        with pytest.raises(KeyError):
            qb.infer(b"x", brain_id="omega")

    def test_parallel_infer_returns_four_results(self):
        qb = _make_quad_brain()
        results = qb.parallel_infer(b"multi")
        assert len(results) == 4

    def test_consensus_infer_returns_class(self):
        qb = _make_quad_brain()
        consensus = qb.consensus_infer(b"vote_me")
        assert "consensus_class" in consensus
        assert "consensus_confidence" in consensus

    def test_caching_hit(self):
        qb = _make_quad_brain()
        r1 = qb.infer(b"cache_test")
        r2 = qb.infer(b"cache_test")
        # Same input → same class
        assert r1.predicted_class == r2.predicted_class

    def test_persist_and_retrieve(self):
        from quantum_quad_brain.nvme_dram.storage_pipeline import NVMeDevice
        from pathlib import Path
        qb = _make_quad_brain()
        qb.storage.add_device(
            NVMeDevice(device_id="test_nvme", path=Path("/tmp"), capacity_gb=100.0)
        )
        data = b"nvme_test_payload"
        key = qb.persist("mykey", data)
        retrieved = qb.retrieve(key, size_hint=len(data))
        assert isinstance(retrieved, bytes)

    def test_wrong_brain_count_raises(self):
        with pytest.raises(ValueError, match="exactly 4"):
            QuantumQuadBrain(
                brain_configs=[BrainConfig("only_one", n_qubits=4, shots=8)],
            )

    def test_system_summary_structure(self):
        qb = _make_quad_brain()
        s = qb.system_summary()
        for key in ("brains", "storage", "cache", "cluster", "distributor"):
            assert key in s
