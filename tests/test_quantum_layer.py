"""Tests for the QuantumLayer."""

import pytest

from quantum_neural_brain.neuron import Neuron
from quantum_neural_brain.quantum_layer import QuantumLayer, QuantumState


class TestQuantumState:
    def test_probability_of_ground_state(self):
        s = QuantumState("|0⟩", amplitude=complex(1.0, 0.0))
        assert s.probability == pytest.approx(1.0)

    def test_probability_of_excited_state(self):
        s = QuantumState("|1⟩", amplitude=complex(0.0, 1.0))
        assert s.probability == pytest.approx(1.0)

    def test_normalise(self):
        s = QuantumState("|0⟩", amplitude=complex(2.0, 0.0))
        s.normalise(4.0)
        assert s.probability == pytest.approx(1.0)


class TestQuantumLayer:
    def setup_method(self):
        self.neurons = [Neuron(f"n{i}", layer_name="hidden") for i in range(4)]
        self.ql = QuantumLayer(self.neurons, entanglement_pairs=[(0, 1)])

    def test_initial_state_vector(self):
        sv = self.ql.get_state_vector("n0")
        assert sv is not None
        assert len(sv) == 2
        labels = [label for label, _ in sv]
        assert "|0⟩" in labels and "|1⟩" in labels

    def test_unknown_neuron_returns_none(self):
        assert self.ql.get_state_vector("does_not_exist") is None

    def test_hadamard_creates_superposition(self):
        self.ql.apply_hadamard("n0")
        sv = self.ql.get_state_vector("n0")
        # After H, both amplitudes should be ~1/√2
        for _, amp in sv:
            assert abs(amp) == pytest.approx(1.0 / (2 ** 0.5), abs=1e-9)

    def test_measure_returns_bool(self):
        result = self.ql.measure("n0")
        assert isinstance(result, bool)

    def test_step_returns_list(self):
        fired = self.ql.step(dt_ms=1.0, noise_level=0.1)
        assert isinstance(fired, list)

    def test_interference_log_populated_after_step(self):
        self.ql.step(dt_ms=1.0)
        assert len(self.ql.interference_log) > 0

    def test_compute_interference_range(self):
        """Firing probability should be in [0, 1]."""
        for signal in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            p = self.ql.compute_interference("n0", signal)
            assert 0.0 <= p <= 1.0 + 1e-9

    def test_repr(self):
        r = repr(self.ql)
        assert "QuantumLayer" in r

    def test_cx_gate_does_not_crash(self):
        self.ql.apply_cx_gate("n0", "n1")

    def test_phase_rotation_does_not_crash(self):
        import math
        self.ql.apply_phase_rotation("n0", math.pi / 4)

    def test_total_quantum_collapses_increments(self):
        initial = self.ql.total_quantum_collapses
        # Force a collapse into |1⟩ by putting all amplitude there
        states = self.ql._states["n0"]
        states[0].amplitude = complex(0.0, 0.0)
        states[1].amplitude = complex(1.0, 0.0)
        result = self.ql.measure("n0")
        assert result is True
        assert self.ql.total_quantum_collapses == initial + 1
