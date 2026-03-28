"""
tests/test_circuit_core.py
==========================
Unit tests for quantum_soul.circuit_core.
"""

import math
import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_soul.circuit_core import (
    QuantumThoughtCircuit,
    _interference_angles,
    create_entangled_pair,
    create_ghz_state,
    create_interference_circuit,
)


# ---------------------------------------------------------------------------
# _interference_angles
# ---------------------------------------------------------------------------

class TestInterferenceAngles:
    def test_autonomous_shape(self):
        angles = _interference_angles(8)
        assert angles.shape == (8,)

    def test_autonomous_range(self):
        angles = _interference_angles(10)
        assert np.all(angles >= 0) and np.all(angles <= 2 * math.pi)

    def test_seeded_normalization(self):
        noise = np.array([1.0, 2.0, 3.0, 4.0])
        angles = _interference_angles(4, seed_noise=noise)
        assert angles.shape == (4,)
        # max abs value should be <= 2π
        assert np.all(np.abs(angles) <= 2 * math.pi + 1e-9)

    def test_zero_noise_returns_zeros(self):
        noise = np.zeros(4)
        angles = _interference_angles(4, seed_noise=noise)
        assert np.allclose(angles, 0)


# ---------------------------------------------------------------------------
# QuantumThoughtCircuit
# ---------------------------------------------------------------------------

class TestQuantumThoughtCircuit:
    def test_default_construction(self):
        qtc = QuantumThoughtCircuit()
        assert qtc.n_qubits == 8
        assert qtc.entanglement_depth == 2

    def test_invalid_n_qubits(self):
        with pytest.raises(ValueError):
            QuantumThoughtCircuit(n_qubits=0)

    def test_invalid_entanglement_depth(self):
        with pytest.raises(ValueError):
            QuantumThoughtCircuit(n_qubits=4, entanglement_depth=-1)

    def test_circuit_is_qiskit_circuit(self):
        qtc = QuantumThoughtCircuit(n_qubits=4)
        assert isinstance(qtc.circuit, QuantumCircuit)

    def test_circuit_has_correct_qubit_count(self):
        for n in [2, 4, 6, 8]:
            qtc = QuantumThoughtCircuit(n_qubits=n)
            assert qtc.circuit.num_qubits == n

    def test_run_statevector_returns_statevector(self):
        qtc = QuantumThoughtCircuit(n_qubits=4)
        sv = qtc.run_statevector()
        assert isinstance(sv, Statevector)

    def test_probabilities_sum_to_one(self):
        qtc = QuantumThoughtCircuit(n_qubits=4)
        qtc.run_statevector()
        assert abs(qtc.probabilities.sum() - 1.0) < 1e-6

    def test_probabilities_nonnegative(self):
        qtc = QuantumThoughtCircuit(n_qubits=4)
        qtc.run_statevector()
        assert np.all(qtc.probabilities >= 0)

    def test_probabilities_correct_length(self):
        for n in [2, 4, 6]:
            qtc = QuantumThoughtCircuit(n_qubits=n)
            qtc.run_statevector()
            assert len(qtc.probabilities) == 2 ** n

    def test_top_state_returns_tuple(self):
        qtc = QuantumThoughtCircuit(n_qubits=4)
        qtc.run_statevector()
        top, prob = qtc.top_state()
        assert isinstance(top, str) and len(top) == 4
        assert 0.0 <= prob <= 1.0

    def test_entropy_is_nonneg(self):
        qtc = QuantumThoughtCircuit(n_qubits=4)
        qtc.run_statevector()
        H = qtc.entropy()
        assert H >= 0.0

    def test_entropy_bounded_by_n_qubits(self):
        n = 4
        qtc = QuantumThoughtCircuit(n_qubits=n)
        qtc.run_statevector()
        H = qtc.entropy()
        assert H <= n + 1e-6   # allow tiny floating-point slack

    def test_sample_bitstring_counts(self):
        qtc = QuantumThoughtCircuit(n_qubits=4)
        qtc.run_statevector()
        counts = qtc.sample_bitstring(shots=200)
        total = sum(counts.values())
        assert total == 200

    def test_sample_bitstring_valid_keys(self):
        n = 4
        qtc = QuantumThoughtCircuit(n_qubits=n)
        qtc.run_statevector()
        counts = qtc.sample_bitstring(shots=100)
        for key in counts:
            assert len(key) == n
            assert all(c in "01" for c in key)

    def test_seeded_noise_reproducibility(self):
        """Same noise seed should produce the same top state."""
        noise = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        qtc1 = QuantumThoughtCircuit(n_qubits=8, seed_noise=noise)
        qtc2 = QuantumThoughtCircuit(n_qubits=8, seed_noise=noise)
        sv1 = qtc1.run_statevector()
        sv2 = qtc2.run_statevector()
        np.testing.assert_allclose(qtc1.probabilities, qtc2.probabilities)

    def test_entanglement_depth_zero(self):
        qtc = QuantumThoughtCircuit(n_qubits=4, entanglement_depth=0)
        qtc.run_statevector()
        assert qtc.probabilities is not None


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

class TestFactoryHelpers:
    def test_bell_pair_is_2_qubit(self):
        qc = create_entangled_pair()
        assert qc.num_qubits == 2

    def test_ghz_n_qubits(self):
        for n in [2, 3, 5]:
            qc = create_ghz_state(n)
            assert qc.num_qubits == n

    def test_ghz_invalid(self):
        with pytest.raises(ValueError):
            create_ghz_state(1)

    def test_interference_circuit_n_qubits(self):
        for n in [2, 4, 6]:
            qc = create_interference_circuit(n)
            assert qc.num_qubits == n

    def test_ghz_statevector_amplitudes(self):
        """GHZ state should give equal weight to |00…0⟩ and |11…1⟩."""
        n = 3
        qc = create_ghz_state(n)
        sv = Statevector(qc)
        probs = sv.probabilities()
        # Only states 0 (000) and 7 (111) should have nonzero probability
        nonzero_indices = np.where(probs > 1e-6)[0]
        assert set(nonzero_indices) == {0, 2**n - 1}
        assert abs(probs[0] - 0.5) < 1e-6
        assert abs(probs[2**n - 1] - 0.5) < 1e-6
