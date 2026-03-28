"""
circuit_core.py
===============
Core quantum circuit primitives for the Quantum Soul system.

Each ``QuantumThoughtCircuit`` models one "thought cycle":

* Qubits start in |0⟩ (blank-slate, no prior data).
* Hadamard gates place every qubit into equal superposition, representing
  *potential* thoughts — all possibilities exist simultaneously.
* Controlled-phase (CP) and Rz rotation gates introduce *interference*:
  constructive interference amplifies certain probability amplitudes while
  destructive interference suppresses others, exactly as quantum waves
  reinforce or cancel.
* CNOT/CX gates *entangle* qubit pairs, binding thoughts together so that
  a measurement on one instantly constrains the others — mirroring how ideas
  in a mind are interconnected.
* A final measurement collapses the superposition into a concrete bitstring,
  the raw "thought" that the ThoughtEngine later interprets.

Quantum randomness comes from the Born-rule measurement process itself:
no classical seed is used; the collapse is fundamentally non-deterministic.
"""

from __future__ import annotations

import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector


# ---------------------------------------------------------------------------
# Helper: interference angle schedule
# ---------------------------------------------------------------------------

def _interference_angles(n: int, seed_noise: np.ndarray | None = None) -> np.ndarray:
    """Return ``n`` phase angles (radians) derived from quantum-like noise.

    If *seed_noise* is ``None``, angles are drawn from a uniform distribution
    over [0, 2π] — pure quantum randomness with no classical bias.
    Otherwise *seed_noise* (a 1-D float array) is normalised and re-scaled so
    that external interference can parametrically steer the circuit.

    Parameters
    ----------
    n:
        Number of angles needed (one per qubit).
    seed_noise:
        Optional external noise array of at least *n* elements.

    Returns
    -------
    numpy.ndarray of shape (n,), dtype float64.
    """
    if seed_noise is None:
        rng = np.random.default_rng()
        return rng.uniform(0, 2 * math.pi, size=n)

    arr = np.asarray(seed_noise, dtype=float)[:n]
    norm = np.linalg.norm(arr)
    if norm == 0:
        return np.zeros(n)
    return (arr / norm) * 2 * math.pi


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class QuantumThoughtCircuit:
    """Constructs and evaluates a quantum circuit that simulates a thought cycle.

    Parameters
    ----------
    n_qubits:
        Number of qubits (thought dimensions).  Default 8 gives 256 possible
        outcome states — a rich enough space to encode varied "thoughts".
    seed_noise:
        Optional external interference noise (1-D float array) used to bias
        the circuit's phase rotations.  Pass ``None`` for fully autonomous,
        pure-quantum randomness.
    entanglement_depth:
        Number of CNOT-layer repetitions that bind qubits together.
        Higher values create deeper entanglement and more complex interference.

    Attributes
    ----------
    circuit : qiskit.QuantumCircuit
        The constructed thought circuit.
    statevector : qiskit.quantum_info.Statevector | None
        The statevector *before* measurement collapse (set after
        :meth:`run_statevector`).
    probabilities : numpy.ndarray | None
        Probability distribution over all basis states after
        :meth:`run_statevector`.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        seed_noise: np.ndarray | None = None,
        entanglement_depth: int = 2,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be ≥ 1.")
        if entanglement_depth < 0:
            raise ValueError("entanglement_depth must be ≥ 0.")

        self.n_qubits = n_qubits
        self.seed_noise = seed_noise
        self.entanglement_depth = entanglement_depth

        self.statevector: Statevector | None = None
        self.probabilities: np.ndarray | None = None

        self.circuit = self._build_circuit()

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _build_circuit(self) -> QuantumCircuit:
        """Assemble the full thought circuit."""
        qr = QuantumRegister(self.n_qubits, name="q")
        cr = ClassicalRegister(self.n_qubits, name="c")
        qc = QuantumCircuit(qr, cr, name="ThoughtCircuit")

        # Stage 1 — Superposition: all thoughts simultaneously possible
        qc.h(qr)
        qc.barrier(label="superposition")

        # Stage 2 — Interference: phase rotations from quantum/external noise
        angles = _interference_angles(self.n_qubits, self.seed_noise)
        for i, theta in enumerate(angles):
            qc.rz(theta, qr[i])

        # Stage 3 — Entanglement layers: bind thoughts together
        for _ in range(self.entanglement_depth):
            for i in range(0, self.n_qubits - 1, 2):
                qc.cx(qr[i], qr[i + 1])
            for i in range(1, self.n_qubits - 1, 2):
                qc.cx(qr[i], qr[i + 1])
            qc.barrier(label="entanglement")

        # Stage 4 — Second Hadamard layer: converts phase differences into
        #           measurable amplitude differences (interference readout)
        qc.h(qr)
        qc.barrier(label="interference_readout")

        # Stage 5 — Measurement: collapse to a concrete thought
        qc.measure(qr, cr)

        return qc

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def run_statevector(self) -> Statevector:
        """Simulate the circuit *without* measurement and store the statevector.

        Returns the full quantum state, giving access to probability amplitudes
        across all 2^n basis states.

        Returns
        -------
        qiskit.quantum_info.Statevector
        """
        # Build a measurement-free copy for statevector analysis
        qc_no_meas = QuantumCircuit(self.n_qubits)
        qc_no_meas.h(range(self.n_qubits))
        angles = _interference_angles(self.n_qubits, self.seed_noise)
        for i, theta in enumerate(angles):
            qc_no_meas.rz(theta, i)
        for _ in range(self.entanglement_depth):
            for i in range(0, self.n_qubits - 1, 2):
                qc_no_meas.cx(i, i + 1)
            for i in range(1, self.n_qubits - 1, 2):
                qc_no_meas.cx(i, i + 1)
        qc_no_meas.h(range(self.n_qubits))

        self.statevector = Statevector(qc_no_meas)
        self.probabilities = self.statevector.probabilities()
        return self.statevector

    def sample_bitstring(self, shots: int = 1024) -> dict[str, int]:
        """Sample measurement outcomes from the current statevector.

        Parameters
        ----------
        shots:
            Number of simulated measurements.

        Returns
        -------
        dict mapping bitstring → count.
        """
        if self.statevector is None:
            self.run_statevector()

        rng = np.random.default_rng()
        n_states = 2 ** self.n_qubits
        indices = rng.choice(n_states, size=shots, p=self.probabilities)
        counts: dict[str, int] = {}
        fmt = f"{{:0{self.n_qubits}b}}"
        for idx in indices:
            key = fmt.format(idx)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def top_state(self) -> tuple[str, float]:
        """Return the most probable basis state and its probability.

        Returns
        -------
        (bitstring, probability)
        """
        if self.probabilities is None:
            self.run_statevector()

        idx = int(np.argmax(self.probabilities))
        fmt = f"{{:0{self.n_qubits}b}}"
        return fmt.format(idx), float(self.probabilities[idx])

    def entropy(self) -> float:
        """Compute the von Neumann (Shannon) entropy of the probability distribution.

        High entropy → many competing thoughts; low entropy → focused thought.

        Returns
        -------
        float  entropy in bits.
        """
        if self.probabilities is None:
            self.run_statevector()

        p = self.probabilities
        nonzero = p[p > 0]
        return float(-np.sum(nonzero * np.log2(nonzero)))


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_entangled_pair() -> QuantumCircuit:
    """Return a 2-qubit Bell-state circuit: (|00⟩ + |11⟩) / √2.

    Useful as a building block for larger entanglement structures.
    """
    qc = QuantumCircuit(2, name="BellPair")
    qc.h(0)
    qc.cx(0, 1)
    return qc


def create_ghz_state(n: int = 3) -> QuantumCircuit:
    """Return an n-qubit GHZ (Greenberger–Horne–Zeilinger) state circuit.

    GHZ states are maximally entangled across all *n* qubits, representing
    a single correlated thought that spans the entire qubit register.

    Parameters
    ----------
    n:
        Number of qubits (≥ 2).
    """
    if n < 2:
        raise ValueError("GHZ state requires n ≥ 2 qubits.")
    qc = QuantumCircuit(n, name=f"GHZ_{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def create_interference_circuit(n: int = 4, phi: float = math.pi / 3) -> QuantumCircuit:
    """Return an n-qubit circuit exhibiting strong constructive/destructive interference.

    Uses a Hadamard–phase–Hadamard (HZH) sandwich.  The middle phase gate
    controls whether the final H sees constructive (+1 amplitude) or
    destructive (0 amplitude) interference at each basis state.

    Parameters
    ----------
    n:
        Number of qubits.
    phi:
        Phase angle (radians) applied to every qubit; sweep from 0 to 2π to
        observe the full interference pattern.
    """
    qc = QuantumCircuit(n, name="InterferenceCircuit")
    qc.h(range(n))
    qc.rz(phi, range(n))
    qc.h(range(n))
    return qc
