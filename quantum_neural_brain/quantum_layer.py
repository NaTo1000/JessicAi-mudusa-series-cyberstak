"""
quantum_layer.py — Quantum-mechanical enhancements for neural processing.

Simulates quantum phenomena that influence neural computation:
- Superposition: a neuron can hold multiple potential states simultaneously.
- Entanglement: correlated firing between distant neurons.
- Interference: constructive/destructive wave-function collapse determines output.
- Quantum tunneling: sub-threshold signals occasionally cross the firing barrier.

These mechanisms introduce a layer of probabilistic richness beyond classical
stochastic neurons, enabling the system to explore a wider solution space.
"""

import math
import random
import cmath
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from .neuron import Neuron


@dataclass
class QuantumState:
    """
    Quantum wave-function representation of a neuron's potential states.

    The amplitude is a complex number; its squared magnitude gives the
    probability of observing that state upon measurement (Born rule).
    """
    label: str
    amplitude: complex = field(default_factory=lambda: complex(1.0, 0.0))

    @property
    def probability(self) -> float:
        """Born rule: P = |amplitude|²"""
        return abs(self.amplitude) ** 2

    def normalise(self, norm: float) -> None:
        """Normalise amplitude so all state probabilities sum to 1."""
        if norm > 0:
            self.amplitude /= math.sqrt(norm)


class QuantumLayer:
    """
    Quantum processing layer wrapping classical neurons.

    Each neuron in this layer has an associated quantum state vector.
    During each timestep:
    1. States are evolved by applying quantum gates (Hadamard, phase rotation).
    2. Interference between states is computed.
    3. Collapse (measurement) determines if the neuron fires.
    4. Entangled pairs exhibit correlated firing probabilities.
    """

    def __init__(self, neurons: List[Neuron], entanglement_pairs: Optional[List[Tuple[int, int]]] = None):
        self.neurons = neurons
        self.entanglement_pairs: List[Tuple[int, int]] = entanglement_pairs or []

        # Each neuron has two basis states: |0⟩ (resting) and |1⟩ (firing)
        self._states: Dict[str, List[QuantumState]] = {}
        for neuron in neurons:
            self._states[neuron.neuron_id] = [
                QuantumState(label="|0⟩", amplitude=complex(1.0, 0.0)),
                QuantumState(label="|1⟩", amplitude=complex(0.0, 0.0)),
            ]

        # Interference history for debugging/visualisation
        self.interference_log: List[Dict] = []

        # Metrics
        self.total_quantum_collapses: int = 0
        self.entangled_firing_events: int = 0

    # ------------------------------------------------------------------
    # Gate operations
    # ------------------------------------------------------------------

    def apply_hadamard(self, neuron_id: str) -> None:
        """
        Apply a Hadamard gate, placing the neuron in equal superposition.

        H|0⟩ = (|0⟩ + |1⟩) / √2
        H|1⟩ = (|0⟩ − |1⟩) / √2
        """
        states = self._states.get(neuron_id)
        if states is None:
            return
        a0, a1 = states[0].amplitude, states[1].amplitude
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        states[0].amplitude = (a0 + a1) * inv_sqrt2
        states[1].amplitude = (a0 - a1) * inv_sqrt2

    def apply_phase_rotation(self, neuron_id: str, theta: float) -> None:
        """
        Apply a phase rotation gate R(θ) to the |1⟩ state.

        This encodes the neuron's membrane potential as a quantum phase,
        allowing interference to amplify near-threshold signals.
        """
        states = self._states.get(neuron_id)
        if states is None:
            return
        states[1].amplitude *= cmath.exp(1j * theta)

    def apply_cx_gate(self, control_id: str, target_id: str) -> None:
        """
        Controlled-NOT (CX/CNOT) gate for entanglement.

        Flips the target qubit when the control qubit is in state |1⟩.
        This creates quantum entanglement between two neurons.
        """
        ctrl_states = self._states.get(control_id)
        tgt_states = self._states.get(target_id)
        if ctrl_states is None or tgt_states is None:
            return
        # If control is in |1⟩, swap target amplitudes
        if abs(ctrl_states[1].amplitude) > 0:
            tgt_states[0].amplitude, tgt_states[1].amplitude = (
                tgt_states[1].amplitude,
                tgt_states[0].amplitude,
            )

    # ------------------------------------------------------------------
    # Interference and measurement
    # ------------------------------------------------------------------

    def compute_interference(self, neuron_id: str, external_signal: float = 0.0) -> float:
        """
        Compute constructive/destructive interference for a neuron.

        The external_signal (normalised −1 to 1) is encoded as a phase
        that either constructively or destructively interferes with the
        current quantum state.

        Returns the effective firing probability after interference.
        """
        states = self._states.get(neuron_id)
        if states is None:
            return 0.0

        # Encode external signal as phase
        phase = math.pi * external_signal
        signal_amplitude = complex(math.cos(phase), math.sin(phase))

        # Interference: |ψ + signal|² for the |1⟩ state
        interfered = states[1].amplitude + signal_amplitude * 0.3
        probability = abs(interfered) ** 2

        # Normalise to [0, 1]
        total = abs(states[0].amplitude) ** 2 + abs(interfered) ** 2
        if total > 0:
            probability /= total

        self.interference_log.append({
            "neuron_id": neuron_id,
            "probability": probability,
            "external_signal": external_signal,
        })

        # Keep only the last 500 log entries
        if len(self.interference_log) > 500:
            self.interference_log = self.interference_log[-500:]

        return probability

    def measure(self, neuron_id: str) -> bool:
        """
        Perform a quantum measurement (wave-function collapse).

        Returns True if the neuron collapses into the |1⟩ (firing) state.
        After measurement, the state is reset to definite |0⟩ or |1⟩.
        """
        states = self._states.get(neuron_id)
        if states is None:
            return False

        # Normalise the state vector
        norm = sum(abs(s.amplitude) ** 2 for s in states)
        if norm > 0:
            for s in states:
                s.normalise(norm)

        p_fire = abs(states[1].amplitude) ** 2
        fired = random.random() < p_fire

        # Collapse
        if fired:
            states[0].amplitude = complex(0.0, 0.0)
            states[1].amplitude = complex(1.0, 0.0)
            self.total_quantum_collapses += 1
        else:
            states[0].amplitude = complex(1.0, 0.0)
            states[1].amplitude = complex(0.0, 0.0)

        return fired

    # ------------------------------------------------------------------
    # High-level step
    # ------------------------------------------------------------------

    def step(self, dt_ms: float = 1.0, noise_level: float = 0.1) -> List[str]:
        """
        Advance the quantum layer by one timestep.

        Steps:
        1. Apply Hadamard to all neurons to create superposition.
        2. Encode membrane potential as quantum phase.
        3. Apply entanglement gates for correlated pairs.
        4. Add interference noise (simulates thought variability).
        5. Measure/collapse states, triggering force_fire if needed.

        Returns list of neuron IDs that fired this step.
        """
        fired_ids = []

        # Step 1 & 2: Superposition + phase encoding
        for neuron in self.neurons:
            self.apply_hadamard(neuron.neuron_id)
            # Encode membrane potential as phase: theta ∝ (Vm - resting)
            config = neuron.config
            vm_normalised = (neuron.membrane_potential - config.resting_potential) / (
                config.threshold_potential - config.resting_potential
            )
            theta = math.pi * vm_normalised
            self.apply_phase_rotation(neuron.neuron_id, theta)

        # Step 3: Entanglement gates
        for (i, j) in self.entanglement_pairs:
            if i < len(self.neurons) and j < len(self.neurons):
                self.apply_cx_gate(
                    self.neurons[i].neuron_id,
                    self.neurons[j].neuron_id,
                )

        # Step 4 & 5: Interference noise + measurement
        for neuron in self.neurons:
            noise = random.gauss(0.0, noise_level)
            p_fire = self.compute_interference(neuron.neuron_id, noise)
            fired = self.measure(neuron.neuron_id)

            if fired and not neuron.is_refractory:
                neuron.force_fire()
                fired_ids.append(neuron.neuron_id)

        # Handle entangled pair correlated firing
        for (i, j) in self.entanglement_pairs:
            if i < len(self.neurons) and j < len(self.neurons):
                ni = self.neurons[i]
                nj = self.neurons[j]
                if ni.neuron_id in fired_ids and nj.neuron_id not in fired_ids:
                    if not nj.is_refractory and random.random() < 0.6:
                        nj.force_fire()
                        fired_ids.append(nj.neuron_id)
                        self.entangled_firing_events += 1

        return fired_ids

    def get_state_vector(self, neuron_id: str) -> Optional[List[Tuple[str, complex]]]:
        """Return the current quantum state vector for a neuron."""
        states = self._states.get(neuron_id)
        if states is None:
            return None
        return [(s.label, s.amplitude) for s in states]

    def __repr__(self) -> str:
        return (
            f"QuantumLayer(neurons={len(self.neurons)}, "
            f"entangled_pairs={len(self.entanglement_pairs)}, "
            f"collapses={self.total_quantum_collapses})"
        )
