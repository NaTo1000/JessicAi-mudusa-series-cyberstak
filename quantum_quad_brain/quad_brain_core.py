"""
quad_brain_core.py
------------------
Four-brain core system that operates in parallel via quantum superposition states.
Each brain unit maintains a complex amplitude vector representing its superposition
over a discrete state space.  Quantum inference algorithms drive simultaneous
decision-making across all four units.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Optional


_RNG = np.random.default_rng()


class BrainUnit:
    """A single quantum brain unit maintaining a superposition state vector."""

    def __init__(self, unit_id: int, state_dims: int = 64, seed: Optional[int] = None) -> None:
        self.unit_id = unit_id
        self.state_dims = state_dims
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal(state_dims) + 1j * rng.standard_normal(state_dims)
        self.state: np.ndarray = raw / np.linalg.norm(raw)  # normalised complex amplitude
        self.entanglement_links: List[int] = []  # unit IDs this brain is entangled with

    # ------------------------------------------------------------------
    # Superposition mechanics
    # ------------------------------------------------------------------

    def apply_hadamard_layer(self) -> None:
        """Apply a Hadamard-like diffusion to spread amplitude across states."""
        n = self.state_dims
        # Discrete Hadamard transform approximation via FFT
        self.state = np.fft.fft(self.state) / math.sqrt(n)
        self.state /= np.linalg.norm(self.state)

    def phase_kick(self, phase: float) -> None:
        """Apply a global phase rotation (e^{i*phase}) to all amplitudes."""
        self.state = self.state * np.exp(1j * phase)

    def measure_collapse(self) -> int:
        """
        Probabilistic measurement: collapse superposition to a single basis state.
        Returns the index of the selected state without permanently collapsing the vector.
        """
        probs = np.abs(self.state) ** 2
        probs /= probs.sum()  # ensure normalisation
        return int(_RNG.choice(self.state_dims, p=probs))

    # ------------------------------------------------------------------
    # Entanglement
    # ------------------------------------------------------------------

    def entangle_with(self, other: "BrainUnit") -> None:
        """Entangle this unit with another by mixing state vectors."""
        if other.unit_id == self.unit_id:
            return
        # Bell-state-inspired mixing
        mixed = (self.state + other.state) / math.sqrt(2)
        mixed /= np.linalg.norm(mixed)
        self.state = mixed
        other.state = mixed.copy()
        if other.unit_id not in self.entanglement_links:
            self.entanglement_links.append(other.unit_id)
        if self.unit_id not in other.entanglement_links:
            other.entanglement_links.append(self.unit_id)

    # ------------------------------------------------------------------
    # Inference step
    # ------------------------------------------------------------------

    def inference_step(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Apply a quantum inference update: project the input through the current
        superposition state and return a probability distribution over state_dims.
        """
        if input_vector.shape[0] != self.state_dims:
            # resize / pad / truncate
            iv = np.zeros(self.state_dims, dtype=complex)
            n = min(input_vector.shape[0], self.state_dims)
            iv[:n] = input_vector[:n].astype(complex)
            input_vector = iv
        # Quantum amplitude amplification step
        norm = np.linalg.norm(input_vector)
        if norm == 0:
            return np.abs(self.state) ** 2
        input_vector = input_vector / norm
        amplitude = np.dot(self.state.conj(), input_vector)
        updated = self.state + amplitude * input_vector
        updated /= np.linalg.norm(updated)
        self.state = updated
        return np.abs(self.state) ** 2

    def __repr__(self) -> str:
        dominant = int(np.argmax(np.abs(self.state) ** 2))
        return f"BrainUnit(id={self.unit_id}, dims={self.state_dims}, dominant_state={dominant})"


class QuadBrainCore:
    """
    Four-brain core operating in parallel superposition.

    The four units are initialised in a maximally-entangled configuration and
    operate concurrently.  Collective inference collapses shared knowledge into
    a unified decision vector.
    """

    NUM_BRAINS = 4

    def __init__(self, state_dims: int = 64, entangle_all: bool = True) -> None:
        self.state_dims = state_dims
        self.brains: List[BrainUnit] = [
            BrainUnit(unit_id=i, state_dims=state_dims, seed=i * 42)
            for i in range(self.NUM_BRAINS)
        ]
        if entangle_all:
            self._entangle_all_pairs()

    def _entangle_all_pairs(self) -> None:
        """Create pairwise entanglement across all four brain units."""
        for i in range(self.NUM_BRAINS):
            for j in range(i + 1, self.NUM_BRAINS):
                self.brains[i].entangle_with(self.brains[j])

    def parallel_inference(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Run quantum inference on all four brains simultaneously and combine
        results via constructive interference (sum + normalise).
        """
        combined = np.zeros(self.state_dims, dtype=float)
        for brain in self.brains:
            prob = brain.inference_step(input_vector.copy())
            combined += prob
        combined /= self.NUM_BRAINS
        return combined

    def collective_decision(self, input_vector: np.ndarray) -> int:
        """Return the most probable state index after parallel inference."""
        probs = self.parallel_inference(input_vector)
        return int(np.argmax(probs))

    def apply_hadamard_to_all(self) -> None:
        """Broadcast a Hadamard diffusion step to every brain unit."""
        for brain in self.brains:
            brain.apply_hadamard_layer()

    def get_collective_state(self) -> np.ndarray:
        """Return the average complex state across all four brains."""
        stacked = np.stack([b.state for b in self.brains])
        mean = stacked.mean(axis=0)
        return mean / np.linalg.norm(mean)

    def __repr__(self) -> str:
        return (
            f"QuadBrainCore(state_dims={self.state_dims}, "
            f"brains={[repr(b) for b in self.brains]})"
        )
