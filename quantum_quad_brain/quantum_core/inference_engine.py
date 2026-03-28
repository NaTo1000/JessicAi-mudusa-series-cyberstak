"""
Quantum Inference Engine
========================
Simulates a quantum-inspired inference pipeline using superposition
sampling and interference-based probability amplitude collapse for
the Quad-Brain decision making system.

Each inference call maps an input vector onto a simulated quantum
state register and returns the most probable output class along with
confidence amplitudes.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QuantumState:
    """A simulated quantum register holding probability amplitudes."""

    n_qubits: int
    amplitudes: List[complex] = field(init=False)

    def __post_init__(self) -> None:
        n_states = 2 ** self.n_qubits
        # Initialise to uniform superposition  |+⟩^n
        amp = complex(1.0 / math.sqrt(n_states), 0.0)
        self.amplitudes = [amp] * n_states

    def apply_hadamard(self, qubit: int) -> None:
        """Apply a Hadamard gate to the specified qubit index."""
        n_states = len(self.amplitudes)
        step = 2 ** (qubit + 1)
        h = 1.0 / math.sqrt(2)
        for i in range(0, n_states, step):
            for j in range(i, i + step // 2):
                a, b = self.amplitudes[j], self.amplitudes[j + step // 2]
                self.amplitudes[j] = h * (a + b)
                self.amplitudes[j + step // 2] = h * (a - b)

    def apply_phase(self, qubit: int, theta: float) -> None:
        """Apply a phase-shift gate R(θ) to the specified qubit."""
        n_states = len(self.amplitudes)
        step = 2 ** qubit
        phase = complex(math.cos(theta), math.sin(theta))
        for i in range(n_states):
            if (i // step) % 2 == 1:
                self.amplitudes[i] *= phase

    def measure(self) -> int:
        """Collapse the state and return the measured basis index."""
        probs = [abs(a) ** 2 for a in self.amplitudes]
        total = sum(probs)
        probs = [p / total for p in probs]
        r = random.random()
        cumulative = 0.0
        for idx, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return idx
        return len(probs) - 1

    @property
    def probabilities(self) -> List[float]:
        total = sum(abs(a) ** 2 for a in self.amplitudes)
        return [abs(a) ** 2 / total for a in self.amplitudes]


@dataclass
class InferenceResult:
    """Output of a single quantum inference run."""

    input_hash: str
    predicted_class: int
    confidence: float
    amplitudes: List[float]
    latency_ms: float
    n_qubits: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_hash": self.input_hash[:12],
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            "top_amplitudes": [round(a, 6) for a in self.amplitudes[:8]],
            "latency_ms": round(self.latency_ms, 4),
            "n_qubits": self.n_qubits,
        }

    def to_cache_dict(self) -> Dict[str, Any]:
        """Full serialisation suitable for JSON round-trip caching."""
        return {
            "input_hash": self.input_hash,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "amplitudes": self.amplitudes,
            "latency_ms": self.latency_ms,
            "n_qubits": self.n_qubits,
        }

    @classmethod
    def from_cache_dict(cls, data: Dict[str, Any]) -> "InferenceResult":
        return cls(
            input_hash=data["input_hash"],
            predicted_class=data["predicted_class"],
            confidence=data["confidence"],
            amplitudes=data["amplitudes"],
            latency_ms=data["latency_ms"],
            n_qubits=data["n_qubits"],
        )


class QuantumInferenceEngine:
    """
    Quantum-inspired inference engine for the Quad-Brain system.

    The engine encodes input data into a quantum state register,
    applies a learned rotation schedule, then measures the outcome
    to produce a classification decision.

    Parameters
    ----------
    n_qubits : int
        Number of simulated qubits (determines output class resolution).
    shots : int
        Number of measurement shots averaged to reduce noise.
    seed : int | None
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> None:
        if n_qubits < 2 or n_qubits > 20:
            raise ValueError("n_qubits must be between 2 and 20.")
        self.n_qubits = n_qubits
        self.shots = shots
        if seed is not None:
            random.seed(seed)
        self._inference_count: int = 0
        self._total_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_input(self, state: QuantumState, data: bytes) -> None:
        """
        Encode raw bytes into phase rotations across qubits.
        Each byte maps to a rotation angle on successive qubits.
        """
        for i, byte_val in enumerate(data[: self.n_qubits]):
            theta = (byte_val / 255.0) * math.pi
            state.apply_phase(i % self.n_qubits, theta)

    def _apply_entanglement_layer(self, state: QuantumState) -> None:
        """Apply Hadamard gates to create inter-qubit entanglement."""
        for q in range(self.n_qubits):
            state.apply_hadamard(q)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, data: bytes) -> InferenceResult:
        """
        Run quantum inference on raw byte data.

        Returns the predicted class (basis state with highest frequency)
        and associated confidence derived from shot statistics.
        """
        t0 = time.perf_counter()
        input_hash = hashlib.sha256(data).hexdigest()
        counts: Dict[int, int] = {}

        for _ in range(self.shots):
            state = QuantumState(n_qubits=self.n_qubits)
            self._apply_entanglement_layer(state)
            self._encode_input(state, data)
            outcome = state.measure()
            counts[outcome] = counts.get(outcome, 0) + 1

        predicted_class = max(counts, key=counts.__getitem__)
        confidence = counts[predicted_class] / self.shots

        # Return amplitude profile from a single final state for reporting
        final_state = QuantumState(n_qubits=self.n_qubits)
        self._apply_entanglement_layer(final_state)
        self._encode_input(final_state, data)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._inference_count += 1
        self._total_latency_ms += elapsed_ms

        result = InferenceResult(
            input_hash=input_hash,
            predicted_class=predicted_class,
            confidence=confidence,
            amplitudes=final_state.probabilities,
            latency_ms=elapsed_ms,
            n_qubits=self.n_qubits,
        )
        logger.debug(
            "Inference #%d: class=%d confidence=%.4f latency=%.2f ms.",
            self._inference_count,
            predicted_class,
            confidence,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def avg_latency_ms(self) -> float:
        return (
            self._total_latency_ms / self._inference_count
            if self._inference_count
            else 0.0
        )

    def engine_summary(self) -> Dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "shots": self.shots,
            "inferences_run": self._inference_count,
            "avg_latency_ms": round(self.avg_latency_ms, 4),
        }
