"""
Twinbrain Algorithm
====================

The Twinbrain algorithm models a dual-brain processing system with
four quantum-inspired thinking layers.  Each "brain" operates
independently on the same input, producing its own activation
sequence.  A *fusion layer* then merges both activation sequences
into a single decision vector.

Quad-quantum thinking layers
-----------------------------
Layer 0 – Superposition    : assigns equal probability to all outcomes
Layer 1 – Entanglement     : correlates features across the input vector
Layer 2 – Interference     : amplifies constructive and dampens destructive
                             feature interactions
Layer 3 – Measurement      : collapses the superposition into a discrete
                             activation via argmax

Each layer is implemented as a pure function operating on a list of
floats, making the pipeline easy to reason about, test, and extend.

Redundancy / split-clustering safety
--------------------------------------
Both brains process the same input independently.  If either brain
produces a null or all-zero activation (a "split-cluster failure"),
the fusion step falls back to the other brain's output.  If both
fail simultaneously, a uniform activation vector is returned so that
downstream components always receive a valid signal.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUAD_LAYERS: int = 4
_LAYER_NAMES = ["superposition", "entanglement", "interference", "measurement"]


# ---------------------------------------------------------------------------
# Quantum-inspired layer functions
# ---------------------------------------------------------------------------


def _layer_superposition(vector: List[float]) -> List[float]:
    """Assign equal weight to every dimension (uniform superposition)."""
    n = len(vector)
    if n == 0:
        return []
    weight = 1.0 / n
    return [v * weight for v in vector]


def _layer_entanglement(vector: List[float]) -> List[float]:
    """Cross-correlate adjacent dimensions to entangle features."""
    n = len(vector)
    if n == 0:
        return []
    result = []
    for i in range(n):
        left = vector[i - 1] if i > 0 else vector[-1]
        right = vector[(i + 1) % n]
        result.append((vector[i] + left * 0.5 + right * 0.5) / 2.0)
    return result


def _layer_interference(vector: List[float]) -> List[float]:
    """Apply constructive / destructive interference via sine modulation."""
    n = len(vector)
    if n == 0:
        return []
    return [
        v * math.sin(math.pi * (i + 1) / (n + 1))
        for i, v in enumerate(vector)
    ]


def _layer_measurement(vector: List[float]) -> List[float]:
    """Collapse the probability amplitudes into a one-hot activation."""
    if not vector:
        return []
    max_val = max(vector)
    if max_val == 0.0:
        return [0.0] * len(vector)
    return [1.0 if v == max_val else 0.0 for v in vector]


_LAYER_FN = [
    _layer_superposition,
    _layer_entanglement,
    _layer_interference,
    _layer_measurement,
]


# ---------------------------------------------------------------------------
# Brain
# ---------------------------------------------------------------------------


class Brain:
    """One half of the Twinbrain system.

    Parameters
    ----------
    brain_id:
        Label for this brain (e.g. ``"A"`` or ``"B"``).
    """

    def __init__(self, brain_id: str) -> None:
        self.brain_id = brain_id
        self._activation: Optional[List[float]] = None
        self._layer_outputs: List[List[float]] = []

    # ------------------------------------------------------------------

    def process(self, vector: List[float]) -> List[float]:
        """Run the input *vector* through all four quantum thinking layers.

        Parameters
        ----------
        vector:
            Non-empty list of floats representing the input signal.

        Returns
        -------
        List[float]
            Final activation vector after the measurement layer.
        """
        if not vector:
            raise ValueError(f"Brain {self.brain_id!r}: input vector must not be empty.")
        current = list(vector)
        self._layer_outputs = []
        for fn in _LAYER_FN:
            current = fn(current)
            self._layer_outputs.append(list(current))
        self._activation = current
        logger.debug(
            "Brain %r activation after %d layers: %s",
            self.brain_id,
            QUAD_LAYERS,
            self._activation,
        )
        return list(self._activation)

    # ------------------------------------------------------------------

    @property
    def activation(self) -> Optional[List[float]]:
        """Most recent activation output, or ``None`` if not yet run."""
        return self._activation

    # ------------------------------------------------------------------

    def is_alive(self) -> bool:
        """Return ``True`` if the activation is non-null and non-zero."""
        return (
            self._activation is not None
            and any(v != 0.0 for v in self._activation)
        )

    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """SHA-256 fingerprint of the latest activation."""
        payload = str(self._activation).encode()
        return hashlib.sha256(payload).hexdigest()

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Brain(id={self.brain_id!r}, alive={self.is_alive()}, "
            f"activation={self._activation})"
        )


# ---------------------------------------------------------------------------
# TwinbrainAlgorithm
# ---------------------------------------------------------------------------


class TwinbrainAlgorithm:
    """Dual-brain quad-quantum processing system with redundancy.

    Both :class:`Brain` instances (A and B) process the same input
    independently.  Their activations are fused into a single decision
    vector by element-wise summation followed by normalisation.  If one
    brain has failed (all-zero or null output) only the surviving brain's
    activation is used.  If both fail, a uniform fallback is returned.

    Parameters
    ----------
    brain_a_label:
        Label for the first brain (default ``"A"``).
    brain_b_label:
        Label for the second brain (default ``"B"``).

    Examples
    --------
    >>> tb = TwinbrainAlgorithm()
    >>> decision = tb.decide([0.1, 0.5, 0.9, 0.3])
    >>> print(decision)
    """

    def __init__(
        self,
        brain_a_label: str = "A",
        brain_b_label: str = "B",
    ) -> None:
        self.brain_a = Brain(brain_a_label)
        self.brain_b = Brain(brain_b_label)

    # ------------------------------------------------------------------

    def decide(self, vector: List[float]) -> List[float]:
        """Process *vector* through both brains and return a fused decision.

        Parameters
        ----------
        vector:
            Input feature vector.

        Returns
        -------
        List[float]
            Normalised decision vector.
        """
        act_a = self.brain_a.process(vector)
        act_b = self.brain_b.process(vector)

        a_alive = self.brain_a.is_alive()
        b_alive = self.brain_b.is_alive()

        if a_alive and b_alive:
            fused = [a + b for a, b in zip(act_a, act_b)]
        elif a_alive:
            logger.warning("Brain B failed; using Brain A exclusively.")
            fused = list(act_a)
        elif b_alive:
            logger.warning("Brain A failed; using Brain B exclusively.")
            fused = list(act_b)
        else:
            logger.error("Both brains failed; returning uniform fallback.")
            n = len(vector)
            fused = [1.0 / n] * n if n > 0 else []

        return self._normalise(fused)

    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(vector: List[float]) -> List[float]:
        """Return a unit-normalised version of *vector*."""
        total = sum(vector)
        if total == 0.0:
            n = len(vector)
            return [1.0 / n] * n if n > 0 else []
        return [v / total for v in vector]

    # ------------------------------------------------------------------

    def diagnostics(self) -> Dict[str, Any]:
        """Return a structured diagnostics report for both brains."""
        return {
            "brain_a": {
                "id": self.brain_a.brain_id,
                "alive": self.brain_a.is_alive(),
                "activation": self.brain_a.activation,
                "fingerprint": self.brain_a.fingerprint(),
                "layer_outputs": self.brain_a._layer_outputs,
            },
            "brain_b": {
                "id": self.brain_b.brain_id,
                "alive": self.brain_b.is_alive(),
                "activation": self.brain_b.activation,
                "fingerprint": self.brain_b.fingerprint(),
                "layer_outputs": self.brain_b._layer_outputs,
            },
        }

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TwinbrainAlgorithm(brain_a={self.brain_a!r}, "
            f"brain_b={self.brain_b!r})"
        )
