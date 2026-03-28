"""
Quantum Inference Engine
=========================

Probabilistic inference layer supporting adaptive learning paths.

Conceptual model
----------------
The engine represents the current "quantum state" as a probability
distribution over *n* discrete outcomes.  Each inference step
combines:

1. **Prior** – the current state vector (probability distribution).
2. **Evidence** – a new observation vector provided by the caller.
3. **Bayesian update** – the posterior is computed as the element-wise
   product of the prior and the likelihood derived from the evidence,
   then renormalised.
4. **Adaptive learning** – a learning-rate parameter ``alpha``
   controls how aggressively the evidence shifts the prior.  At
   ``alpha=1.0`` the prior is fully replaced by the evidence;
   at ``alpha=0.0`` the prior is unchanged.

This gives a lightweight but principled probabilistic inference
mechanism that can be embedded in a larger decision pipeline.

Clustering integration
-----------------------
:meth:`infer_clusters` accepts a :class:`~nonganon_tesseract.triple_cluster.TripleCluster`
summary dict and derives outcome weights from the per-cluster member
counts and centroid norms, providing an inference path that reflects
the data distribution discovered during clustering.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QuantumInferenceEngine
# ---------------------------------------------------------------------------


class QuantumInferenceEngine:
    """Probabilistic quantum-inference engine with adaptive learning.

    Parameters
    ----------
    n_outcomes:
        Number of discrete outcomes the engine models (≥ 2).
    alpha:
        Learning rate in ``[0.0, 1.0]``.  Controls how strongly each
        new evidence vector updates the prior.
    seed_state:
        Optional list of *n_outcomes* non-negative floats to use as the
        initial state.  Will be renormalised automatically.  If
        ``None``, a uniform prior is used.

    Examples
    --------
    >>> engine = QuantumInferenceEngine(n_outcomes=4, alpha=0.6)
    >>> engine.update([0.1, 0.7, 0.1, 0.1])
    >>> print(engine.state)
    """

    def __init__(
        self,
        n_outcomes: int = 4,
        alpha: float = 0.5,
        seed_state: Optional[List[float]] = None,
    ) -> None:
        if n_outcomes < 2:
            raise ValueError("n_outcomes must be at least 2.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0.0, 1.0].")
        self.n_outcomes = n_outcomes
        self.alpha = alpha
        self._steps: int = 0

        if seed_state is not None:
            if len(seed_state) != n_outcomes:
                raise ValueError(
                    f"seed_state length {len(seed_state)} != n_outcomes {n_outcomes}."
                )
            self._state = self._normalise(seed_state)
        else:
            self._state = [1.0 / n_outcomes] * n_outcomes

    # ------------------------------------------------------------------

    @property
    def state(self) -> List[float]:
        """Current probability distribution over outcomes."""
        return list(self._state)

    # ------------------------------------------------------------------

    def update(self, evidence: List[float]) -> List[float]:
        """Update the internal state with a new evidence vector.

        Parameters
        ----------
        evidence:
            Non-negative likelihood vector of length *n_outcomes*.
            Need not be normalised.

        Returns
        -------
        List[float]
            Updated (posterior) probability distribution.
        """
        if len(evidence) != self.n_outcomes:
            raise ValueError(
                f"Evidence length {len(evidence)} != n_outcomes {self.n_outcomes}."
            )
        if any(v < 0.0 for v in evidence):
            raise ValueError("Evidence values must be non-negative.")
        normed_evidence = self._normalise(evidence)
        # Adaptive blend: posterior = (1-alpha)*prior + alpha*evidence
        posterior = [
            (1.0 - self.alpha) * p + self.alpha * e
            for p, e in zip(self._state, normed_evidence)
        ]
        self._state = self._normalise(posterior)
        self._steps += 1
        logger.debug(
            "Inference step %d: state=%s", self._steps, self._state
        )
        return list(self._state)

    # ------------------------------------------------------------------

    def infer_clusters(self, cluster_summary: Dict[str, Any]) -> List[float]:
        """Derive outcome weights from a :class:`TripleCluster` summary.

        The weight for each cluster is proportional to the product of
        its member count and the L2-norm of its centroid.

        Parameters
        ----------
        cluster_summary:
            The dict returned by
            :meth:`~nonganon_tesseract.triple_cluster.TripleCluster.summary`.

        Returns
        -------
        List[float]
            Updated probability distribution after incorporating cluster
            information.
        """
        clusters = cluster_summary.get("clusters", [])
        if not clusters:
            logger.warning("infer_clusters: empty cluster summary, skipping update.")
            return self.state

        weights: List[float] = []
        for c in clusters:
            centroid: Optional[List[float]] = c.get("centroid")
            members: int = c.get("members", 0)
            norm = math.sqrt(sum(v * v for v in centroid)) if centroid else 0.0
            weights.append(float(members) * norm)

        # Pad or trim to n_outcomes
        while len(weights) < self.n_outcomes:
            weights.append(0.0)
        weights = weights[: self.n_outcomes]

        return self.update(weights)

    # ------------------------------------------------------------------

    def most_probable_outcome(self) -> int:
        """Return the index of the outcome with highest probability."""
        return max(range(self.n_outcomes), key=lambda i: self._state[i])

    # ------------------------------------------------------------------

    def entropy(self) -> float:
        """Shannon entropy of the current state distribution (in nats)."""
        return -sum(p * math.log(p) for p in self._state if p > 0.0)

    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """SHA-256 fingerprint of the current state vector."""
        payload = str(self._state).encode()
        return hashlib.sha256(payload).hexdigest()

    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(vector: List[float]) -> List[float]:
        total = sum(vector)
        if total == 0.0:
            n = len(vector)
            return [1.0 / n] * n if n > 0 else []
        return [v / total for v in vector]

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"QuantumInferenceEngine(n_outcomes={self.n_outcomes}, "
            f"alpha={self.alpha}, steps={self._steps})"
        )
