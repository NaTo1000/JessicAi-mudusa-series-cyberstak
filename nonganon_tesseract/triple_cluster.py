"""
Triple-Clustering Thought Processor
=====================================

Implements the three-cluster engine that sits inside the
TesseractOctagon framework.  Each cluster is independent and
processes a sub-partition of the input feature space.  After
individual clusters converge, a *meta-cluster* step aggregates
their centroids into a unified cluster representation.

Algorithm summary
-----------------
1. **Partition** – the input data set is split into three
   non-overlapping shards.
2. **Local convergence** – each shard is processed by its own
   :class:`Cluster` instance using an iterative centroid-update
   loop.
3. **Meta-aggregation** – the three per-cluster centroids are
   themselves clustered to produce a final *super-centroid* that
   represents the global structure of the input.

This three-level structure maps directly onto the triple-clustering
thought-processor concept: each cluster is a "thought", and the
meta-cluster is the synthesised decision.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _euclidean(a: List[float], b: List[float]) -> float:
    """Return the Euclidean distance between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have equal length.")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mean_vector(points: List[List[float]]) -> List[float]:
    """Return the element-wise mean of a list of equal-length vectors."""
    if not points:
        return []
    dim = len(points[0])
    totals = [0.0] * dim
    for p in points:
        for i, v in enumerate(p):
            totals[i] += v
    return [t / len(points) for t in totals]


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


class Cluster:
    """A single cluster within the triple-clustering processor.

    Parameters
    ----------
    cluster_id:
        Unique identifier string for this cluster (e.g. ``"alpha"``).
    max_iterations:
        Maximum centroid-update iterations before convergence is
        declared regardless.
    tolerance:
        Minimum centroid shift below which convergence is assumed.
    """

    def __init__(
        self,
        cluster_id: str,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> None:
        self.cluster_id = cluster_id
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroid: Optional[List[float]] = None
        self.members: List[List[float]] = []
        self.iterations_run: int = 0
        self.converged: bool = False
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------

    def fit(self, data: List[List[float]]) -> None:
        """Compute the centroid for *data* using iterative mean updates.

        Parameters
        ----------
        data:
            List of equal-length numeric vectors.

        Raises
        ------
        ValueError
            If *data* is empty.
        """
        if not data:
            raise ValueError(f"Cluster {self.cluster_id!r}: data must not be empty.")
        self.members = [list(p) for p in data]
        # Initialise centroid to first data point
        self.centroid = list(data[0])
        for iteration in range(self.max_iterations):
            new_centroid = _mean_vector(self.members)
            shift = _euclidean(self.centroid, new_centroid)
            self.centroid = new_centroid
            self.iterations_run = iteration + 1
            if shift < self.tolerance:
                self.converged = True
                break
        logger.debug(
            "Cluster %r converged=%s after %d iteration(s), centroid=%s.",
            self.cluster_id,
            self.converged,
            self.iterations_run,
            self.centroid,
        )

    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """SHA-256 fingerprint of this cluster's centroid."""
        payload = str(self.centroid).encode()
        return hashlib.sha256(payload).hexdigest()

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Cluster(id={self.cluster_id!r}, members={len(self.members)}, "
            f"converged={self.converged})"
        )


# ---------------------------------------------------------------------------
# TripleCluster
# ---------------------------------------------------------------------------


class TripleCluster:
    """Triple-clustering thought processor.

    Splits input data into three equal (or near-equal) partitions,
    fits a :class:`Cluster` to each partition, then produces a
    *meta-centroid* by averaging the three per-cluster centroids.

    Parameters
    ----------
    max_iterations:
        Passed to each individual :class:`Cluster`.
    tolerance:
        Convergence tolerance passed to each individual :class:`Cluster`.
    seed:
        Optional random seed for reproducible shard shuffling.

    Examples
    --------
    >>> data = [[float(i), float(i * 2)] for i in range(30)]
    >>> tc = TripleCluster(seed=42)
    >>> tc.fit(data)
    >>> print(tc.meta_centroid)
    """

    CLUSTER_IDS: Tuple[str, str, str] = ("alpha", "beta", "gamma")

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._rng = random.Random(seed)
        self.clusters: List[Cluster] = [
            Cluster(cid, max_iterations, tolerance)
            for cid in self.CLUSTER_IDS
        ]
        self.meta_centroid: Optional[List[float]] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------

    def fit(self, data: List[List[float]]) -> None:
        """Partition *data* across three clusters and compute centroids.

        Parameters
        ----------
        data:
            Non-empty list of equal-length numeric vectors.

        Raises
        ------
        ValueError
            If *data* contains fewer than 3 points (one per cluster).
        """
        if len(data) < 3:
            raise ValueError(
                f"TripleCluster requires at least 3 data points, got {len(data)}."
            )
        shuffled = list(data)
        self._rng.shuffle(shuffled)
        shards = self._partition(shuffled, 3)
        for cluster, shard in zip(self.clusters, shards):
            cluster.fit(shard)
        centroids = [c.centroid for c in self.clusters if c.centroid is not None]
        self.meta_centroid = _mean_vector(centroids)  # type: ignore[arg-type]
        self._fitted = True
        logger.info(
            "TripleCluster fitted. meta_centroid=%s", self.meta_centroid
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _partition(
        items: List[Any], n: int
    ) -> List[List[Any]]:
        """Split *items* into *n* roughly equal sub-lists."""
        size, remainder = divmod(len(items), n)
        partitions: List[List[Any]] = []
        start = 0
        for i in range(n):
            end = start + size + (1 if i < remainder else 0)
            partitions.append(items[start:end])
            start = end
        return partitions

    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a structured summary of the fitted clusters."""
        if not self._fitted:
            return {"status": "not_fitted"}
        return {
            "clusters": [
                {
                    "id": c.cluster_id,
                    "members": len(c.members),
                    "centroid": c.centroid,
                    "converged": c.converged,
                    "iterations": c.iterations_run,
                    "fingerprint": c.fingerprint(),
                }
                for c in self.clusters
            ],
            "meta_centroid": self.meta_centroid,
        }

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TripleCluster(fitted={self._fitted}, "
            f"meta_centroid={self.meta_centroid})"
        )
