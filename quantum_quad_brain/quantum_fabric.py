"""
quantum_fabric.py
-----------------
Multi-tiered quantum fabric layer for data-state management and cluster processes.

Each fabric layer maintains a register of quantum data clusters.  Clusters are
grouped into vaulted containers that can be folded/re-folded dynamically to
optimise real-time inference throughput.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class DataCluster:
    """A cluster of quantum data states within a fabric layer."""

    def __init__(self, cluster_id: int, size: int = 16, seed: Optional[int] = None) -> None:
        self.cluster_id = cluster_id
        self.size = size
        rng = np.random.default_rng(seed if seed is not None else cluster_id)
        raw = rng.standard_normal(size) + 1j * rng.standard_normal(size)
        self.state: np.ndarray = raw / np.linalg.norm(raw)
        self.metadata: Dict[str, object] = {}

    def update(self, data: np.ndarray) -> None:
        """Absorb new data into the cluster state via interference."""
        if data.shape[0] != self.size:
            tmp = np.zeros(self.size, dtype=complex)
            n = min(data.shape[0], self.size)
            tmp[:n] = data[:n].astype(complex)
            data = tmp
        combined = self.state + data / (np.linalg.norm(data) + 1e-12)
        self.state = combined / np.linalg.norm(combined)

    def entropy(self) -> float:
        """Von-Neumann-like entropy: -Σ p_i log(p_i)."""
        probs = np.abs(self.state) ** 2
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def __repr__(self) -> str:
        return f"DataCluster(id={self.cluster_id}, size={self.size}, entropy={self.entropy():.4f})"


class VaultedContainer:
    """
    A vaulted container holding multiple data clusters.
    Containers can be stacked inside one another for nested isolation.
    """

    def __init__(self, container_id: int, clusters: int = 4, cluster_size: int = 16) -> None:
        self.container_id = container_id
        self.clusters: List[DataCluster] = [
            DataCluster(cluster_id=i, size=cluster_size, seed=container_id * 100 + i)
            for i in range(clusters)
        ]
        self.inner_container: Optional["VaultedContainer"] = None

    def nest(self, inner: "VaultedContainer") -> None:
        """Place another container inside this vault."""
        self.inner_container = inner

    def aggregate_state(self) -> np.ndarray:
        """Concatenate all cluster states into a single vector."""
        return np.concatenate([c.state for c in self.clusters])

    def total_entropy(self) -> float:
        return sum(c.entropy() for c in self.clusters)

    def __repr__(self) -> str:
        inner_repr = f", inner={repr(self.inner_container)}" if self.inner_container else ""
        return (
            f"VaultedContainer(id={self.container_id}, "
            f"clusters={len(self.clusters)}, "
            f"entropy={self.total_entropy():.4f}{inner_repr})"
        )


class QuantumFabricLayer:
    """
    A full quantum fabric layer consisting of multiple vaulted containers,
    supporting dynamic folding and re-folding for real-time optimisation.
    """

    def __init__(
        self,
        layer_id: int,
        num_containers: int = 4,
        clusters_per_container: int = 4,
        cluster_size: int = 16,
        enable_nesting: bool = True,
    ) -> None:
        self.layer_id = layer_id
        self.cluster_size = cluster_size
        self.containers: List[VaultedContainer] = [
            VaultedContainer(
                container_id=i,
                clusters=clusters_per_container,
                cluster_size=cluster_size,
            )
            for i in range(num_containers)
        ]
        if enable_nesting:
            self._nest_containers()
        self.fold_operations: int = 0

    def _nest_containers(self) -> None:
        """Create a nested container-in-container structure."""
        n = len(self.containers)
        for i in range(n - 1):
            self.containers[i].nest(self.containers[i + 1])

    # ------------------------------------------------------------------
    # Dynamic folding
    # ------------------------------------------------------------------

    def fold(self) -> None:
        """
        Re-fold: redistribute cluster states between containers via
        a quantum swap operation to optimise load balance.
        """
        if len(self.containers) < 2:
            return
        n = len(self.containers)
        for i in range(0, n - 1, 2):
            a = self.containers[i]
            b = self.containers[i + 1]
            for ca, cb in zip(a.clusters, b.clusters):
                # Quantum swap via CNOT-like operation
                tmp = ca.state.copy()
                ca.state = (ca.state + cb.state) / np.sqrt(2)
                cb.state = (tmp - cb.state) / np.sqrt(2)
                # Renormalise
                na = np.linalg.norm(ca.state)
                nb = np.linalg.norm(cb.state)
                if na > 1e-12:
                    ca.state /= na
                if nb > 1e-12:
                    cb.state /= nb
        self.fold_operations += 1

    def re_fold(self) -> None:
        """Re-fold in reverse order for bidirectional optimisation."""
        n = len(self.containers)
        for i in range(n - 1, 0, -2):
            a = self.containers[i]
            b = self.containers[i - 1]
            for ca, cb in zip(a.clusters, b.clusters):
                tmp = ca.state.copy()
                ca.state = (ca.state + cb.state) / np.sqrt(2)
                cb.state = (tmp - cb.state) / np.sqrt(2)
                na = np.linalg.norm(ca.state)
                nb = np.linalg.norm(cb.state)
                if na > 1e-12:
                    ca.state /= na
                if nb > 1e-12:
                    cb.state /= nb
        self.fold_operations += 1

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def ingest(self, data: np.ndarray) -> None:
        """
        Route incoming data to clusters across all containers.
        Data is split evenly and each chunk routed to one cluster.
        """
        total_clusters = sum(len(c.clusters) for c in self.containers)
        chunk_size = self.cluster_size
        idx = 0
        for container in self.containers:
            for cluster in container.clusters:
                start = (idx * chunk_size) % max(data.shape[0], 1)
                end = start + chunk_size
                chunk = data[start:end] if start < data.shape[0] else np.zeros(chunk_size)
                cluster.update(chunk)
                idx += 1

    def fabric_state(self) -> np.ndarray:
        """Full fabric state: concatenation of all container aggregate states."""
        return np.concatenate([c.aggregate_state() for c in self.containers])

    def total_entropy(self) -> float:
        return sum(c.total_entropy() for c in self.containers)

    def __repr__(self) -> str:
        return (
            f"QuantumFabricLayer("
            f"id={self.layer_id}, "
            f"containers={len(self.containers)}, "
            f"folds={self.fold_operations}, "
            f"entropy={self.total_entropy():.4f})"
        )
