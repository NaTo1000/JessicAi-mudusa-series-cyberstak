"""
Quantum Quad-Brain Coordinator
==============================
Orchestrates the four inference engines (the "quad-brain"), the NVMe/DRAM
storage layer, and the CM4 cluster into a unified, self-managing compute
array.

Each "brain" is an independent QuantumInferenceEngine with its own qubit
count and shot configuration. The coordinator fans out tasks to the
CM4 cluster, caches hot results in DRAM, and persists cold data to NVMe.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .inference_engine import InferenceResult, QuantumInferenceEngine
from ..nvme_dram.storage_pipeline import NVMeStoragePipeline, NVMeDevice
from ..nvme_dram.cache_manager import DRAMCacheManager
from ..cm4_cluster.cluster_manager import CM4ClusterManager, CM4Node
from ..cm4_cluster.workload_distributor import WorkloadDistributor, SchedulingStrategy

logger = logging.getLogger(__name__)


@dataclass
class BrainConfig:
    """Configuration for one of the four quantum inference engines."""

    brain_id: str
    n_qubits: int = 8
    shots: int = 512
    seed: Optional[int] = None


class QuantumQuadBrain:
    """
    The Quantum Quad-Brain: four quantum inference engines coordinated over
    a CM4 cluster with NVMe-backed DRAM caching.

    Parameters
    ----------
    brain_configs : list[BrainConfig], optional
        Per-brain tuning; defaults to four engines with 6–12 qubits.
    nvme_devices : list[NVMeDevice], optional
        NVMe drives to include in the storage pipeline.
    dram_capacity_gb : float
        DRAM allocated to the cache layer.
    n_cm4_nodes : int
        Number of simulated CM4 nodes to pre-populate the cluster with.
    scheduling_strategy : SchedulingStrategy
        Workload distribution algorithm.
    """

    DEFAULT_BRAIN_CONFIGS = [
        BrainConfig("alpha",   n_qubits=6,  shots=256),
        BrainConfig("beta",    n_qubits=8,  shots=512),
        BrainConfig("gamma",   n_qubits=10, shots=512),
        BrainConfig("delta",   n_qubits=12, shots=1024),
    ]

    def __init__(
        self,
        brain_configs: Optional[List[BrainConfig]] = None,
        nvme_devices: Optional[List[NVMeDevice]] = None,
        dram_capacity_gb: float = 8.0,
        n_cm4_nodes: int = 4,
        scheduling_strategy: SchedulingStrategy = SchedulingStrategy.LEAST_LOADED,
    ) -> None:
        configs = brain_configs or self.DEFAULT_BRAIN_CONFIGS
        if len(configs) != 4:
            raise ValueError("QuantumQuadBrain requires exactly 4 brain configs.")

        # Build the four inference engines
        self.brains: Dict[str, QuantumInferenceEngine] = {
            cfg.brain_id: QuantumInferenceEngine(
                n_qubits=cfg.n_qubits,
                shots=cfg.shots,
                seed=cfg.seed,
            )
            for cfg in configs
        }
        self.brain_ids = [cfg.brain_id for cfg in configs]

        # NVMe + DRAM storage layer
        self.storage = NVMeStoragePipeline(devices=nvme_devices or [])
        self.cache = DRAMCacheManager(capacity_gb=dram_capacity_gb)

        # CM4 cluster
        self.cluster = CM4ClusterManager()
        for i in range(n_cm4_nodes):
            self.cluster.register_node(
                CM4Node(
                    node_id=f"cm4-{i:03d}",
                    ip_address=f"10.0.0.{100 + i}",
                )
            )

        self.distributor = WorkloadDistributor(
            cluster=self.cluster,
            strategy=scheduling_strategy,
        )

        self._request_count: int = 0
        self._start_time: float = time.time()
        logger.info(
            "QuantumQuadBrain initialised: %d brains, %d CM4 nodes, %.1f GB DRAM cache.",
            len(self.brains),
            n_cm4_nodes,
            dram_capacity_gb,
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def infer(self, data: bytes, brain_id: Optional[str] = None) -> InferenceResult:
        """
        Run quantum inference on *data*.

        If *brain_id* is ``None`` the delta brain (most powerful, 12 qubits)
        is used.  The result is cached in DRAM; subsequent identical inputs
        are served from cache without re-running the circuit.

        Parameters
        ----------
        data : bytes
            Raw input payload.
        brain_id : str, optional
            Which of the four brains to use: ``"alpha"``, ``"beta"``,
            ``"gamma"``, or ``"delta"``.
        """
        brain_id = brain_id or "delta"
        if brain_id not in self.brains:
            raise KeyError(f"Unknown brain '{brain_id}'. Choose from {list(self.brains)}.")

        cache_key = f"{brain_id}:{data.hex()[:64]}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            import json
            from .inference_engine import InferenceResult
            return InferenceResult.from_cache_dict(json.loads(cached))

        result = self.brains[brain_id].infer(data)

        import json
        self.cache.put(cache_key, json.dumps(result.to_cache_dict()).encode())

        self._request_count += 1
        return result

    def parallel_infer(self, data: bytes) -> Dict[str, InferenceResult]:
        """
        Run all four brains on the same input and return a consensus map.
        Tasks are dispatched to CM4 nodes via the workload distributor.
        """
        results: Dict[str, InferenceResult] = {}

        def _executor(node, payload):
            brain_id, raw = payload
            return self.infer(raw, brain_id=brain_id)

        tasks = self.distributor.submit_batch(
            payloads=[(bid, data) for bid in self.brain_ids],
            executor=_executor,
        )
        for task, bid in zip(tasks, self.brain_ids):
            if task.completed and task.result is not None:
                results[bid] = task.result
            else:
                logger.warning("Brain '%s' inference task failed: %s", bid, task.error)

        return results

    def consensus_infer(self, data: bytes) -> Dict[str, Any]:
        """
        Run all four brains and return a majority-vote consensus decision.
        """
        all_results = self.parallel_infer(data)
        if not all_results:
            return {"error": "All brains failed."}

        # Majority vote
        votes: Dict[int, float] = {}
        for bid, res in all_results.items():
            cls = res.predicted_class
            votes[cls] = votes.get(cls, 0.0) + res.confidence

        winner = max(votes, key=votes.__getitem__)
        total_confidence = sum(votes.values())
        return {
            "consensus_class": winner,
            "consensus_confidence": round(votes[winner] / total_confidence, 4),
            "brain_votes": {
                bid: {"class": r.predicted_class, "confidence": round(r.confidence, 4)}
                for bid, r in all_results.items()
            },
        }

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def persist(self, key: str, data: bytes) -> str:
        """Write data to NVMe, warm the DRAM cache, and return the storage key."""
        storage_key = self.storage.write(key, data)
        self.cache.put(storage_key, data)
        return storage_key

    def retrieve(self, key: str, size_hint: int = 0) -> bytes:
        """Retrieve data from DRAM cache, falling back to NVMe."""
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        data = self.storage.read(key, size_hint=size_hint)
        self.cache.put(key, data)
        return data

    # ------------------------------------------------------------------
    # System-level introspection
    # ------------------------------------------------------------------

    def system_summary(self) -> Dict[str, Any]:
        uptime_s = time.time() - self._start_time
        return {
            "uptime_s": round(uptime_s, 1),
            "requests": self._request_count,
            "throughput_rps": round(self._request_count / uptime_s, 4) if uptime_s else 0.0,
            "brains": {
                bid: engine.engine_summary()
                for bid, engine in self.brains.items()
            },
            "storage": self.storage.summary(),
            "cache": self.cache.summary(),
            "cluster": self.cluster.cluster_summary(),
            "distributor": self.distributor.distributor_summary(),
        }
