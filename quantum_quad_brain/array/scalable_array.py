"""
Scalable Compute Array
======================
Top-level orchestrator that assembles the NVMe pipeline, DRAM cache,
CM4 cluster, and Quantum Quad-Brain into a single, modular system with
plug-and-play node/device management and fault tolerance.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..quantum_core.quad_brain import QuantumQuadBrain, BrainConfig
from ..nvme_dram.storage_pipeline import NVMeDevice
from ..cm4_cluster.cluster_manager import CM4Node
from ..cm4_cluster.workload_distributor import SchedulingStrategy
from ..metrics.performance_monitor import PerformanceMonitor
from ..metrics.visualizer import MetricsVisualizer
from .fault_tolerance import FaultToleranceManager

logger = logging.getLogger(__name__)


class ScalableComputeArray:
    """
    Modular, scalable compute array integrating:
    - Quantum Quad-Brain inference engines
    - NVMe SSD storage pipeline
    - DRAM caching layer
    - Raspberry Pi CM4 distributed compute cluster
    - Real-time performance monitoring
    - Automatic fault detection and recovery

    The array supports plug-and-play addition of NVMe devices and CM4 nodes
    at runtime, enabling horizontal scale-out without downtime.

    Parameters
    ----------
    dram_capacity_gb : float
        DRAM capacity allocated to the cache layer.
    initial_nvme_devices : list[NVMeDevice], optional
        NVMe devices to pre-populate in the storage pipeline.
    initial_cm4_nodes : int
        Number of CM4 nodes to pre-register in the cluster.
    scheduling_strategy : SchedulingStrategy
        Workload scheduling algorithm.
    brain_configs : list[BrainConfig], optional
        Per-brain configuration for the four inference engines.
    """

    def __init__(
        self,
        dram_capacity_gb: float = 8.0,
        initial_nvme_devices: Optional[List[NVMeDevice]] = None,
        initial_cm4_nodes: int = 4,
        scheduling_strategy: SchedulingStrategy = SchedulingStrategy.LEAST_LOADED,
        brain_configs: Optional[List[BrainConfig]] = None,
    ) -> None:
        self.quad_brain = QuantumQuadBrain(
            brain_configs=brain_configs,
            nvme_devices=initial_nvme_devices,
            dram_capacity_gb=dram_capacity_gb,
            n_cm4_nodes=initial_cm4_nodes,
            scheduling_strategy=scheduling_strategy,
        )
        self.monitor = PerformanceMonitor()
        self.visualizer = MetricsVisualizer()
        self.fault_manager = FaultToleranceManager()
        self._created_at = time.time()
        logger.info("ScalableComputeArray online.")

    # ------------------------------------------------------------------
    # Plug-and-play device / node management
    # ------------------------------------------------------------------

    def add_nvme_device(
        self,
        device_id: str,
        path: str,
        capacity_gb: float = 2000.0,
        read_bandwidth_gbps: float = 7.0,
        write_bandwidth_gbps: float = 6.5,
    ) -> None:
        """Hot-add an NVMe SSD to the storage pipeline."""
        device = NVMeDevice(
            device_id=device_id,
            path=Path(path),
            capacity_gb=capacity_gb,
            read_bandwidth_gbps=read_bandwidth_gbps,
            write_bandwidth_gbps=write_bandwidth_gbps,
        )
        self.quad_brain.storage.add_device(device)
        logger.info("NVMe device '%s' hot-added.", device_id)

    def remove_nvme_device(self, device_id: str) -> None:
        """Hot-remove an NVMe SSD from the storage pipeline."""
        self.quad_brain.storage.remove_device(device_id)
        logger.info("NVMe device '%s' hot-removed.", device_id)

    def add_cm4_node(
        self,
        node_id: str,
        ip_address: str,
        ram_gb: float = 8.0,
    ) -> None:
        """Hot-add a Raspberry Pi CM4 node to the cluster."""
        node = CM4Node(node_id=node_id, ip_address=ip_address, ram_gb=ram_gb)
        self.quad_brain.cluster.register_node(node)
        logger.info("CM4 node '%s' hot-added.", node_id)

    def remove_cm4_node(self, node_id: str) -> None:
        """Hot-remove a CM4 node from the cluster."""
        self.quad_brain.cluster.deregister_node(node_id)
        logger.info("CM4 node '%s' hot-removed.", node_id)

    # ------------------------------------------------------------------
    # Inference entry points
    # ------------------------------------------------------------------

    def run_inference(self, data: bytes, brain_id: Optional[str] = None) -> Dict[str, Any]:
        """Run inference on *data* and return the result dict."""
        result = self.quad_brain.infer(data, brain_id=brain_id)
        self.monitor.record_from_system(self.quad_brain)
        return result.to_dict()

    def run_consensus_inference(self, data: bytes) -> Dict[str, Any]:
        """Fan out to all four brains and return a consensus decision."""
        consensus = self.quad_brain.consensus_infer(data)
        self.monitor.record_from_system(self.quad_brain)
        return consensus

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(self, key: str, data: bytes) -> str:
        """Persist data to NVMe and warm the DRAM cache."""
        return self.quad_brain.persist(key, data)

    def load(self, key: str, size_hint: int = 0) -> bytes:
        """Retrieve data, serving from DRAM cache when available."""
        return self.quad_brain.retrieve(key, size_hint=size_hint)

    # ------------------------------------------------------------------
    # Health & diagnostics
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Run fault checks and return the current health status."""
        faults = self.fault_manager.run_all_checks(self.quad_brain)
        return {
            "healthy": len(self.fault_manager.open_incidents) == 0,
            "new_faults": [f.to_dict() for f in faults],
            "incident_summary": self.fault_manager.incident_summary(),
        }

    def dashboard(self) -> str:
        """Return a rendered ASCII performance dashboard."""
        self.monitor.record_from_system(self.quad_brain)
        return self.visualizer.render_dashboard(self.quad_brain, self.monitor)

    def brain_comparison(self, data: bytes) -> str:
        """Run all four brains and return an ASCII comparison chart."""
        consensus = self.quad_brain.consensus_infer(data)
        return self.visualizer.render_brain_comparison(consensus)

    def sparklines(self) -> str:
        """Return ASCII sparkline trends for key metrics."""
        return self.visualizer.render_sparklines(self.monitor)

    def full_summary(self) -> Dict[str, Any]:
        """Return a comprehensive JSON-serialisable summary of the array."""
        return {
            "array": {
                "uptime_s": round(time.time() - self._created_at, 1),
            },
            "system": self.quad_brain.system_summary(),
            "performance": self.monitor.snapshot(),
            "health": self.fault_manager.incident_summary(),
        }
