"""
hardware_layer.py — Distributed hardware node abstraction.

Maps biological neural structures onto physical compute hardware:
- Raspberry Pi CM4 compute modules  → Neuron clusters (processing nodes)
- DRAM (memory)                     → Short-term synaptic state / working memory
- NVMe SSD storage                  → Long-term memory / synaptic weight persistence

Each HardwareNode encapsulates a group of neurons and manages:
- Memory bandwidth simulation (DRAM read/write latency)
- Storage I/O simulation (NVMe access patterns)
- Inter-node communication latency (synaptic delay bus)
- Thermal and power state (affects firing thresholds)
"""

import time
import random
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Any

from .neuron import Neuron


class NodeType(Enum):
    """Physical hardware type for a compute node."""
    CM4_COMPUTE = auto()   # Raspberry Pi CM4 — primary neural processor
    DRAM_CACHE = auto()    # Working memory — fast, volatile state store
    NVME_STORAGE = auto()  # Long-term weight / trace persistence


@dataclass
class HardwareSpec:
    """Specifications for a hardware compute node."""
    node_type: NodeType = NodeType.CM4_COMPUTE

    # CM4 / compute specs
    cpu_cores: int = 4
    clock_speed_mhz: float = 1800.0
    ram_mb: int = 8192

    # DRAM specs
    dram_bandwidth_gbps: float = 25.6     # DDR4-3200 theoretical
    dram_latency_ns: float = 60.0         # Typical DDR4 CAS latency

    # NVMe specs
    nvme_read_gbps: float = 7.0           # PCIe Gen 4 NVMe
    nvme_write_gbps: float = 6.5
    nvme_latency_us: float = 20.0         # Typical NVMe read latency

    # Thermal
    thermal_design_power_w: float = 6.0   # CM4 TDP
    thermal_throttle_temp_c: float = 85.0
    operating_temp_c: float = 45.0        # Simulated operating temperature


class HardwareNode:
    """
    A single compute node hosting a group of neurons.

    Simulates real hardware constraints including:
    - Memory access latency (affects synaptic delay)
    - Storage write latency (affects weight persistence)
    - Thermal throttling (slows clock, raises firing threshold)
    - Inter-node messaging latency (network topology)
    """

    def __init__(
        self,
        node_id: str,
        spec: Optional[HardwareSpec] = None,
        neurons: Optional[List[Neuron]] = None,
    ):
        self.node_id = node_id
        self.spec = spec or HardwareSpec()
        self.neurons: List[Neuron] = neurons or []

        # Runtime state
        self._dram_usage_mb: float = 0.0
        self._nvme_usage_gb: float = 0.0
        self._pending_ops: List[Dict[str, Any]] = []
        self._temperature_c: float = self.spec.operating_temp_c

        # Performance counters
        self.dram_reads: int = 0
        self.dram_writes: int = 0
        self.nvme_reads: int = 0
        self.nvme_writes: int = 0
        self.total_messages_sent: int = 0
        self.total_messages_received: int = 0

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def dram_read(self, size_kb: float = 1.0) -> float:
        """
        Simulate a DRAM read operation.
        Returns the simulated latency in microseconds.
        """
        base_latency_us = self.spec.dram_latency_ns / 1000.0
        bandwidth_factor = size_kb / (self.spec.dram_bandwidth_gbps * 1024 * 1024)
        latency_us = base_latency_us + bandwidth_factor * 1e6
        self.dram_reads += 1
        self._update_temperature(delta_c=0.001)
        return latency_us

    def dram_write(self, size_kb: float = 1.0, data: Optional[Any] = None) -> float:
        """
        Simulate a DRAM write operation.
        Returns the simulated latency in microseconds.
        """
        base_latency_us = self.spec.dram_latency_ns / 1000.0
        latency_us = base_latency_us * 1.1  # Writes slightly slower
        self._dram_usage_mb += size_kb / 1024.0
        self.dram_writes += 1
        self._update_temperature(delta_c=0.002)
        return latency_us

    def nvme_persist(self, data: Dict[str, Any]) -> float:
        """
        Persist synaptic weights / neural state to simulated NVMe storage.

        In a real deployment, this serialises the weight dictionary to an
        NVMe-backed file. Here, we simulate write latency.

        Returns simulated write latency in milliseconds.
        """
        size_bytes = len(str(data).encode("utf-8"))
        size_gb = size_bytes / (1024 ** 3)
        latency_ms = (size_gb / self.spec.nvme_write_gbps) * 1000.0
        latency_ms += self.spec.nvme_latency_us / 1000.0
        self._nvme_usage_gb += size_gb
        self.nvme_writes += 1
        return latency_ms

    def nvme_load(self, size_bytes: int = 1024) -> float:
        """
        Simulate loading data from NVMe.
        Returns simulated read latency in milliseconds.
        """
        size_gb = size_bytes / (1024 ** 3)
        latency_ms = (size_gb / self.spec.nvme_read_gbps) * 1000.0
        latency_ms += self.spec.nvme_latency_us / 1000.0
        self.nvme_reads += 1
        return latency_ms

    # ------------------------------------------------------------------
    # Inter-node communication
    # ------------------------------------------------------------------

    def send_spike_message(self, target_node: "HardwareNode", neuron_id: str, timestamp: float) -> float:
        """
        Send a spike event to another hardware node.

        Simulates gigabit Ethernet / PCIe interconnect latency.
        Returns round-trip communication latency in microseconds.
        """
        # Ethernet ~100 µs, PCIe ~1 µs
        base_latency_us = 2.0 if target_node.node_id != self.node_id else 0.1
        jitter_us = random.gauss(0.0, 0.5)
        latency_us = max(0.1, base_latency_us + jitter_us)

        message = {
            "source_node": self.node_id,
            "target_node": target_node.node_id,
            "neuron_id": neuron_id,
            "timestamp": timestamp,
            "latency_us": latency_us,
        }
        target_node._receive_spike(message)
        self.total_messages_sent += 1
        return latency_us

    def _receive_spike(self, message: Dict[str, Any]) -> None:
        self._pending_ops.append(message)
        self.total_messages_received += 1

    # ------------------------------------------------------------------
    # Thermal management
    # ------------------------------------------------------------------

    def _update_temperature(self, delta_c: float = 0.01) -> None:
        """Update simulated temperature and check throttling."""
        self.spec.operating_temp_c = min(
            self.spec.thermal_throttle_temp_c,
            self.spec.operating_temp_c + delta_c
        )
        # Passive cooling — temperature decays toward ambient (30°C)
        ambient_c = 30.0
        self.spec.operating_temp_c += (ambient_c - self.spec.operating_temp_c) * 0.001

    @property
    def is_throttled(self) -> bool:
        """True if thermal throttling is active."""
        return self.spec.operating_temp_c >= self.spec.thermal_throttle_temp_c * 0.95

    @property
    def effective_clock_mhz(self) -> float:
        """Clock speed accounting for thermal throttling."""
        if self.is_throttled:
            return self.spec.clock_speed_mhz * 0.6
        return self.spec.clock_speed_mhz

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.spec.node_type.name,
            "neurons": len(self.neurons),
            "temperature_c": round(self.spec.operating_temp_c, 1),
            "throttled": self.is_throttled,
            "clock_mhz": round(self.effective_clock_mhz, 1),
            "dram_usage_mb": round(self._dram_usage_mb, 2),
            "nvme_usage_gb": round(self._nvme_usage_gb, 4),
            "dram_reads": self.dram_reads,
            "dram_writes": self.dram_writes,
            "nvme_reads": self.nvme_reads,
            "nvme_writes": self.nvme_writes,
            "messages_sent": self.total_messages_sent,
            "messages_received": self.total_messages_received,
        }

    def __repr__(self) -> str:
        return (
            f"HardwareNode(id={self.node_id!r}, "
            f"type={self.spec.node_type.name}, "
            f"neurons={len(self.neurons)}, "
            f"temp={self.spec.operating_temp_c:.1f}°C)"
        )


class HardwareLayer:
    """
    Orchestrates the full distributed hardware array.

    Represents the complete physical substrate of the Quantum Neural Brain:
    - Multiple CM4 compute clusters (each running a neuron group)
    - DRAM nodes for working memory / short-term state
    - NVMe nodes for long-term weight persistence

    Inspired by the biological architecture where different brain regions
    (cortex, hippocampus, cerebellum) map to different hardware roles.
    """

    def __init__(self):
        self.nodes: Dict[str, HardwareNode] = {}
        self._topology: Dict[str, List[str]] = {}  # node_id → list of connected node_ids

    def add_node(self, node: HardwareNode) -> None:
        """Register a hardware node."""
        self.nodes[node.node_id] = node
        self._topology[node.node_id] = []

    def connect_nodes(self, node_a_id: str, node_b_id: str) -> None:
        """Create a bidirectional inter-node connection."""
        if node_a_id in self._topology:
            self._topology[node_a_id].append(node_b_id)
        if node_b_id in self._topology:
            self._topology[node_b_id].append(node_a_id)

    def get_node(self, node_id: str) -> Optional[HardwareNode]:
        return self.nodes.get(node_id)

    def all_neurons(self) -> List[Neuron]:
        """Return all neurons across all nodes."""
        neurons = []
        for node in self.nodes.values():
            neurons.extend(node.neurons)
        return neurons

    def persist_all_weights(self, synapses: list) -> None:
        """
        Persist synaptic weights to NVMe nodes for long-term storage.

        In a real deployment this serialises weights to disk; here it
        simulates I/O latency and tracks write counts.
        """
        nvme_nodes = [n for n in self.nodes.values() if n.spec.node_type == NodeType.NVME_STORAGE]
        if not nvme_nodes:
            return

        weight_dict = {s.synapse_id: s.config.weight for s in synapses}
        target_node = nvme_nodes[0]
        target_node.nvme_persist(weight_dict)

    def cluster_status(self) -> List[Dict]:
        """Return status for all nodes."""
        return [node.status() for node in self.nodes.values()]

    def create_default_array(self, num_cm4_nodes: int = 4) -> "HardwareLayer":
        """
        Factory: build the default Quantum Neural Brain hardware array.

        Topology:
          4× CM4 compute nodes (ring topology)
          1× DRAM cache node (connected to all CM4 nodes)
          1× NVMe storage node (connected to DRAM)
        """
        # CM4 compute nodes
        for i in range(num_cm4_nodes):
            node = HardwareNode(
                node_id=f"cm4_{i}",
                spec=HardwareSpec(node_type=NodeType.CM4_COMPUTE),
            )
            self.add_node(node)

        # DRAM cache node
        dram_node = HardwareNode(
            node_id="dram_0",
            spec=HardwareSpec(
                node_type=NodeType.DRAM_CACHE,
                cpu_cores=1,
                clock_speed_mhz=3200.0,
                ram_mb=65536,
            ),
        )
        self.add_node(dram_node)

        # NVMe storage node
        nvme_node = HardwareNode(
            node_id="nvme_0",
            spec=HardwareSpec(
                node_type=NodeType.NVME_STORAGE,
                cpu_cores=1,
                clock_speed_mhz=1000.0,
                ram_mb=4096,
            ),
        )
        self.add_node(nvme_node)

        # Connect CM4 nodes in a ring
        for i in range(num_cm4_nodes):
            self.connect_nodes(f"cm4_{i}", f"cm4_{(i + 1) % num_cm4_nodes}")

        # Connect all CM4 nodes to DRAM
        for i in range(num_cm4_nodes):
            self.connect_nodes(f"cm4_{i}", "dram_0")

        # Connect DRAM to NVMe
        self.connect_nodes("dram_0", "nvme_0")

        return self

    def __repr__(self) -> str:
        return f"HardwareLayer(nodes={list(self.nodes.keys())})"
