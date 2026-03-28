"""
CM4 Cluster Manager
===================
Manages a cluster of Raspberry Pi CM4 compute modules, providing
health monitoring, node registration, and seamless communication
for the Quantum Quad-Brain distributed processing array.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeState(Enum):
    IDLE = auto()
    BUSY = auto()
    DEGRADED = auto()
    OFFLINE = auto()


@dataclass
class CM4Node:
    """Represents a single Raspberry Pi CM4 compute module."""

    node_id: str
    ip_address: str
    cpu_cores: int = 4          # BCM2711 quad-core Cortex-A72
    ram_gb: float = 8.0         # max CM4 variant
    has_emmc: bool = True
    state: NodeState = NodeState.IDLE
    cpu_utilisation: float = 0.0   # 0 – 100 %
    ram_utilisation: float = 0.0   # 0 – 100 %
    tasks_completed: int = 0
    last_heartbeat: float = field(default_factory=time.time)

    def is_available(self) -> bool:
        return self.state in (NodeState.IDLE, NodeState.DEGRADED)

    def heartbeat(self) -> None:
        self.last_heartbeat = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "ip_address": self.ip_address,
            "state": self.state.name,
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "cpu_utilisation": round(self.cpu_utilisation, 2),
            "ram_utilisation": round(self.ram_utilisation, 2),
            "tasks_completed": self.tasks_completed,
        }


class CM4ClusterManager:
    """
    Cluster manager for a pool of CM4 compute nodes.

    Responsibilities
    ----------------
    * Node registration / deregistration (plug-and-play).
    * Periodic heartbeat tracking and automatic OFFLINE detection.
    * Health aggregation and cluster-level metrics.

    Parameters
    ----------
    heartbeat_timeout_s : float
        Seconds without a heartbeat before a node is marked OFFLINE.
    """

    def __init__(self, heartbeat_timeout_s: float = 30.0) -> None:
        self._nodes: Dict[str, CM4Node] = {}
        self.heartbeat_timeout_s = heartbeat_timeout_s

    # ------------------------------------------------------------------
    # Node lifecycle
    # ------------------------------------------------------------------

    def register_node(self, node: CM4Node) -> None:
        """Add a CM4 node to the cluster (hot-plug)."""
        if node.node_id in self._nodes:
            raise ValueError(f"Node '{node.node_id}' already registered.")
        self._nodes[node.node_id] = node
        logger.info(
            "CM4 node '%s' registered (IP=%s, RAM=%.1f GB).",
            node.node_id,
            node.ip_address,
            node.ram_gb,
        )

    def deregister_node(self, node_id: str) -> None:
        """Remove a CM4 node (hot-unplug with graceful drain)."""
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in cluster.")
        node = self._nodes.pop(node_id)
        logger.warning("CM4 node '%s' deregistered.", node.node_id)

    def receive_heartbeat(self, node_id: str, cpu: float = 0.0, ram: float = 0.0) -> None:
        """Update a node's last-seen timestamp and resource utilisation."""
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"Unknown node '{node_id}'.")
        node.heartbeat()
        node.cpu_utilisation = cpu
        node.ram_utilisation = ram
        if node.state == NodeState.OFFLINE:
            node.state = NodeState.IDLE
            logger.info("CM4 node '%s' recovered (back ONLINE).", node_id)

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    def check_node_health(self) -> Dict[str, str]:
        """
        Evaluate freshness of heartbeats and update node states.
        Returns a mapping of {node_id: state_name}.
        """
        now = time.time()
        states: Dict[str, str] = {}
        for node_id, node in self._nodes.items():
            age = now - node.last_heartbeat
            if age > self.heartbeat_timeout_s and node.state != NodeState.OFFLINE:
                node.state = NodeState.OFFLINE
                logger.error("CM4 node '%s' is OFFLINE (no heartbeat for %.1f s).", node_id, age)
            states[node_id] = node.state.name
        return states

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> List[CM4Node]:
        return list(self._nodes.values())

    @property
    def available_nodes(self) -> List[CM4Node]:
        return [n for n in self._nodes.values() if n.is_available()]

    @property
    def online_count(self) -> int:
        return sum(1 for n in self._nodes.values() if n.state != NodeState.OFFLINE)

    def get_node(self, node_id: str) -> CM4Node:
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found.")
        return self._nodes[node_id]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def cluster_summary(self) -> Dict[str, Any]:
        nodes = list(self._nodes.values())
        online = [n for n in nodes if n.state != NodeState.OFFLINE]
        avg_cpu = sum(n.cpu_utilisation for n in online) / len(online) if online else 0.0
        avg_ram = sum(n.ram_utilisation for n in online) / len(online) if online else 0.0
        return {
            "total_nodes": len(nodes),
            "online_nodes": len(online),
            "available_nodes": len(self.available_nodes),
            "avg_cpu_utilisation": round(avg_cpu, 2),
            "avg_ram_utilisation": round(avg_ram, 2),
            "total_tasks_completed": sum(n.tasks_completed for n in nodes),
        }
