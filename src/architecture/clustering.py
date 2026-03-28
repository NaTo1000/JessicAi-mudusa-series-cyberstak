"""
Cluster Regulation Framework
==============================
Manages the nested cluster / sub-cluster hierarchy that regulates
resource allocation across the superconductor and tesseract layers.

Design
------
* **ClusterNode** – an individual processing node (maps to one or more
  superconductor layers).
* **SubCluster** – a group of ``ClusterNode`` instances sharing a local
  resource budget.
* **ClusterRegulator** – the top-level orchestrator that monitors all
  sub-clusters, rebalances load, and quarantines misbehaving nodes.

Self-regulation
---------------
Each ``SubCluster`` continuously samples its aggregate load.  When load
exceeds ``high_threshold`` it spawns a *split*: excess nodes are migrated
to a new sub-cluster.  When load drops below ``low_threshold`` the
sub-cluster *merges* with an adjacent peer to avoid fragmentation.

The regulator enforces a global ceiling so no single sub-cluster can claim
more than its fair share of the total budget.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations & constants
# ---------------------------------------------------------------------------

class NodeState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    OVERLOADED = auto()
    QUARANTINED = auto()


class ClusterEvent(Enum):
    NODE_ADDED = auto()
    NODE_REMOVED = auto()
    SPLIT = auto()
    MERGE = auto()
    QUARANTINE = auto()
    REBALANCE = auto()


# ---------------------------------------------------------------------------
# Cluster node
# ---------------------------------------------------------------------------

@dataclass
class ClusterNode:
    """
    A single processing node within a sub-cluster.

    Attributes
    ----------
    node_id:
        Globally unique identifier.
    superconductor_layer_index:
        Index of the associated superconductor layer (if any).
    cpu_budget:
        Fraction of total CPU allocated to this node (0.0–1.0).
    current_load:
        Current utilisation as a fraction (0.0–1.0).
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    superconductor_layer_index: Optional[int] = None
    cpu_budget: float = 0.05           # default 5 % of host CPU
    current_load: float = 0.0
    state: NodeState = NodeState.IDLE
    _last_tick: float = field(default_factory=time.time)

    def tick(self, load_delta: float = 0.0) -> None:
        """Advance the node by one monitoring cycle."""
        self.current_load = max(0.0, min(1.0, self.current_load + load_delta))
        if self.current_load > 0.9:
            self.state = NodeState.OVERLOADED
        elif self.current_load > 0.01:
            self.state = NodeState.ACTIVE
        else:
            self.state = NodeState.IDLE
        self._last_tick = time.time()

    def quarantine(self) -> None:
        self.state = NodeState.QUARANTINED
        self.current_load = 0.0

    def status(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "state": self.state.name,
            "cpu_budget": self.cpu_budget,
            "current_load": round(self.current_load, 4),
            "sc_layer": self.superconductor_layer_index,
        }


# ---------------------------------------------------------------------------
# Sub-cluster
# ---------------------------------------------------------------------------

@dataclass
class SubClusterConfig:
    """Thresholds driving self-regulation within a sub-cluster."""
    high_threshold: float = 0.80    # aggregate load above which a split occurs
    low_threshold: float = 0.20     # aggregate load below which merge is proposed
    max_nodes: int = 64


class SubCluster:
    """
    A self-regulating group of ``ClusterNode`` instances.
    """

    def __init__(
        self,
        config: Optional[SubClusterConfig] = None,
        cluster_id: Optional[str] = None,
    ) -> None:
        self.config = config or SubClusterConfig()
        self.cluster_id = cluster_id or str(uuid.uuid4())
        self._nodes: Dict[str, ClusterNode] = {}
        self._event_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, node: Optional[ClusterNode] = None) -> ClusterNode:
        """Add *node* (or a new default node) to this sub-cluster."""
        if node is None:
            node = ClusterNode()
        if len(self._nodes) >= self.config.max_nodes:
            raise ValueError(
                f"SubCluster {self.cluster_id!r} is at capacity "
                f"({self.config.max_nodes} nodes)."
            )
        self._nodes[node.node_id] = node
        self._log(ClusterEvent.NODE_ADDED, node_id=node.node_id)
        return node

    def remove_node(self, node_id: str) -> Optional[ClusterNode]:
        node = self._nodes.pop(node_id, None)
        if node:
            self._log(ClusterEvent.NODE_REMOVED, node_id=node_id)
        return node

    def quarantine_node(self, node_id: str) -> None:
        node = self._nodes.get(node_id)
        if node:
            node.quarantine()
            self._log(ClusterEvent.QUARANTINE, node_id=node_id)

    # ------------------------------------------------------------------
    # Tick / monitoring
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance all nodes by one monitoring cycle."""
        for node in self._nodes.values():
            if node.state != NodeState.QUARANTINED:
                node.tick()

    # ------------------------------------------------------------------
    # Self-regulation
    # ------------------------------------------------------------------

    @property
    def aggregate_load(self) -> float:
        active = [
            n for n in self._nodes.values()
            if n.state != NodeState.QUARANTINED
        ]
        if not active:
            return 0.0
        return sum(n.current_load for n in active) / len(active)

    def should_split(self) -> bool:
        return self.aggregate_load > self.config.high_threshold

    def should_merge(self) -> bool:
        return self.aggregate_load < self.config.low_threshold

    def split(self) -> "SubCluster":
        """
        Migrate the most-loaded half of nodes to a new ``SubCluster``.

        Returns
        -------
        SubCluster
            The newly created sibling sub-cluster.
        """
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: n.current_load,
            reverse=True,
        )
        half = len(sorted_nodes) // 2
        sibling = SubCluster(config=self.config)
        for node in sorted_nodes[:half]:
            self._nodes.pop(node.node_id)
            sibling._nodes[node.node_id] = node
        self._log(ClusterEvent.SPLIT, sibling_id=sibling.cluster_id)
        return sibling

    def absorb(self, other: "SubCluster") -> None:
        """Merge all nodes from *other* into this sub-cluster."""
        for node in list(other._nodes.values()):
            if len(self._nodes) < self.config.max_nodes:
                self._nodes[node.node_id] = node
        self._log(ClusterEvent.MERGE, absorbed_id=other.cluster_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, event: ClusterEvent, **kwargs: Any) -> None:
        self._event_log.append({
            "event": event.name,
            "cluster_id": self.cluster_id,
            "at": time.time(),
            **kwargs,
        })

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def status(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "node_count": self.node_count,
            "aggregate_load": round(self.aggregate_load, 4),
            "should_split": self.should_split(),
            "should_merge": self.should_merge(),
        }


# ---------------------------------------------------------------------------
# Cluster regulator
# ---------------------------------------------------------------------------

@dataclass
class RegulatorConfig:
    """Configuration for the top-level cluster regulator."""
    max_sub_clusters: int = 1_024
    global_load_ceiling: float = 0.90
    rebalance_interval_s: float = 5.0


class ClusterRegulator:
    """
    Top-level orchestrator for the full cluster hierarchy.

    Responsibilities
    ----------------
    * Creates and destroys sub-clusters on demand.
    * Drives periodic rebalancing: splits overloaded clusters, merges under-
      utilised ones.
    * Quarantines misbehaving nodes and reassigns their load.
    * Reports aggregate system health.
    """

    def __init__(self, config: Optional[RegulatorConfig] = None) -> None:
        self.config = config or RegulatorConfig()
        self.regulator_id = str(uuid.uuid4())
        self._sub_clusters: Dict[str, SubCluster] = {}
        self._event_log: List[Dict[str, Any]] = []
        self._last_rebalance: float = 0.0

    # ------------------------------------------------------------------
    # Sub-cluster lifecycle
    # ------------------------------------------------------------------

    def create_sub_cluster(
        self,
        node_count: int = 4,
        sub_config: Optional[SubClusterConfig] = None,
    ) -> SubCluster:
        """Create a new sub-cluster and register it with the regulator."""
        if len(self._sub_clusters) >= self.config.max_sub_clusters:
            raise RuntimeError("Maximum sub-cluster count reached.")
        sc = SubCluster(config=sub_config)
        for _ in range(node_count):
            sc.add_node()
        self._sub_clusters[sc.cluster_id] = sc
        return sc

    def remove_sub_cluster(self, cluster_id: str) -> Optional[SubCluster]:
        return self._sub_clusters.pop(cluster_id, None)

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(self) -> Dict[str, int]:
        """
        Execute one rebalancing pass across all sub-clusters.

        Returns
        -------
        Dict[str, int]
            Counts of splits and merges performed.
        """
        splits = 0
        merges = 0

        clusters = list(self._sub_clusters.values())

        # Split overloaded clusters
        new_clusters: List[SubCluster] = []
        for sc in clusters:
            if sc.should_split() and len(self._sub_clusters) < self.config.max_sub_clusters:
                sibling = sc.split()
                new_clusters.append(sibling)
                splits += 1

        for sc in new_clusters:
            self._sub_clusters[sc.cluster_id] = sc

        # Merge under-utilised pairs
        idle = [sc for sc in list(self._sub_clusters.values()) if sc.should_merge()]
        while len(idle) >= 2:
            a, b = idle.pop(0), idle.pop(0)
            a.absorb(b)
            self._sub_clusters.pop(b.cluster_id, None)
            merges += 1

        self._last_rebalance = time.time()
        self._event_log.append({
            "event": ClusterEvent.REBALANCE.name,
            "splits": splits,
            "merges": merges,
            "at": self._last_rebalance,
        })
        return {"splits": splits, "merges": merges}

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance all sub-clusters and conditionally rebalance."""
        for sc in self._sub_clusters.values():
            sc.tick()
        now = time.time()
        if now - self._last_rebalance >= self.config.rebalance_interval_s:
            self.rebalance()

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    @property
    def sub_cluster_count(self) -> int:
        return len(self._sub_clusters)

    @property
    def total_nodes(self) -> int:
        return sum(sc.node_count for sc in self._sub_clusters.values())

    @property
    def global_load(self) -> float:
        if not self._sub_clusters:
            return 0.0
        loads = [sc.aggregate_load for sc in self._sub_clusters.values()]
        return sum(loads) / len(loads)

    def status(self) -> Dict[str, Any]:
        return {
            "regulator_id": self.regulator_id,
            "sub_clusters": self.sub_cluster_count,
            "total_nodes": self.total_nodes,
            "global_load": round(self.global_load, 4),
            "last_rebalance": self._last_rebalance,
        }
