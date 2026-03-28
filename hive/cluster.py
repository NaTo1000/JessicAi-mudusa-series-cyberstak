"""
HiveCluster – manages the full set of nodes, performs load balancing,
and provides self-healing when a node disappears.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .node import HiveNode, NodeInfo, TaskPacket, TaskResult

logger = logging.getLogger(__name__)

STALE_NODE_SECS: float = 30.0   # remove peers not heard from within this window
REBALANCE_INTERVAL: float = 2.0  # seconds between scheduler rebalance passes


@dataclass
class ClusterStats:
    total_nodes: int = 0
    healthy_nodes: int = 0
    active_tasks: int = 0
    avg_cpu_headroom: float = 0.0
    avg_ram_headroom: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    uptime_secs: float = 0.0


class HiveCluster:
    """
    Central coordination layer for all hive nodes.

    The cluster does NOT hold compute resources itself – it routes tasks to
    the node with the most available headroom (weighted round-robin with
    resource awareness).

    Self-healing: nodes that stop sending heartbeats are evicted; the
    scheduler automatically redistributes in-flight tasks.
    """

    def __init__(self, local_node: Optional[HiveNode] = None) -> None:
        self._local_node = local_node
        self._peers: Dict[str, NodeInfo] = {}
        self._running = False
        self._stats = ClusterStats()
        self._start_time: float = 0.0
        self._completed = 0
        self._failed = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        if self._local_node:
            await self._local_node.start()
        self._running = True
        self._start_time = time.monotonic()
        asyncio.create_task(self._gc_loop())
        logger.info("HiveCluster started")

    async def stop(self) -> None:
        self._running = False
        if self._local_node:
            await self._local_node.stop()
        logger.info("HiveCluster stopped")

    # ------------------------------------------------------------------
    # Peer management (called by the mesh layer)
    # ------------------------------------------------------------------

    def register_peer(self, info: NodeInfo) -> None:
        """Record or update a peer's advertisement."""
        info.last_seen = time.monotonic()
        self._peers[info.node_id] = info
        logger.debug("Peer registered/updated: %s", info.node_id[:8])

    def remove_peer(self, node_id: str) -> None:
        self._peers.pop(node_id, None)
        logger.info("Peer removed: %s", node_id[:8])

    # ------------------------------------------------------------------
    # Task dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, packet: TaskPacket) -> Optional[TaskResult]:
        """
        Choose the best available node and run the task.

        Selection policy: node with the greatest combined CPU + RAM headroom,
        provided it is not at cap.  Falls back to the local node if no peers
        are available.
        """
        target = self._best_node()
        if target is None:
            logger.warning("No available nodes for task %s", packet.task_id[:8])
            self._failed += 1
            return None

        result = await self._send_to(target, packet)
        if result and not result.error:
            self._completed += 1
        else:
            self._failed += 1
        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> ClusterStats:
        peers = list(self._peers.values())
        healthy = [p for p in peers if self._is_healthy(p)]
        return ClusterStats(
            total_nodes=len(peers) + (1 if self._local_node else 0),
            healthy_nodes=len(healthy) + (1 if self._local_node else 0),
            active_tasks=sum(p.active_tasks for p in peers),
            avg_cpu_headroom=(
                sum(p.cpu_headroom for p in healthy) / len(healthy) if healthy else 0.0
            ),
            avg_ram_headroom=(
                sum(p.ram_headroom for p in healthy) / len(healthy) if healthy else 0.0
            ),
            tasks_completed=self._completed,
            tasks_failed=self._failed,
            uptime_secs=time.monotonic() - self._start_time,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _best_node(self) -> Optional[NodeInfo]:
        candidates = [
            p for p in self._peers.values()
            if self._is_healthy(p) and p.cpu_headroom > 0 and p.ram_headroom > 0
        ]
        if not candidates:
            # Try local node
            if self._local_node and not self._local_node._cap_exceeded:
                return self._local_node.info
            return None
        return max(candidates, key=lambda p: p.cpu_headroom + p.ram_headroom)

    def _is_healthy(self, info: NodeInfo) -> bool:
        return (time.monotonic() - info.last_seen) < STALE_NODE_SECS

    async def _send_to(self, target: NodeInfo, packet: TaskPacket) -> Optional[TaskResult]:
        """Send task to a peer node over TCP, or execute locally."""
        if self._local_node and target.node_id == self._local_node.node_id:
            return await self._local_node.submit_task(packet)

        try:
            import json
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target.host, target.port), timeout=5.0
            )
            payload = json.dumps({
                "task_id": packet.task_id,
                "payload": packet.payload.decode(),
                "priority": packet.priority,
                "checksum": packet.checksum,
            })
            writer.write(payload.encode())
            await writer.drain()
            raw = await asyncio.wait_for(reader.read(65536), timeout=STALE_NODE_SECS)
            writer.close()
            data = json.loads(raw)
            return TaskResult(**data)
        except Exception as exc:
            logger.warning(
                "Failed to send task to %s: %s", target.node_id[:8], exc
            )
            return None

    async def _gc_loop(self) -> None:
        """Periodically evict stale peers (self-healing)."""
        while self._running:
            stale = [
                nid
                for nid, info in self._peers.items()
                if not self._is_healthy(info)
            ]
            for nid in stale:
                logger.info("Evicting stale node %s", nid[:8])
                self.remove_peer(nid)
            await asyncio.sleep(STALE_NODE_SECS / 3)
