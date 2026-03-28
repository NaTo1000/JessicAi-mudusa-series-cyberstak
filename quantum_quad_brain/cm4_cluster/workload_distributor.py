"""
Workload Distributor
====================
Distributes quantum inference tasks across available CM4 compute nodes
using configurable scheduling strategies.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from .cluster_manager import CM4ClusterManager, CM4Node, NodeState

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    ROUND_ROBIN = auto()
    LEAST_LOADED = auto()
    RANDOM = auto()


@dataclass
class Task:
    """A unit of work to be executed on a CM4 node."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Any = None
    priority: int = 5           # 1 (highest) – 10 (lowest)
    created_at: float = field(default_factory=time.time)
    assigned_to: Optional[str] = None
    completed: bool = False
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "priority": self.priority,
            "assigned_to": self.assigned_to,
            "completed": self.completed,
            "error": self.error,
        }


class WorkloadDistributor:
    """
    Schedules and dispatches ``Task`` objects to CM4 nodes.

    Parameters
    ----------
    cluster : CM4ClusterManager
        The cluster whose nodes will receive tasks.
    strategy : SchedulingStrategy
        Scheduling algorithm to use when selecting a target node.
    max_retries : int
        Number of times to retry a failed task on an alternative node.
    """

    def __init__(
        self,
        cluster: CM4ClusterManager,
        strategy: SchedulingStrategy = SchedulingStrategy.LEAST_LOADED,
        max_retries: int = 3,
    ) -> None:
        self.cluster = cluster
        self.strategy = strategy
        self.max_retries = max_retries
        self._rr_index: int = 0
        self._pending: Dict[str, Task] = {}
        self._completed: Dict[str, Task] = {}

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    def submit(
        self,
        payload: Any,
        priority: int = 5,
        executor: Optional[Callable[[CM4Node, Any], Any]] = None,
    ) -> Task:
        """
        Submit a task for execution on the best available node.

        Parameters
        ----------
        payload : Any
            The data / function arguments passed to the executor.
        priority : int
            Task priority (1 = highest, 10 = lowest).
        executor : callable, optional
            ``fn(node, payload) -> result`` called on the chosen node.
            If ``None`` a no-op executor is used (useful for testing).
        """
        task = Task(payload=payload, priority=priority)
        self._pending[task.task_id] = task

        for attempt in range(1, self.max_retries + 1):
            node = self._select_node()
            if node is None:
                task.error = "No available nodes."
                logger.error("Task %s: no available node on attempt %d.", task.task_id[:8], attempt)
                break

            task.assigned_to = node.node_id
            node.state = NodeState.BUSY
            try:
                if executor is not None:
                    task.result = executor(node, payload)
                else:
                    task.result = {"simulated": True, "node": node.node_id}
                task.completed = True
                node.tasks_completed += 1
                logger.info(
                    "Task %s completed on node '%s' (attempt %d).",
                    task.task_id[:8],
                    node.node_id,
                    attempt,
                )
                break
            except Exception as exc:  # noqa: BLE001
                task.error = str(exc)
                logger.warning(
                    "Task %s failed on node '%s': %s (attempt %d/%d).",
                    task.task_id[:8],
                    node.node_id,
                    exc,
                    attempt,
                    self.max_retries,
                    exc_info=True,
                )
            finally:
                node.state = NodeState.IDLE

        del self._pending[task.task_id]
        self._completed[task.task_id] = task
        return task

    def submit_batch(
        self,
        payloads: List[Any],
        priority: int = 5,
        executor: Optional[Callable[[CM4Node, Any], Any]] = None,
    ) -> List[Task]:
        """Submit multiple tasks and return results in order."""
        return [self.submit(p, priority=priority, executor=executor) for p in payloads]

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _select_node(self) -> Optional[CM4Node]:
        available = self.cluster.available_nodes
        if not available:
            return None

        if self.strategy == SchedulingStrategy.ROUND_ROBIN:
            node = available[self._rr_index % len(available)]
            self._rr_index += 1
            return node

        if self.strategy == SchedulingStrategy.LEAST_LOADED:
            return min(available, key=lambda n: n.cpu_utilisation)

        # RANDOM
        import random
        return random.choice(available)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def distributor_summary(self) -> Dict[str, Any]:
        failed = sum(1 for t in self._completed.values() if not t.completed)
        return {
            "strategy": self.strategy.name,
            "pending": self.pending_count,
            "completed": self.completed_count,
            "failed": failed,
            "success_rate": round(
                (self.completed_count - failed) / self.completed_count, 4
            ) if self.completed_count else 0.0,
        }
