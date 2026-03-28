"""
TaskScheduler – priority queue with dynamic resource-aware dispatching.

Tasks are queued, prioritised, and dispatched to the HiveCluster.
When a node rejects a task (cap exceeded) the scheduler retries on the
next best candidate.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .cluster import HiveCluster
from .node import TaskPacket, TaskResult

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 3
QUEUE_POLL_SECS: float = 0.1


@dataclass(order=True)
class _QueuedTask:
    priority: int
    submitted_at: float = field(compare=False)
    packet: TaskPacket = field(compare=False)
    retries: int = field(default=0, compare=False)


class TaskScheduler:
    """
    Priority-based scheduler that feeds tasks to the HiveCluster.

    Lower priority values run first.  Tasks that exceed MAX_RETRIES are
    dropped with a warning.
    """

    def __init__(self, cluster: HiveCluster) -> None:
        self._cluster = cluster
        self._heap: List[_QueuedTask] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop())
        logger.info("TaskScheduler started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TaskScheduler stopped (queue depth=%d)", len(self._heap))

    def enqueue(self, packet: TaskPacket) -> None:
        """Add a task to the priority queue."""
        item = _QueuedTask(
            priority=packet.priority,
            submitted_at=packet.submitted_at,
            packet=packet,
        )
        heapq.heappush(self._heap, item)
        logger.debug("Enqueued task %s (priority=%d)", packet.task_id[:8], packet.priority)

    @property
    def queue_depth(self) -> int:
        return len(self._heap)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        while self._running:
            if not self._heap:
                await asyncio.sleep(QUEUE_POLL_SECS)
                continue

            item = heapq.heappop(self._heap)
            asyncio.create_task(self._try_dispatch(item))

    async def _try_dispatch(self, item: _QueuedTask) -> Optional[TaskResult]:
        result = await self._cluster.dispatch(item.packet)

        if result is None or result.error == "cap_exceeded":
            if item.retries < MAX_RETRIES:
                item.retries += 1
                wait = 0.5 * (2 ** item.retries)  # exponential back-off
                logger.debug(
                    "Retrying task %s in %.1fs (attempt %d/%d)",
                    item.packet.task_id[:8], wait, item.retries, MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                heapq.heappush(self._heap, item)
            else:
                logger.warning(
                    "Task %s dropped after %d retries",
                    item.packet.task_id[:8], MAX_RETRIES,
                )
        return result
