"""
Resource Monitor – continuously polls CPU/RAM and enforces the 5 % per-node cap.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import psutil

logger = logging.getLogger(__name__)

# Hard ceiling: no single hive node may consume more than this fraction of the
# host's resources.  Value is intentionally conservative so bystander devices
# are never noticeably impacted.
MAX_CPU_PERCENT: float = 5.0
MAX_RAM_PERCENT: float = 5.0

POLL_INTERVAL_SECS: float = 1.0


@dataclass
class ResourceSnapshot:
    """Point-in-time view of host resources."""

    timestamp: float = field(default_factory=time.monotonic)
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_available_mb: float = 0.0
    cpu_count: int = 1
    load_avg_1m: float = 0.0

    @property
    def headroom_cpu(self) -> float:
        """Percentage points of CPU still available under the cap."""
        return max(0.0, MAX_CPU_PERCENT - self.cpu_percent)

    @property
    def headroom_ram(self) -> float:
        """Percentage points of RAM still available under the cap."""
        return max(0.0, MAX_RAM_PERCENT - self.ram_percent)

    @property
    def is_within_cap(self) -> bool:
        return self.cpu_percent <= MAX_CPU_PERCENT and self.ram_percent <= MAX_RAM_PERCENT

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": round(self.cpu_percent, 3),
            "ram_percent": round(self.ram_percent, 3),
            "ram_available_mb": round(self.ram_available_mb, 1),
            "cpu_count": self.cpu_count,
            "load_avg_1m": round(self.load_avg_1m, 3),
            "headroom_cpu": round(self.headroom_cpu, 3),
            "headroom_ram": round(self.headroom_ram, 3),
        }


class ResourceMonitor:
    """
    Asynchronous resource monitor.

    * Continuously samples CPU + RAM usage attributed to the hive process group.
    * Notifies registered callbacks when usage exceeds the cap so the scheduler
      can throttle or shed tasks immediately.
    * Exposes the latest snapshot for peer advertisement via the mesh layer.
    """

    def __init__(
        self,
        poll_interval: float = POLL_INTERVAL_SECS,
        max_cpu: float = MAX_CPU_PERCENT,
        max_ram: float = MAX_RAM_PERCENT,
        process: Optional[psutil.Process] = None,
    ) -> None:
        self._poll_interval = poll_interval
        self._max_cpu = max_cpu
        self._max_ram = max_ram
        self._process = process or psutil.Process()
        self._latest: ResourceSnapshot = ResourceSnapshot()
        self._callbacks: List[Callable[[ResourceSnapshot], None]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def snapshot(self) -> ResourceSnapshot:
        """Return the most recent resource snapshot (thread-safe read)."""
        return self._latest

    def register_callback(self, fn: Callable[[ResourceSnapshot], None]) -> None:
        """Register a function to call whenever a new snapshot is captured."""
        self._callbacks.append(fn)

    async def start(self) -> None:
        """Begin the background polling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("ResourceMonitor started (interval=%.1fs)", self._poll_interval)

    async def stop(self) -> None:
        """Gracefully stop the polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ResourceMonitor stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                snap = self._sample()
                self._latest = snap
                if not snap.is_within_cap:
                    logger.warning(
                        "Resource cap exceeded – cpu=%.2f%% ram=%.2f%%",
                        snap.cpu_percent,
                        snap.ram_percent,
                    )
                for cb in self._callbacks:
                    try:
                        cb(snap)
                    except Exception:
                        logger.exception("Callback error in ResourceMonitor")
            except Exception:
                logger.exception("Sampling error in ResourceMonitor")
            await asyncio.sleep(self._poll_interval)

    def _sample(self) -> ResourceSnapshot:
        """Collect a single resource snapshot for this process + children."""
        try:
            children = self._process.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []

        procs = [self._process] + children
        total_cpu = 0.0
        total_mem_bytes = 0

        for p in procs:
            try:
                with p.oneshot():
                    total_cpu += p.cpu_percent(interval=None)
                    total_mem_bytes += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        mem = psutil.virtual_memory()
        ram_pct = (total_mem_bytes / mem.total * 100) if mem.total else 0.0

        try:
            load_avg = psutil.getloadavg()[0]
        except AttributeError:
            load_avg = 0.0

        return ResourceSnapshot(
            cpu_percent=min(total_cpu, 100.0),
            ram_percent=round(ram_pct, 3),
            ram_available_mb=mem.available / 1_048_576,
            cpu_count=psutil.cpu_count(logical=True) or 1,
            load_avg_1m=load_avg,
        )
