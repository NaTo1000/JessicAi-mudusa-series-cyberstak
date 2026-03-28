"""
Performance Monitor
===================
Collects, aggregates, and exposes real-time performance metrics for
the Quantum Quad-Brain compute array, covering CPU, storage, cache,
and energy dimensions.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Number of historical samples retained for sliding-window averages
HISTORY_SIZE = 300


@dataclass
class Sample:
    """A single point-in-time snapshot of array performance."""

    timestamp: float = field(default_factory=time.time)
    cpu_utilisation: float = 0.0    # 0–100 %
    ram_utilisation: float = 0.0    # 0–100 %
    nvme_throughput_gbps: float = 0.0
    cache_hit_ratio: float = 0.0
    active_nodes: int = 0
    inference_latency_ms: float = 0.0
    power_watts: float = 0.0        # estimated


class PerformanceMonitor:
    """
    Sliding-window performance monitor for the compute array.

    The monitor maintains a fixed-size deque of ``Sample`` objects
    and exposes convenience methods for current, average, and peak
    metrics, plus an energy-efficiency KPI.

    Parameters
    ----------
    window_size : int
        Number of samples retained in the sliding window.
    sample_interval_s : float
        Target interval between automatic samples (informational; actual
        collection is driven by ``record()`` calls).
    """

    def __init__(
        self,
        window_size: int = HISTORY_SIZE,
        sample_interval_s: float = 1.0,
    ) -> None:
        self.window_size = window_size
        self.sample_interval_s = sample_interval_s
        self._history: Deque[Sample] = deque(maxlen=window_size)
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Sample recording
    # ------------------------------------------------------------------

    def record(
        self,
        cpu_utilisation: float = 0.0,
        ram_utilisation: float = 0.0,
        nvme_throughput_gbps: float = 0.0,
        cache_hit_ratio: float = 0.0,
        active_nodes: int = 0,
        inference_latency_ms: float = 0.0,
        power_watts: float = 0.0,
    ) -> Sample:
        """Record a new performance sample and append it to the window."""
        sample = Sample(
            cpu_utilisation=cpu_utilisation,
            ram_utilisation=ram_utilisation,
            nvme_throughput_gbps=nvme_throughput_gbps,
            cache_hit_ratio=cache_hit_ratio,
            active_nodes=active_nodes,
            inference_latency_ms=inference_latency_ms,
            power_watts=power_watts,
        )
        self._history.append(sample)
        return sample

    def record_from_system(self, quad_brain: Any) -> Sample:
        """
        Convenience method: pull metrics directly from a
        ``QuantumQuadBrain`` instance and record a sample.
        """
        summary = quad_brain.system_summary()
        cluster = summary.get("cluster", {})
        storage = summary.get("storage", {})
        cache = summary.get("cache", {})

        active_nodes = cluster.get("online_nodes", 0)
        avg_cpu = cluster.get("avg_cpu_utilisation", 0.0)
        avg_ram = cluster.get("avg_ram_utilisation", 0.0)
        throughput = storage.get("throughput_gbps", 0.0)
        hit_ratio = cache.get("hit_ratio", 0.0)
        avg_latency = sum(
            b.get("avg_latency_ms", 0.0)
            for b in summary.get("brains", {}).values()
        ) / max(len(summary.get("brains", {})), 1)

        # Rough power estimate: 5 W idle + 2 W per active node
        power = 5.0 + active_nodes * 2.0

        return self.record(
            cpu_utilisation=avg_cpu,
            ram_utilisation=avg_ram,
            nvme_throughput_gbps=throughput,
            cache_hit_ratio=hit_ratio,
            active_nodes=active_nodes,
            inference_latency_ms=avg_latency,
            power_watts=power,
        )

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    @property
    def latest(self) -> Optional[Sample]:
        return self._history[-1] if self._history else None

    def _avg(self, attr: str) -> float:
        if not self._history:
            return 0.0
        return sum(getattr(s, attr) for s in self._history) / len(self._history)

    def _peak(self, attr: str) -> float:
        if not self._history:
            return 0.0
        return max(getattr(s, attr) for s in self._history)

    @property
    def avg_cpu(self) -> float:
        return self._avg("cpu_utilisation")

    @property
    def avg_latency_ms(self) -> float:
        return self._avg("inference_latency_ms")

    @property
    def peak_throughput_gbps(self) -> float:
        return self._peak("nvme_throughput_gbps")

    @property
    def avg_cache_hit_ratio(self) -> float:
        return self._avg("cache_hit_ratio")

    @property
    def energy_efficiency(self) -> float:
        """Inferences per watt (samples × nodes / total energy)."""
        total_inferences = sum(s.active_nodes for s in self._history)
        total_watts = sum(s.power_watts for s in self._history)
        return total_inferences / total_watts if total_watts else 0.0

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a comprehensive performance snapshot dict."""
        uptime = time.time() - self._start_time
        latest = self.latest
        return {
            "uptime_s": round(uptime, 1),
            "samples_collected": len(self._history),
            "current": {
                "cpu_utilisation": round(latest.cpu_utilisation, 2) if latest else 0.0,
                "ram_utilisation": round(latest.ram_utilisation, 2) if latest else 0.0,
                "nvme_throughput_gbps": round(latest.nvme_throughput_gbps, 2) if latest else 0.0,
                "cache_hit_ratio": round(latest.cache_hit_ratio, 4) if latest else 0.0,
                "inference_latency_ms": round(latest.inference_latency_ms, 2) if latest else 0.0,
                "power_watts": round(latest.power_watts, 2) if latest else 0.0,
            },
            "averages": {
                "cpu_utilisation": round(self.avg_cpu, 2),
                "inference_latency_ms": round(self.avg_latency_ms, 2),
                "cache_hit_ratio": round(self.avg_cache_hit_ratio, 4),
            },
            "peaks": {
                "nvme_throughput_gbps": round(self.peak_throughput_gbps, 2),
                "cpu_utilisation": round(self._peak("cpu_utilisation"), 2),
            },
            "energy_efficiency": round(self.energy_efficiency, 4),
        }
