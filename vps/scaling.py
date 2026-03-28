"""VPS Scaler — manage on-demand compute resource allocation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResourceAllocation:
    """Desired resource allocation for a workload."""

    cpu_cores: int = 2
    ram_gb: float = 4.0
    gpu_count: int = 0
    gpu_vram_gb: float = 0.0
    storage_gb: float = 20.0
    region: str = "local"           # "local" or a cloud region slug
    provider: str = "local"         # "local" | "aws" | "gcp" | "azure" | "hetzner"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "gpu_count": self.gpu_count,
            "gpu_vram_gb": self.gpu_vram_gb,
            "storage_gb": self.storage_gb,
            "region": self.region,
            "provider": self.provider,
        }


@dataclass
class ScalingEvent:
    """Record of a scaling operation."""

    timestamp: float
    action: str          # "scale_up" | "scale_down" | "migrate"
    previous: ResourceAllocation
    current: ResourceAllocation
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class VPSScaler:
    """
    Manages on-demand VPS resource scaling and workload migration.

    Usage::

        scaler = VPSScaler()
        scaler.scale_up(cpu_cores=8, ram_gb=32, gpu_count=1)
        print(scaler.current_allocation)
    """

    # Reasonable upper bounds for safety
    _MAX_CPU = 128
    _MAX_RAM_GB = 1024.0
    _MAX_GPU = 8

    def __init__(
        self,
        initial: ResourceAllocation | None = None,
    ) -> None:
        self._allocation = initial or ResourceAllocation()
        self._history: list[ScalingEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_allocation(self) -> ResourceAllocation:
        return self._allocation

    def scale_up(
        self,
        cpu_cores: int | None = None,
        ram_gb: float | None = None,
        gpu_count: int | None = None,
        gpu_vram_gb: float | None = None,
        storage_gb: float | None = None,
        reason: str = "manual scale-up",
    ) -> ResourceAllocation:
        """Increase resource allocation. Returns the new allocation."""
        previous = ResourceAllocation(**self._allocation.to_dict())
        if cpu_cores is not None:
            self._allocation.cpu_cores = min(cpu_cores, self._MAX_CPU)
        if ram_gb is not None:
            self._allocation.ram_gb = min(ram_gb, self._MAX_RAM_GB)
        if gpu_count is not None:
            self._allocation.gpu_count = min(gpu_count, self._MAX_GPU)
        if gpu_vram_gb is not None:
            self._allocation.gpu_vram_gb = gpu_vram_gb
        if storage_gb is not None:
            self._allocation.storage_gb = storage_gb

        self._record_event("scale_up", previous, self._allocation, reason)
        return ResourceAllocation(**self._allocation.to_dict())

    def scale_down(
        self,
        cpu_cores: int | None = None,
        ram_gb: float | None = None,
        gpu_count: int | None = None,
        reason: str = "manual scale-down",
    ) -> ResourceAllocation:
        """Decrease resource allocation. Values are floored at minimums."""
        previous = ResourceAllocation(**self._allocation.to_dict())
        if cpu_cores is not None:
            self._allocation.cpu_cores = max(1, cpu_cores)
        if ram_gb is not None:
            self._allocation.ram_gb = max(0.5, ram_gb)
        if gpu_count is not None:
            self._allocation.gpu_count = max(0, gpu_count)

        self._record_event("scale_down", previous, self._allocation, reason)
        return ResourceAllocation(**self._allocation.to_dict())

    def migrate(
        self,
        provider: str,
        region: str,
        reason: str = "workload migration",
    ) -> ResourceAllocation:
        """Migrate workload to a different provider / region."""
        previous = ResourceAllocation(**self._allocation.to_dict())
        self._allocation.provider = provider
        self._allocation.region = region
        self._record_event("migrate", previous, self._allocation, reason)
        return ResourceAllocation(**self._allocation.to_dict())

    def auto_scale(self, cpu_percent: float, ram_percent: float) -> ResourceAllocation | None:
        """
        Automatically scale up if resource utilisation exceeds 80%.

        Parameters
        ----------
        cpu_percent:
            Current CPU utilisation (0–100).
        ram_percent:
            Current RAM utilisation (0–100).

        Returns
        -------
        New :class:`ResourceAllocation` if a scale-up was triggered, else None.
        """
        if cpu_percent > 80:
            new_cpu = min(self._allocation.cpu_cores * 2, self._MAX_CPU)
            return self.scale_up(
                cpu_cores=new_cpu,
                reason=f"auto-scale: CPU at {cpu_percent:.1f}%",
            )
        if ram_percent > 80:
            new_ram = min(self._allocation.ram_gb * 2, self._MAX_RAM_GB)
            return self.scale_up(
                ram_gb=new_ram,
                reason=f"auto-scale: RAM at {ram_percent:.1f}%",
            )
        return None

    def local_resource_stats(self) -> dict[str, Any]:
        """Return basic resource stats from the local machine."""
        stats: dict[str, Any] = {}
        try:
            import psutil  # type: ignore

            stats["cpu_percent"] = psutil.cpu_percent(interval=0.2)
            mem = psutil.virtual_memory()
            stats["ram_total_gb"] = round(mem.total / 1024 ** 3, 2)
            stats["ram_used_gb"] = round(mem.used / 1024 ** 3, 2)
            stats["ram_percent"] = mem.percent
        except ImportError:
            stats["error"] = "psutil not installed — install it for live stats"
        return stats

    def history(self) -> list[ScalingEvent]:
        return list(self._history)

    def reset(self, reason: str = "reset to defaults") -> None:
        """Reset to default allocation and record the event."""
        previous = ResourceAllocation(**self._allocation.to_dict())
        self._allocation = ResourceAllocation()
        self._record_event("reset", previous, self._allocation, reason)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_event(
        self,
        action: str,
        previous: ResourceAllocation,
        current: ResourceAllocation,
        reason: str,
    ) -> None:
        self._history.append(
            ScalingEvent(
                timestamp=time.time(),
                action=action,
                previous=previous,
                current=current,
                reason=reason,
            )
        )
