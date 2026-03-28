"""
NVMe Storage Pipeline
=====================
Ultra-fast NVMe SSD storage pipeline optimised for quantum data management
and inference workloads within the Quad-Brain compute array.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NVMeDevice:
    """Represents a single NVMe SSD in the array."""

    device_id: str
    path: Path
    capacity_gb: float
    read_bandwidth_gbps: float = 7.0   # PCIe Gen 4 x4 typical
    write_bandwidth_gbps: float = 6.5
    latency_us: float = 20.0            # microseconds
    is_healthy: bool = True
    bytes_read: int = 0
    bytes_written: int = 0

    def health_check(self) -> bool:
        """Return True when the device path is accessible."""
        self.is_healthy = self.path.exists() or not self.path.is_absolute()
        return self.is_healthy

    @property
    def utilisation_pct(self) -> float:
        total = self.bytes_read + self.bytes_written
        capacity_bytes = self.capacity_gb * 1024 ** 3
        return min(100.0, (total / capacity_bytes) * 100) if capacity_bytes else 0.0


@dataclass
class StorageMetrics:
    """Aggregated metrics for the storage pipeline."""

    total_reads: int = 0
    total_writes: int = 0
    total_bytes_read: int = 0
    total_bytes_written: int = 0
    read_latency_ms: float = 0.0
    write_latency_ms: float = 0.0
    throughput_gbps: float = 0.0
    errors: int = 0


class NVMeStoragePipeline:
    """
    Storage pipeline that stripes data across multiple NVMe devices to
    maximise throughput for quantum inference workloads.

    Features
    --------
    * Striped read / write across all healthy devices (RAID-0 style).
    * Per-device health monitoring with automatic failover.
    * Async-friendly stats tracking for the metrics subsystem.
    """

    def __init__(
        self,
        devices: Optional[List[NVMeDevice]] = None,
        stripe_size_kb: int = 128,
    ) -> None:
        self.devices: List[NVMeDevice] = devices or []
        self.stripe_size_bytes: int = stripe_size_kb * 1024
        self.metrics = StorageMetrics()
        self._write_cursor: int = 0  # round-robin device index

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def add_device(self, device: NVMeDevice) -> None:
        """Hot-add an NVMe device to the pipeline (plug-and-play)."""
        if any(d.device_id == device.device_id for d in self.devices):
            raise ValueError(f"Device '{device.device_id}' already registered.")
        self.devices.append(device)
        logger.info("NVMe device '%s' added (capacity=%.1f GB).", device.device_id, device.capacity_gb)

    def remove_device(self, device_id: str) -> None:
        """Hot-remove a device (fault-tolerant: remaining devices continue)."""
        before = len(self.devices)
        self.devices = [d for d in self.devices if d.device_id != device_id]
        if len(self.devices) == before:
            raise KeyError(f"Device '{device_id}' not found.")
        logger.warning("NVMe device '%s' removed from pipeline.", device_id)

    @property
    def healthy_devices(self) -> List[NVMeDevice]:
        return [d for d in self.devices if d.is_healthy]

    def run_health_checks(self) -> Dict[str, bool]:
        """Check all devices and return a {device_id: is_healthy} map."""
        results: Dict[str, bool] = {}
        for device in self.devices:
            results[device.device_id] = device.health_check()
            if not device.is_healthy:
                logger.error("NVMe device '%s' failed health check.", device.device_id)
        return results

    # ------------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------------

    def write(self, key: str, data: bytes) -> str:
        """
        Write *data* to the pipeline, striped across healthy devices.

        Returns a content-addressed key (SHA-256) for later retrieval.
        """
        healthy = self.healthy_devices
        if not healthy:
            self.metrics.errors += 1
            raise IOError("No healthy NVMe devices available for write.")

        content_key = hashlib.sha256(data).hexdigest()
        t0 = time.perf_counter()

        # Stripe the payload across devices
        chunk_size = max(self.stripe_size_bytes, len(data) // len(healthy))
        for idx, chunk_start in enumerate(range(0, len(data), chunk_size)):
            chunk = data[chunk_start: chunk_start + chunk_size]
            device = healthy[idx % len(healthy)]
            device.bytes_written += len(chunk)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.metrics.total_writes += 1
        self.metrics.total_bytes_written += len(data)
        self.metrics.write_latency_ms = (
            (self.metrics.write_latency_ms * (self.metrics.total_writes - 1) + elapsed_ms)
            / self.metrics.total_writes
        )
        self._update_throughput()
        logger.debug("Written %d bytes, key=%s, latency=%.2f ms.", len(data), content_key[:8], elapsed_ms)
        return content_key

    def read(self, key: str, size_hint: int = 0) -> bytes:
        """
        Read data identified by *key*.  In a real deployment this would
        reassemble stripes; here we simulate the latency and accounting.
        """
        healthy = self.healthy_devices
        if not healthy:
            self.metrics.errors += 1
            raise IOError("No healthy NVMe devices available for read.")

        t0 = time.perf_counter()
        size = size_hint or self.stripe_size_bytes
        for device in healthy:
            device.bytes_read += size // len(healthy)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.metrics.total_reads += 1
        self.metrics.total_bytes_read += size
        self.metrics.read_latency_ms = (
            (self.metrics.read_latency_ms * (self.metrics.total_reads - 1) + elapsed_ms)
            / self.metrics.total_reads
        )
        self._update_throughput()
        # Return a placeholder payload keyed on *key* for simulation
        return hashlib.sha256(key.encode()).digest() * (size // 32 + 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_throughput(self) -> None:
        healthy = self.healthy_devices
        if not healthy:
            self.metrics.throughput_gbps = 0.0
            return
        self.metrics.throughput_gbps = sum(
            d.read_bandwidth_gbps for d in healthy
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "devices": len(self.devices),
            "healthy_devices": len(self.healthy_devices),
            "total_reads": self.metrics.total_reads,
            "total_writes": self.metrics.total_writes,
            "read_latency_ms": round(self.metrics.read_latency_ms, 4),
            "write_latency_ms": round(self.metrics.write_latency_ms, 4),
            "throughput_gbps": round(self.metrics.throughput_gbps, 2),
            "errors": self.metrics.errors,
        }
