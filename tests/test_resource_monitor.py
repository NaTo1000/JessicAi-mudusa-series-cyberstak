"""
Tests for the resource monitor and 5% cap enforcement.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from hive.resource_monitor import (
    MAX_CPU_PERCENT,
    MAX_RAM_PERCENT,
    ResourceMonitor,
    ResourceSnapshot,
)


class TestResourceSnapshot:
    def test_headroom_within_cap(self):
        snap = ResourceSnapshot(cpu_percent=2.0, ram_percent=3.0)
        assert snap.headroom_cpu == pytest.approx(3.0)
        assert snap.headroom_ram == pytest.approx(2.0)
        assert snap.is_within_cap

    def test_headroom_at_exact_cap(self):
        snap = ResourceSnapshot(cpu_percent=MAX_CPU_PERCENT, ram_percent=MAX_RAM_PERCENT)
        assert snap.headroom_cpu == pytest.approx(0.0)
        assert snap.headroom_ram == pytest.approx(0.0)
        assert snap.is_within_cap

    def test_headroom_exceeds_cap(self):
        snap = ResourceSnapshot(cpu_percent=6.0, ram_percent=0.0)
        assert snap.headroom_cpu == pytest.approx(0.0)
        assert not snap.is_within_cap

    def test_headroom_never_negative(self):
        snap = ResourceSnapshot(cpu_percent=10.0, ram_percent=10.0)
        assert snap.headroom_cpu >= 0.0
        assert snap.headroom_ram >= 0.0

    def test_to_dict_keys(self):
        snap = ResourceSnapshot(cpu_percent=1.5, ram_percent=2.5, ram_available_mb=1024.0)
        d = snap.to_dict()
        assert "cpu_percent" in d
        assert "ram_percent" in d
        assert "headroom_cpu" in d
        assert "headroom_ram" in d
        assert "is_within_cap" not in d  # property, not serialised directly


class TestResourceMonitor:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        monitor = ResourceMonitor(poll_interval=0.05)
        await monitor.start()
        await asyncio.sleep(0.15)
        snap = monitor.snapshot
        assert isinstance(snap.cpu_percent, float)
        assert isinstance(snap.ram_percent, float)
        assert snap.cpu_count >= 1
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        results = []
        monitor = ResourceMonitor(poll_interval=0.05)
        monitor.register_callback(results.append)
        await monitor.start()
        await asyncio.sleep(0.2)
        await monitor.stop()
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self):
        monitor = ResourceMonitor(poll_interval=0.05)
        await monitor.start()
        await monitor.start()  # second call should be no-op
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_snapshot_populated_after_poll(self):
        monitor = ResourceMonitor(poll_interval=0.05)
        await monitor.start()
        await asyncio.sleep(0.15)
        snap = monitor.snapshot
        assert snap.ram_available_mb > 0
        await monitor.stop()
