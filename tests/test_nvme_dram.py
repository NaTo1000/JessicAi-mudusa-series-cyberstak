"""Tests for NVMe storage pipeline and DRAM cache manager."""

import pytest
from pathlib import Path

from quantum_quad_brain.nvme_dram.storage_pipeline import NVMeStoragePipeline, NVMeDevice
from quantum_quad_brain.nvme_dram.cache_manager import DRAMCacheManager


# ---------------------------------------------------------------------------
# NVMe Storage Pipeline
# ---------------------------------------------------------------------------

def _make_pipeline(n: int = 2) -> NVMeStoragePipeline:
    devices = [
        NVMeDevice(
            device_id=f"nvme{i}",
            path=Path(f"/tmp/nvme{i}"),
            capacity_gb=500.0,
        )
        for i in range(n)
    ]
    return NVMeStoragePipeline(devices=devices)


class TestNVMeStoragePipeline:
    def test_write_returns_hex_key(self):
        pipeline = _make_pipeline()
        key = pipeline.write("test", b"hello world")
        assert len(key) == 64  # SHA-256 hex

    def test_read_returns_bytes(self):
        pipeline = _make_pipeline()
        data = pipeline.read("abc123", size_hint=256)
        assert isinstance(data, bytes)

    def test_metrics_accumulate(self):
        pipeline = _make_pipeline()
        pipeline.write("k1", b"data1")
        pipeline.write("k2", b"data2")
        assert pipeline.metrics.total_writes == 2
        pipeline.read("k1", size_hint=64)
        assert pipeline.metrics.total_reads == 1

    def test_add_remove_device(self):
        pipeline = _make_pipeline(n=1)
        assert len(pipeline.devices) == 1
        new_dev = NVMeDevice(device_id="nvme99", path=Path("/tmp/nvme99"), capacity_gb=1000.0)
        pipeline.add_device(new_dev)
        assert len(pipeline.devices) == 2
        pipeline.remove_device("nvme99")
        assert len(pipeline.devices) == 1

    def test_duplicate_device_raises(self):
        pipeline = _make_pipeline(n=1)
        with pytest.raises(ValueError, match="already registered"):
            pipeline.add_device(NVMeDevice(device_id="nvme0", path=Path("/tmp/nvme0"), capacity_gb=1.0))

    def test_remove_nonexistent_raises(self):
        pipeline = _make_pipeline()
        with pytest.raises(KeyError):
            pipeline.remove_device("does_not_exist")

    def test_write_no_devices_raises(self):
        pipeline = NVMeStoragePipeline()
        with pytest.raises(IOError, match="No healthy"):
            pipeline.write("k", b"data")

    def test_health_checks_return_map(self):
        pipeline = _make_pipeline()
        result = pipeline.run_health_checks()
        assert set(result.keys()) == {"nvme0", "nvme1"}

    def test_summary_keys(self):
        pipeline = _make_pipeline()
        summary = pipeline.summary()
        for key in ("devices", "healthy_devices", "throughput_gbps", "errors"):
            assert key in summary


# ---------------------------------------------------------------------------
# DRAM Cache Manager
# ---------------------------------------------------------------------------

class TestDRAMCacheManager:
    def test_put_and_get(self):
        cache = DRAMCacheManager(capacity_gb=0.001)
        cache.put("key1", b"value1")
        assert cache.get("key1") == b"value1"

    def test_miss_returns_none(self):
        cache = DRAMCacheManager()
        assert cache.get("nonexistent") is None

    def test_hit_ratio(self):
        cache = DRAMCacheManager()
        cache.put("k", b"v")
        cache.get("k")          # hit
        cache.get("missing")    # miss
        assert cache.hit_ratio == pytest.approx(0.5)

    def test_lru_eviction(self):
        # Tiny cache to force eviction
        cache = DRAMCacheManager(capacity_gb=1e-9, eviction_policy="lru")
        cache.put("a", b"x" * 10)
        cache.put("b", b"y" * 10)
        # "a" should be evicted
        assert cache.get("a") is None

    def test_lfu_policy_accepted(self):
        cache = DRAMCacheManager(eviction_policy="lfu")
        cache.put("k", b"data")
        assert cache.get("k") == b"data"

    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown eviction policy"):
            DRAMCacheManager(eviction_policy="fifo")

    def test_invalidate(self):
        cache = DRAMCacheManager()
        cache.put("k", b"v")
        assert cache.invalidate("k") is True
        assert cache.get("k") is None

    def test_flush(self):
        cache = DRAMCacheManager()
        cache.put("a", b"1")
        cache.put("b", b"2")
        cache.flush()
        assert cache.get("a") is None
        assert cache.utilisation_pct == 0.0

    def test_summary_keys(self):
        cache = DRAMCacheManager()
        summary = cache.summary()
        for key in ("capacity_gb", "used_gb", "entries", "hit_ratio"):
            assert key in summary
