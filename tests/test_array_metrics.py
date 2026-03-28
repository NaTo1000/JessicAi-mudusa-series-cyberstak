"""Tests for performance monitoring, visualization, and fault tolerance."""

import time
import pytest

from quantum_quad_brain.quantum_core.quad_brain import QuantumQuadBrain, BrainConfig
from quantum_quad_brain.metrics.performance_monitor import PerformanceMonitor, Sample
from quantum_quad_brain.metrics.visualizer import MetricsVisualizer, _sparkline, _hbar
from quantum_quad_brain.array.fault_tolerance import (
    FaultToleranceManager,
    FaultSeverity,
    FaultEvent,
)
from quantum_quad_brain.array.scalable_array import ScalableComputeArray
from quantum_quad_brain.nvme_dram.storage_pipeline import NVMeStoragePipeline, NVMeDevice
from quantum_quad_brain.cm4_cluster.cluster_manager import CM4ClusterManager, CM4Node
from pathlib import Path


def _mini_brain() -> QuantumQuadBrain:
    configs = [
        BrainConfig("alpha", n_qubits=4, shots=8, seed=1),
        BrainConfig("beta",  n_qubits=4, shots=8, seed=2),
        BrainConfig("gamma", n_qubits=4, shots=8, seed=3),
        BrainConfig("delta", n_qubits=4, shots=8, seed=4),
    ]
    return QuantumQuadBrain(brain_configs=configs, n_cm4_nodes=2, dram_capacity_gb=0.001)


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------

class TestPerformanceMonitor:
    def test_record_sample(self):
        mon = PerformanceMonitor()
        sample = mon.record(cpu_utilisation=50.0, ram_utilisation=30.0, power_watts=15.0)
        assert isinstance(sample, Sample)
        assert sample.cpu_utilisation == pytest.approx(50.0)

    def test_latest(self):
        mon = PerformanceMonitor()
        mon.record(cpu_utilisation=20.0)
        mon.record(cpu_utilisation=80.0)
        assert mon.latest.cpu_utilisation == pytest.approx(80.0)

    def test_avg_cpu(self):
        mon = PerformanceMonitor()
        mon.record(cpu_utilisation=0.0)
        mon.record(cpu_utilisation=100.0)
        assert mon.avg_cpu == pytest.approx(50.0)

    def test_peak_throughput(self):
        mon = PerformanceMonitor()
        mon.record(nvme_throughput_gbps=5.0)
        mon.record(nvme_throughput_gbps=10.0)
        assert mon.peak_throughput_gbps == pytest.approx(10.0)

    def test_window_size_capped(self):
        mon = PerformanceMonitor(window_size=5)
        for i in range(10):
            mon.record(cpu_utilisation=float(i))
        assert len(list(mon._history)) == 5

    def test_record_from_system(self):
        brain = _mini_brain()
        mon = PerformanceMonitor()
        sample = mon.record_from_system(brain)
        assert isinstance(sample, Sample)

    def test_snapshot_keys(self):
        mon = PerformanceMonitor()
        mon.record(cpu_utilisation=10.0)
        snap = mon.snapshot()
        for key in ("uptime_s", "current", "averages", "peaks", "energy_efficiency"):
            assert key in snap


# ---------------------------------------------------------------------------
# MetricsVisualizer
# ---------------------------------------------------------------------------

class TestMetricsVisualizer:
    def test_sparkline_empty(self):
        assert _sparkline([]) == ""

    def test_sparkline_returns_string(self):
        result = _sparkline([0.0, 25.0, 50.0, 75.0, 100.0])
        assert isinstance(result, str)
        assert len(result) == 5

    def test_hbar_full(self):
        bar = _hbar(10, 10, width=10)
        assert bar == "█" * 10

    def test_hbar_empty(self):
        bar = _hbar(0, 10, width=10)
        assert bar == "░" * 10

    def test_render_dashboard_returns_string(self):
        brain = _mini_brain()
        mon = PerformanceMonitor()
        mon.record_from_system(brain)
        viz = MetricsVisualizer()
        output = viz.render_dashboard(brain, mon)
        assert isinstance(output, str)
        assert "QUANTUM QUAD-BRAIN" in output

    def test_render_brain_comparison(self):
        brain = _mini_brain()
        consensus = brain.consensus_infer(b"test")
        viz = MetricsVisualizer()
        output = viz.render_brain_comparison(consensus)
        assert "QUAD-BRAIN CONSENSUS" in output

    def test_render_sparklines(self):
        mon = PerformanceMonitor()
        for _ in range(5):
            mon.record(cpu_utilisation=50.0)
        viz = MetricsVisualizer()
        output = viz.render_sparklines(mon)
        assert "METRIC TRENDS" in output


# ---------------------------------------------------------------------------
# FaultToleranceManager
# ---------------------------------------------------------------------------

class TestFaultToleranceManager:
    def test_no_faults_on_healthy_pipeline(self):
        pipeline = NVMeStoragePipeline(
            devices=[NVMeDevice(device_id="n0", path=Path("/tmp"), capacity_gb=100)]
        )
        ftm = FaultToleranceManager()
        faults = ftm.check_storage(pipeline)
        assert len(faults) == 0

    def test_handler_invoked_on_fault(self):
        invoked = []
        ftm = FaultToleranceManager(min_healthy_nvme=5)
        ftm.register_handler(FaultSeverity.CRITICAL, lambda f: invoked.append(f))
        pipeline = NVMeStoragePipeline()   # no devices → healthy_count = 0
        ftm.check_storage(pipeline)
        assert len(invoked) > 0

    def test_incident_summary_keys(self):
        ftm = FaultToleranceManager()
        summary = ftm.incident_summary()
        for key in ("total", "open", "critical_open", "by_severity"):
            assert key in summary

    def test_cluster_offline_fault(self):
        cluster = CM4ClusterManager(heartbeat_timeout_s=0.01)
        cluster.register_node(CM4Node(node_id="n0", ip_address="10.0.0.1"))
        time.sleep(0.05)
        ftm = FaultToleranceManager()
        faults = ftm.check_cluster(cluster)
        offline_faults = [f for f in faults if "OFFLINE" in f.message or "cm4:n0" == f.component]
        assert len(offline_faults) > 0


# ---------------------------------------------------------------------------
# ScalableComputeArray integration
# ---------------------------------------------------------------------------

def _make_array() -> ScalableComputeArray:
    return ScalableComputeArray(
        dram_capacity_gb=0.001,
        initial_cm4_nodes=2,
        brain_configs=[
            BrainConfig("alpha", n_qubits=4, shots=8, seed=1),
            BrainConfig("beta",  n_qubits=4, shots=8, seed=2),
            BrainConfig("gamma", n_qubits=4, shots=8, seed=3),
            BrainConfig("delta", n_qubits=4, shots=8, seed=4),
        ],
    )


class TestScalableComputeArray:
    def test_run_inference(self):
        array = _make_array()
        result = array.run_inference(b"input data")
        assert "predicted_class" in result

    def test_run_consensus_inference(self):
        array = _make_array()
        result = array.run_consensus_inference(b"vote")
        assert "consensus_class" in result

    def test_store_and_load(self):
        from quantum_quad_brain.nvme_dram.storage_pipeline import NVMeDevice
        from pathlib import Path
        array = _make_array()
        array.add_nvme_device("test_nvme", "/tmp", capacity_gb=100.0)
        key = array.store("mykey", b"content")
        data = array.load(key, size_hint=7)
        assert isinstance(data, bytes)

    def test_add_remove_nvme(self):
        array = _make_array()
        array.add_nvme_device("nvme_new", "/tmp/nvme_new", capacity_gb=1000.0)
        assert any(d.device_id == "nvme_new" for d in array.quad_brain.storage.devices)
        array.remove_nvme_device("nvme_new")
        assert not any(d.device_id == "nvme_new" for d in array.quad_brain.storage.devices)

    def test_add_remove_cm4(self):
        array = _make_array()
        array.add_cm4_node("extra_node", "10.0.0.200")
        assert any(n.node_id == "extra_node" for n in array.quad_brain.cluster.nodes)
        array.remove_cm4_node("extra_node")
        assert not any(n.node_id == "extra_node" for n in array.quad_brain.cluster.nodes)

    def test_health_check(self):
        array = _make_array()
        health = array.health_check()
        assert "healthy" in health

    def test_dashboard_string(self):
        array = _make_array()
        array.run_inference(b"data")
        dash = array.dashboard()
        assert isinstance(dash, str)
        assert len(dash) > 0

    def test_full_summary_keys(self):
        array = _make_array()
        summary = array.full_summary()
        for key in ("array", "system", "performance", "health"):
            assert key in summary
