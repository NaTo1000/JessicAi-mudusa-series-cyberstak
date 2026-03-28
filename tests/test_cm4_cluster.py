"""Tests for CM4 cluster manager and workload distributor."""

import time
import pytest

from quantum_quad_brain.cm4_cluster.cluster_manager import CM4ClusterManager, CM4Node, NodeState
from quantum_quad_brain.cm4_cluster.workload_distributor import (
    WorkloadDistributor,
    SchedulingStrategy,
    Task,
)


# ---------------------------------------------------------------------------
# CM4 Cluster Manager
# ---------------------------------------------------------------------------

def _make_cluster(n: int = 3) -> CM4ClusterManager:
    mgr = CM4ClusterManager()
    for i in range(n):
        mgr.register_node(CM4Node(node_id=f"node{i}", ip_address=f"10.0.0.{i}"))
    return mgr


class TestCM4ClusterManager:
    def test_register_and_count(self):
        mgr = _make_cluster(3)
        assert len(mgr.nodes) == 3

    def test_duplicate_registration_raises(self):
        mgr = _make_cluster(1)
        with pytest.raises(ValueError, match="already registered"):
            mgr.register_node(CM4Node(node_id="node0", ip_address="10.0.0.99"))

    def test_deregister(self):
        mgr = _make_cluster(2)
        mgr.deregister_node("node0")
        assert len(mgr.nodes) == 1

    def test_deregister_unknown_raises(self):
        mgr = _make_cluster(1)
        with pytest.raises(KeyError):
            mgr.deregister_node("ghost")

    def test_heartbeat_updates_state(self):
        mgr = _make_cluster(1)
        mgr.receive_heartbeat("node0", cpu=50.0, ram=30.0)
        node = mgr.get_node("node0")
        assert node.cpu_utilisation == pytest.approx(50.0)
        assert node.ram_utilisation == pytest.approx(30.0)

    def test_heartbeat_unknown_raises(self):
        mgr = CM4ClusterManager()
        with pytest.raises(KeyError):
            mgr.receive_heartbeat("unknown")

    def test_offline_detection(self):
        mgr = CM4ClusterManager(heartbeat_timeout_s=0.01)
        mgr.register_node(CM4Node(node_id="stale", ip_address="10.0.0.1"))
        time.sleep(0.05)
        states = mgr.check_node_health()
        assert states["stale"] == "OFFLINE"

    def test_recovery_after_heartbeat(self):
        mgr = CM4ClusterManager(heartbeat_timeout_s=0.01)
        mgr.register_node(CM4Node(node_id="n0", ip_address="10.0.0.1"))
        time.sleep(0.05)
        mgr.check_node_health()
        mgr.receive_heartbeat("n0")
        assert mgr.get_node("n0").state != NodeState.OFFLINE

    def test_cluster_summary_keys(self):
        mgr = _make_cluster(2)
        s = mgr.cluster_summary()
        for key in ("total_nodes", "online_nodes", "available_nodes"):
            assert key in s


# ---------------------------------------------------------------------------
# Workload Distributor
# ---------------------------------------------------------------------------

class TestWorkloadDistributor:
    def _make_distributor(self, strategy=SchedulingStrategy.ROUND_ROBIN):
        cluster = _make_cluster(3)
        return WorkloadDistributor(cluster=cluster, strategy=strategy)

    def test_submit_completes(self):
        dist = self._make_distributor()
        task = dist.submit(payload=b"data")
        assert task.completed is True

    def test_submit_with_executor(self):
        dist = self._make_distributor()
        task = dist.submit(payload=42, executor=lambda node, p: p * 2)
        assert task.result == 84

    def test_submit_batch(self):
        dist = self._make_distributor()
        tasks = dist.submit_batch([b"a", b"b", b"c"])
        assert len(tasks) == 3
        assert all(t.completed for t in tasks)

    def test_completed_count_accumulates(self):
        dist = self._make_distributor()
        for _ in range(5):
            dist.submit(b"x")
        assert dist.completed_count == 5

    def test_least_loaded_strategy(self):
        cluster = _make_cluster(2)
        cluster.get_node("node0").cpu_utilisation = 80.0
        cluster.get_node("node1").cpu_utilisation = 10.0
        dist = WorkloadDistributor(cluster=cluster, strategy=SchedulingStrategy.LEAST_LOADED)
        task = dist.submit(b"payload")
        assert task.assigned_to == "node1"

    def test_no_nodes_returns_failed_task(self):
        dist = WorkloadDistributor(cluster=CM4ClusterManager())
        task = dist.submit(b"x")
        assert task.completed is False
        assert task.error is not None

    def test_summary_keys(self):
        dist = self._make_distributor()
        dist.submit(b"y")
        s = dist.distributor_summary()
        for key in ("strategy", "completed", "failed", "success_rate"):
            assert key in s
