"""Tests for the cluster regulation framework."""

from __future__ import annotations

import pytest

from src.architecture.clustering import (
    ClusterNode,
    SubCluster,
    SubClusterConfig,
    ClusterRegulator,
    RegulatorConfig,
    NodeState,
)


class TestClusterNode:
    def test_initial_state_idle(self):
        node = ClusterNode()
        assert node.state == NodeState.IDLE

    def test_tick_with_load_sets_active(self):
        node = ClusterNode(current_load=0.5)
        node.tick(load_delta=0.0)
        assert node.state == NodeState.ACTIVE

    def test_tick_with_high_load_sets_overloaded(self):
        node = ClusterNode(current_load=0.9)
        node.tick(load_delta=0.05)
        assert node.state == NodeState.OVERLOADED

    def test_tick_load_clamps_to_unit(self):
        node = ClusterNode(current_load=0.99)
        node.tick(load_delta=100.0)
        assert node.current_load <= 1.0

    def test_tick_load_cannot_go_negative(self):
        node = ClusterNode(current_load=0.1)
        node.tick(load_delta=-1000.0)
        assert node.current_load >= 0.0

    def test_quarantine(self):
        node = ClusterNode(current_load=0.8)
        node.quarantine()
        assert node.state == NodeState.QUARANTINED
        assert node.current_load == 0.0

    def test_status_keys(self):
        node = ClusterNode()
        s = node.status()
        for key in ("node_id", "state", "cpu_budget", "current_load"):
            assert key in s


class TestSubCluster:
    def _make(self, **kwargs) -> SubCluster:
        cfg = SubClusterConfig(**kwargs)
        return SubCluster(config=cfg)

    def test_add_node_increases_count(self):
        sc = self._make()
        sc.add_node()
        assert sc.node_count == 1

    def test_add_explicit_node(self):
        sc = self._make()
        node = ClusterNode(node_id="test-node")
        sc.add_node(node)
        assert sc.node_count == 1

    def test_capacity_exceeded_raises(self):
        sc = self._make(max_nodes=2)
        sc.add_node()
        sc.add_node()
        with pytest.raises(ValueError):
            sc.add_node()

    def test_remove_node(self):
        sc = self._make()
        node = sc.add_node()
        removed = sc.remove_node(node.node_id)
        assert removed is node
        assert sc.node_count == 0

    def test_remove_missing_node_returns_none(self):
        sc = self._make()
        assert sc.remove_node("does-not-exist") is None

    def test_quarantine_node(self):
        sc = self._make()
        node = sc.add_node()
        sc.quarantine_node(node.node_id)
        assert sc._nodes[node.node_id].state == NodeState.QUARANTINED

    def test_aggregate_load_empty_cluster(self):
        sc = self._make()
        assert sc.aggregate_load == 0.0

    def test_aggregate_load_calculation(self):
        sc = self._make()
        n1 = ClusterNode(current_load=0.4)
        n2 = ClusterNode(current_load=0.6)
        sc.add_node(n1)
        sc.add_node(n2)
        assert abs(sc.aggregate_load - 0.5) < 1e-9

    def test_should_split_high_load(self):
        sc = self._make(high_threshold=0.5)
        n = ClusterNode(current_load=0.9)
        sc.add_node(n)
        sc.add_node(ClusterNode(current_load=0.9))
        assert sc.should_split()

    def test_should_merge_low_load(self):
        sc = self._make(low_threshold=0.5)
        n = ClusterNode(current_load=0.1)
        sc.add_node(n)
        assert sc.should_merge()

    def test_split_creates_sibling(self):
        sc = self._make()
        for _ in range(4):
            n = ClusterNode(current_load=0.9)
            sc.add_node(n)
        original_count = sc.node_count
        sibling = sc.split()
        assert sibling is not sc
        assert sc.node_count < original_count
        assert sibling.node_count > 0

    def test_absorb_merges_nodes(self):
        sc_a = self._make()
        sc_b = self._make()
        for _ in range(2):
            sc_a.add_node()
            sc_b.add_node()
        sc_a.absorb(sc_b)
        assert sc_a.node_count >= 2

    def test_tick_advances_nodes(self):
        sc = self._make()
        n = ClusterNode(current_load=0.5)
        sc.add_node(n)
        sc.tick()
        # node state should have been evaluated (stays ACTIVE at 0.5 load)
        assert n.state == NodeState.ACTIVE

    def test_status_keys(self):
        sc = self._make()
        s = sc.status()
        for key in ("cluster_id", "node_count", "aggregate_load"):
            assert key in s


class TestClusterRegulator:
    def _make(self, **kwargs) -> ClusterRegulator:
        # Default rebalance_interval_s=0.0 so tests always trigger rebalancing
        defaults = {"rebalance_interval_s": 0.0}
        defaults.update(kwargs)
        cfg = RegulatorConfig(**defaults)
        return ClusterRegulator(config=cfg)

    def test_create_sub_cluster(self):
        reg = self._make()
        sc = reg.create_sub_cluster(node_count=2)
        assert reg.sub_cluster_count == 1
        assert sc.node_count == 2

    def test_max_sub_clusters_raises(self):
        reg = self._make(max_sub_clusters=1)
        reg.create_sub_cluster(node_count=1)
        with pytest.raises(RuntimeError):
            reg.create_sub_cluster(node_count=1)

    def test_remove_sub_cluster(self):
        reg = self._make()
        sc = reg.create_sub_cluster()
        reg.remove_sub_cluster(sc.cluster_id)
        assert reg.sub_cluster_count == 0

    def test_rebalance_splits_overloaded(self):
        reg = self._make(max_sub_clusters=16)
        sc = reg.create_sub_cluster(
            node_count=4,
            sub_config=SubClusterConfig(high_threshold=0.5),
        )
        # Force high load
        for node in sc._nodes.values():
            node.current_load = 0.9
            node.state = NodeState.OVERLOADED
        result = reg.rebalance()
        assert result["splits"] >= 1

    def test_rebalance_merges_idle_clusters(self):
        reg = self._make(max_sub_clusters=16)
        for _ in range(4):
            reg.create_sub_cluster(
                node_count=1,
                sub_config=SubClusterConfig(low_threshold=0.5),
            )
        # All loads are 0 → should merge
        result = reg.rebalance()
        assert result["merges"] >= 1

    def test_total_nodes(self):
        reg = self._make()
        reg.create_sub_cluster(node_count=3)
        reg.create_sub_cluster(node_count=2)
        assert reg.total_nodes == 5

    def test_global_load_zero_initially(self):
        reg = self._make()
        reg.create_sub_cluster(node_count=2)
        assert reg.global_load == 0.0

    def test_status_keys(self):
        reg = self._make()
        s = reg.status()
        for key in ("regulator_id", "sub_clusters", "total_nodes", "global_load"):
            assert key in s
