"""Tests for the superconductor layer simulation."""

from __future__ import annotations

import pytest

from src.architecture.superconductor import (
    SuperconductorLayer,
    SuperconductorArray,
    ArrayConfig,
    ConductionState,
    build_billion_layer_array,
    BILLION,
    _CRITICAL_TEMP_K,
)


class TestSuperconductorLayer:
    def test_initial_state_superconducting(self):
        layer = SuperconductorLayer(logical_index=0)
        assert layer.state == ConductionState.SUPERCONDUCTING

    def test_tick_returns_positive_ops(self):
        layer = SuperconductorLayer(logical_index=0)
        ops = layer.tick(workload=1.0)
        assert ops > 0

    def test_tick_updates_metrics(self):
        layer = SuperconductorLayer(logical_index=0)
        layer.tick(workload=0.5)
        assert layer.metrics.total_ops > 0
        assert layer.metrics.uptime_seconds >= 0

    def test_cool_down_reduces_temperature(self):
        layer = SuperconductorLayer(logical_index=0)
        original = layer.temperature_k
        layer.cool_down(delta_k=20.0)
        assert layer.temperature_k < original

    def test_cool_down_cannot_go_below_floor(self):
        layer = SuperconductorLayer(logical_index=0)
        layer.cool_down(delta_k=1_000.0)
        assert layer.temperature_k >= _CRITICAL_TEMP_K * 0.7

    def test_state_transitions_to_normal_at_high_temp(self):
        layer = SuperconductorLayer(logical_index=0)
        layer.temperature_k = _CRITICAL_TEMP_K * 1.1
        layer._update_state()
        assert layer.state == ConductionState.NORMAL

    def test_state_transitions_to_transitioning(self):
        layer = SuperconductorLayer(logical_index=0)
        layer.temperature_k = _CRITICAL_TEMP_K * 1.0
        layer._update_state()
        assert layer.state == ConductionState.TRANSITIONING

    def test_status_keys(self):
        layer = SuperconductorLayer(logical_index=42)
        s = layer.status()
        for key in ("layer_id", "logical_index", "state", "temperature_k", "throughput"):
            assert key in s

    def test_peak_throughput_tracked(self):
        layer = SuperconductorLayer(logical_index=0)
        layer.tick(workload=1.0)
        layer.tick(workload=0.1)
        assert layer.metrics.peak_throughput >= layer.throughput


class TestSuperconductorArray:
    def _make(self, **kwargs) -> SuperconductorArray:
        cfg = ArrayConfig(
            logical_layer_count=BILLION,
            max_active_layers=16,
            default_workload=0.8,
            **kwargs,
        )
        return SuperconductorArray(config=cfg)

    def test_allocate_layer_creates_layer(self):
        arr = self._make()
        layer = arr.allocate_layer(logical_index=0)
        assert isinstance(layer, SuperconductorLayer)
        assert arr.active_layer_count == 1

    def test_allocate_same_index_idempotent(self):
        arr = self._make()
        a = arr.allocate_layer(0)
        b = arr.allocate_layer(0)
        assert a is b

    def test_eviction_on_overflow(self):
        cfg = ArrayConfig(logical_layer_count=BILLION, max_active_layers=3)
        arr = SuperconductorArray(config=cfg)
        for i in range(5):
            arr.allocate_layer(i)
        assert arr.active_layer_count <= 3

    def test_evicted_layer_logged(self):
        cfg = ArrayConfig(logical_layer_count=BILLION, max_active_layers=2)
        arr = SuperconductorArray(config=cfg)
        arr.allocate_layer(0)
        arr.allocate_layer(1)
        arr.allocate_layer(2)   # evicts 0
        assert 0 in arr._eviction_log

    def test_run_tick_returns_total_ops(self):
        arr = self._make()
        arr.bootstrap(count=4)
        ops = arr.run_tick()
        assert ops > 0

    def test_cool_all(self):
        arr = self._make()
        arr.bootstrap(count=4)
        before = arr.mean_temperature_k
        # Heat up layers manually
        for layer in arr._active.values():
            layer.temperature_k += 50
        arr.cool_all(delta_k=20.0)
        after = arr.mean_temperature_k
        assert after < before + 50    # cooling happened

    def test_bootstrap_returns_correct_count(self):
        arr = self._make()
        layers = arr.bootstrap(count=8)
        assert len(layers) == 8

    def test_superconducting_ratio_all_cold(self):
        arr = self._make()
        arr.bootstrap(count=4)
        # All layers start cold → ratio ≈ 1.0
        assert arr.superconducting_ratio == 1.0

    def test_aggregate_throughput_after_tick(self):
        arr = self._make()
        arr.bootstrap(count=4)
        arr.run_tick(workload=1.0)
        assert arr.aggregate_throughput > 0

    def test_status_keys(self):
        arr = self._make()
        s = arr.status()
        for key in ("array_id", "logical_layers", "active_layers",
                    "aggregate_throughput", "superconducting_ratio"):
            assert key in s

    def test_build_billion_layer_factory(self):
        arr = build_billion_layer_array(max_active_layers=8)
        assert arr.config.logical_layer_count == BILLION
