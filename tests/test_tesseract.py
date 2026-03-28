"""Tests for the topological tesseract architecture."""

from __future__ import annotations

import pytest

from src.architecture.tesseract import (
    Tesseract,
    TesseractConfig,
    TesseractLayer,
    LayerState,
    build_trillion_layer_tesseract,
    TRILLION,
)


class TestTesseractConfig:
    def test_default_config_has_trillion_layers(self):
        cfg = TesseractConfig()
        assert cfg.logical_layer_count == TRILLION

    def test_custom_config(self):
        cfg = TesseractConfig(logical_layer_count=1_000, dimensions=3)
        assert cfg.logical_layer_count == 1_000
        assert cfg.dimensions == 3


class TestTesseractLayer:
    def test_activate_sets_state(self):
        layer = TesseractLayer(logical_index=0, coordinates=(0, 0, 0, 0))
        layer.activate()
        assert layer.state == LayerState.ACTIVE
        assert "activated_at" in layer.metadata

    def test_suspend_from_active(self):
        layer = TesseractLayer(logical_index=0, coordinates=(0,))
        layer.activate()
        layer.suspend()
        assert layer.state == LayerState.SUSPENDED

    def test_suspend_noop_when_not_active(self):
        layer = TesseractLayer(logical_index=0, coordinates=(0,))
        layer.suspend()
        assert layer.state == LayerState.INITIALISING

    def test_quarantine(self):
        layer = TesseractLayer(logical_index=0, coordinates=(0,))
        layer.activate()
        layer.quarantine()
        assert layer.state == LayerState.QUARANTINED
        assert "quarantined_at" in layer.metadata

    def test_fingerprint_deterministic(self):
        layer = TesseractLayer(logical_index=5, coordinates=(1, 2, 3, 4))
        fp1 = layer.fingerprint()
        fp2 = layer.fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 64   # SHA-256 hex

    def test_fingerprint_changes_with_index(self):
        a = TesseractLayer(logical_index=0, coordinates=(0,), layer_id="x")
        b = TesseractLayer(logical_index=1, coordinates=(0,), layer_id="x")
        assert a.fingerprint() != b.fingerprint()


class TestTesseract:
    def _make(self, **kwargs) -> Tesseract:
        cfg = TesseractConfig(
            logical_layer_count=1_024,
            max_active_layers=32,
            dimensions=4,
            max_nesting_depth=2,
            quantum_routing=True,
            **kwargs,
        )
        return Tesseract(config=cfg)

    def test_create_layer_returns_active_layer(self):
        t = self._make()
        layer = t.create_layer(logical_index=0)
        assert layer.state == LayerState.ACTIVE
        assert t.active_layer_count == 1

    def test_create_same_index_returns_same_layer(self):
        t = self._make()
        a = t.create_layer(logical_index=10)
        b = t.create_layer(logical_index=10)
        assert a is b

    def test_eviction_on_window_overflow(self):
        cfg = TesseractConfig(
            logical_layer_count=1_000, max_active_layers=4, dimensions=2
        )
        t = Tesseract(config=cfg)
        for i in range(6):
            t.create_layer(logical_index=i)
        assert t.active_layer_count <= 4

    def test_evicted_layer_checkpointed(self):
        cfg = TesseractConfig(
            logical_layer_count=1_000, max_active_layers=2, dimensions=2
        )
        t = Tesseract(config=cfg)
        t.create_layer(logical_index=0)
        t.create_layer(logical_index=1)
        t.create_layer(logical_index=2)   # triggers eviction of index 0
        assert 0 in t._checkpoints

    def test_bootstrap_creates_correct_count(self):
        t = self._make()
        layers = t.bootstrap(count=8)
        assert len(layers) == 8
        assert all(l.state == LayerState.ACTIVE for l in layers)

    def test_bootstrap_with_children(self):
        t = self._make()
        layers = t.bootstrap(count=4, spawn_children=True)
        assert all(l.child_tesseract is not None for l in layers)

    def test_routing_populated(self):
        t = self._make()
        t.create_layer(logical_index=5)
        # Routing should have been updated
        assert 5 in t._routing_table

    def test_route_returns_neighbour_indices(self):
        t = self._make()
        t.create_layer(logical_index=0)
        neighbours = t.route(0, {})
        assert isinstance(neighbours, list)

    def test_get_layer_returns_none_for_evicted(self):
        cfg = TesseractConfig(
            logical_layer_count=1_000, max_active_layers=1, dimensions=2
        )
        t = Tesseract(config=cfg)
        t.create_layer(logical_index=0)
        t.create_layer(logical_index=1)   # evicts 0
        assert t.get_layer(0) is None
        assert t.get_layer(1) is not None

    def test_status_keys(self):
        t = self._make()
        t.bootstrap(count=4)
        s = t.status()
        for key in ("tesseract_id", "active_layers", "logical_layers", "dimensions"):
            assert key in s

    def test_nesting_depth_in_child(self):
        t = self._make()
        layers = t.bootstrap(count=1, spawn_children=True)
        child = layers[0].child_tesseract
        assert child is not None
        assert child.nesting_depth == 1
        assert child.parent_id == t.tesseract_id

    def test_iter_active_layers_sorted(self):
        t = self._make()
        for i in [3, 1, 2]:
            t.create_layer(logical_index=i)
        indices = [l.logical_index for l in t.iter_active_layers()]
        assert indices == sorted(indices)

    def test_build_trillion_layer_tesseract_factory(self):
        t = build_trillion_layer_tesseract(max_active_layers=8)
        assert t.config.logical_layer_count == 1_000_000_000_000
        assert t.config.max_active_layers == 8

    def test_coord_for_varies_across_indices(self):
        t = self._make()
        coords = {t._coord_for(i) for i in range(20)}
        assert len(coords) > 1
