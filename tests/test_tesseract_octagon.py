"""Tests for TesseractOctagon."""

import pytest
from nonganon_tesseract.tesseract_octagon import (
    OCTAGON_LAYERS,
    OctagonLayer,
    TesseractOctagon,
    _TESSERACT_TOPOLOGY,
)


class TestOctagonLayer:
    def test_init_valid(self):
        layer = OctagonLayer(0, "test")
        assert layer.index == 0
        assert layer.label == "test"
        assert layer.state == {}

    def test_default_label(self):
        layer = OctagonLayer(3)
        assert layer.label == "layer-3"

    def test_init_invalid_index(self):
        with pytest.raises(ValueError):
            OctagonLayer(8)
        with pytest.raises(ValueError):
            OctagonLayer(-1)

    def test_adjacent_indices_match_topology(self):
        for idx in range(OCTAGON_LAYERS):
            layer = OctagonLayer(idx)
            assert sorted(layer.adjacent_indices) == sorted(_TESSERACT_TOPOLOGY[idx])

    def test_update_state(self):
        layer = OctagonLayer(0)
        layer.update_state("x", 42)
        assert layer.state["x"] == 42

    def test_fingerprint_changes_with_state(self):
        layer = OctagonLayer(0)
        fp1 = layer.fingerprint()
        layer.update_state("key", "value")
        fp2 = layer.fingerprint()
        assert fp1 != fp2

    def test_fingerprint_is_hex_string(self):
        layer = OctagonLayer(0)
        fp = layer.fingerprint()
        assert len(fp) == 64
        int(fp, 16)  # must be valid hex


class TestTesseractOctagon:
    def test_init_default_labels(self):
        to = TesseractOctagon()
        assert len(to.layers) == OCTAGON_LAYERS

    def test_init_custom_labels(self):
        labels = [f"L{i}" for i in range(OCTAGON_LAYERS)]
        to = TesseractOctagon(labels=labels)
        for i, lyr in enumerate(to.layers):
            assert lyr.label == f"L{i}"

    def test_init_wrong_label_count(self):
        with pytest.raises(ValueError):
            TesseractOctagon(labels=["a", "b"])

    def test_layer_returns_correct_layer(self):
        to = TesseractOctagon()
        for i in range(OCTAGON_LAYERS):
            assert to.layer(i).index == i

    def test_layer_index_out_of_range(self):
        to = TesseractOctagon()
        with pytest.raises(IndexError):
            to.layer(8)

    def test_push_and_read(self):
        to = TesseractOctagon()
        to.push(layer_index=0, key="signal", value=99)
        assert to.layer(0).state["signal"] == 99

    def test_route_propagates_to_adjacent(self):
        to = TesseractOctagon()
        to.push(layer_index=0, key="ping", value="hello")
        results = to.route(layer_index=0, key="ping")
        adjacent = _TESSERACT_TOPOLOGY[0]
        assert set(results.keys()) == set(adjacent)
        for idx in adjacent:
            assert to.layer(idx).state["ping"] == "hello"

    def test_broadcast_sets_all_layers(self):
        to = TesseractOctagon()
        to.broadcast(key="global", value=True)
        for lyr in to.layers:
            assert lyr.state["global"] is True

    def test_collect_states(self):
        to = TesseractOctagon()
        to.push(layer_index=2, key="data", value=[1, 2, 3])
        states = to.collect_states()
        assert states[2]["data"] == [1, 2, 3]

    def test_topology_summary_contains_all_layers(self):
        to = TesseractOctagon()
        summary = to.topology_summary()
        for i in range(OCTAGON_LAYERS):
            assert f"[{i}]" in summary

    def test_all_pairs_count(self):
        to = TesseractOctagon()
        pairs = to.all_pairs()
        # C(8,2) = 28
        assert len(pairs) == 28
