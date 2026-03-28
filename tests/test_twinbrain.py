"""Tests for TwinbrainAlgorithm."""

import math
import pytest
from nonganon_tesseract.twinbrain import (
    Brain,
    TwinbrainAlgorithm,
    _layer_superposition,
    _layer_entanglement,
    _layer_interference,
    _layer_measurement,
)


class TestLayerFunctions:
    def test_superposition_sums_to_original_weighted(self):
        v = [4.0, 8.0, 12.0]
        result = _layer_superposition(v)
        # Each element should be divided by n=3
        assert abs(result[0] - 4.0 / 3) < 1e-9

    def test_superposition_empty(self):
        assert _layer_superposition([]) == []

    def test_entanglement_length_preserved(self):
        v = [1.0, 2.0, 3.0, 4.0]
        assert len(_layer_entanglement(v)) == 4

    def test_interference_length_preserved(self):
        v = [1.0, 2.0, 3.0]
        result = _layer_interference(v)
        assert len(result) == 3

    def test_interference_modulated_by_sine(self):
        v = [1.0, 1.0, 1.0]
        result = _layer_interference(v)
        # middle element should have full sine amplitude (sin(pi/2)=1 for n=3, i=1)
        expected_mid = math.sin(math.pi * 2 / 4)
        assert abs(result[1] - expected_mid) < 1e-9

    def test_measurement_one_hot(self):
        v = [0.1, 0.9, 0.3]
        result = _layer_measurement(v)
        assert result == [0.0, 1.0, 0.0]

    def test_measurement_all_zero_input(self):
        v = [0.0, 0.0, 0.0]
        result = _layer_measurement(v)
        assert result == [0.0, 0.0, 0.0]

    def test_measurement_empty(self):
        assert _layer_measurement([]) == []


class TestBrain:
    def test_process_returns_correct_length(self):
        b = Brain("A")
        out = b.process([0.1, 0.5, 0.9, 0.3])
        assert len(out) == 4

    def test_process_empty_raises(self):
        b = Brain("A")
        with pytest.raises(ValueError):
            b.process([])

    def test_initial_activation_none(self):
        b = Brain("X")
        assert b.activation is None

    def test_is_alive_after_nonzero_input(self):
        b = Brain("A")
        b.process([0.1, 0.5, 0.9])
        assert b.is_alive()

    def test_fingerprint_is_hex(self):
        b = Brain("A")
        b.process([1.0, 0.0, 0.0])
        fp = b.fingerprint()
        assert len(fp) == 64
        int(fp, 16)

    def test_layer_outputs_stored(self):
        b = Brain("A")
        b.process([0.2, 0.4, 0.6, 0.8])
        assert len(b._layer_outputs) == 4


class TestTwinbrainAlgorithm:
    def test_decide_returns_correct_length(self):
        tb = TwinbrainAlgorithm()
        result = tb.decide([0.1, 0.5, 0.9, 0.3])
        assert len(result) == 4

    def test_decide_normalised(self):
        tb = TwinbrainAlgorithm()
        result = tb.decide([0.1, 0.5, 0.9, 0.3])
        assert abs(sum(result) - 1.0) < 1e-9

    def test_diagnostics_has_both_brains(self):
        tb = TwinbrainAlgorithm()
        tb.decide([1.0, 0.0, 0.0])
        diag = tb.diagnostics()
        assert "brain_a" in diag
        assert "brain_b" in diag

    def test_custom_brain_labels(self):
        tb = TwinbrainAlgorithm(brain_a_label="primary", brain_b_label="secondary")
        assert tb.brain_a.brain_id == "primary"
        assert tb.brain_b.brain_id == "secondary"

    def test_normalise_zero_vector(self):
        result = TwinbrainAlgorithm._normalise([0.0, 0.0, 0.0])
        # Should return uniform 1/3 each
        assert abs(result[0] - 1 / 3) < 1e-9

    def test_normalise_already_normalised(self):
        v = [0.25, 0.25, 0.25, 0.25]
        result = TwinbrainAlgorithm._normalise(v)
        assert abs(sum(result) - 1.0) < 1e-9
