"""Tests for the Neuron model."""

import math
import time
import pytest

from quantum_neural_brain.neuron import Neuron, NeuronConfig, NeuronState


class TestNeuronConfig:
    def test_defaults(self):
        cfg = NeuronConfig()
        assert cfg.resting_potential == -70.0
        assert cfg.threshold_potential == -55.0
        assert cfg.peak_potential == 40.0
        assert cfg.reset_potential == -75.0
        assert cfg.refractory_period_ms == 2.0


class TestNeuronBasics:
    def setup_method(self):
        self.neuron = Neuron(neuron_id="test_n", layer_name="hidden")  # use keyword args

    def test_initial_state(self):
        assert self.neuron.state == NeuronState.RESTING
        assert self.neuron.total_spikes == 0
        assert self.neuron.membrane_potential == pytest.approx(-70.0, abs=0.1)

    def test_receive_input_transitions_to_integrating(self):
        self.neuron.receive_input(10.0)
        assert self.neuron.state == NeuronState.INTEGRATING

    def test_step_advances_without_error(self):
        result = self.neuron.step(dt_ms=1.0)
        assert isinstance(result, bool)

    def test_force_fire_increments_spike_count(self):
        fired = self.neuron.force_fire()
        assert fired is True
        assert self.neuron.total_spikes == 1

    def test_refractory_prevents_immediate_refire(self):
        self.neuron.force_fire()
        # Should be refractory immediately after
        assert self.neuron.state == NeuronState.REFRACTORY
        # Another force_fire should be blocked
        fired = self.neuron.force_fire()
        assert fired is False
        assert self.neuron.total_spikes == 1  # Still just 1

    def test_reset_restores_resting_state(self):
        self.neuron.receive_input(50.0)
        self.neuron.step()
        self.neuron.reset()
        assert self.neuron.state == NeuronState.RESTING
        assert self.neuron.membrane_potential == pytest.approx(-70.0, abs=0.1)

    def test_membrane_potential_clamped(self):
        # Give it enormous input to push past peak
        for _ in range(100):
            self.neuron.receive_input(1000.0)
            self.neuron.step(dt_ms=1.0)
        assert self.neuron.membrane_potential <= self.neuron.config.peak_potential + 5.0

    def test_firing_rate_zero_initially(self):
        n = Neuron("fresh", layer_name="input")
        assert n.firing_rate_hz == 0.0

    def test_spike_times_recorded(self):
        self.neuron.force_fire()
        assert len(self.neuron.spike_times) == 1

    def test_is_firing_property(self):
        # After force_fire the neuron enters refractory (not firing anymore in the step)
        self.neuron.force_fire()
        # State is refractory after _fire() completes
        assert self.neuron.is_refractory is True

    def test_repr(self):
        r = repr(self.neuron)
        assert "test_n" in r
        assert "hidden" in r


class TestNeuronIntegration:
    """Test the leaky integrate-and-fire dynamics."""

    def test_large_input_causes_firing(self):
        """Sufficient sustained input should cause a spike."""
        n = Neuron("fire_test", layer_name="hidden")
        fired_any = False
        for _ in range(50):
            n.receive_input(100.0)
            if n.step(dt_ms=1.0):
                fired_any = True
                break
        assert fired_any, "Neuron should fire with large persistent input"

    def test_zero_input_stays_resting(self):
        """No input → neuron decays back to resting."""
        n = Neuron("quiet", layer_name="hidden")
        # Push membrane up slightly
        n.receive_input(5.0)
        n.step(dt_ms=1.0)
        # Run without input — should decay
        for _ in range(30):
            n.step(dt_ms=1.0)
        assert abs(n.membrane_potential - n.config.resting_potential) < 10.0
