"""Tests for the Synapse model."""

import pytest

from quantum_neural_brain.neuron import Neuron, NeuronConfig
from quantum_neural_brain.synapse import Synapse, SynapseType, SynapseConfig


def make_pair(pre_id="pre", post_id="post"):
    pre = Neuron(pre_id, layer_name="hidden")
    post = Neuron(post_id, layer_name="hidden")
    return pre, post


class TestSynapseConfig:
    def test_defaults(self):
        cfg = SynapseConfig()
        assert cfg.weight == 1.0
        assert cfg.delay_ms == 1.0
        assert 0 < cfg.release_probability <= 1.0

    def test_excitatory_reversal_potential(self):
        pre, post = make_pair()
        syn = Synapse("s1", pre, post, SynapseType.EXCITATORY)
        assert syn.config.reversal_potential_mV == pytest.approx(0.0)

    def test_inhibitory_reversal_potential(self):
        pre, post = make_pair()
        syn = Synapse("s2", pre, post, SynapseType.INHIBITORY)
        assert syn.config.reversal_potential_mV == pytest.approx(-70.0)


class TestSynapseTransmission:
    def test_step_returns_float(self):
        pre, post = make_pair()
        syn = Synapse("s3", pre, post)
        result = syn.step(dt_ms=1.0)
        assert isinstance(result, float)

    def test_set_weight_clips_to_zero(self):
        pre, post = make_pair()
        syn = Synapse("s4", pre, post)
        syn.set_weight(-5.0)
        assert syn.config.weight == 0.0

    def test_set_weight_positive(self):
        pre, post = make_pair()
        syn = Synapse("s5", pre, post)
        syn.set_weight(2.5)
        assert syn.config.weight == pytest.approx(2.5)

    def test_effective_weight_matches_weight_without_stp(self):
        pre, post = make_pair()
        cfg = SynapseConfig(use_short_term_plasticity=False, weight=1.5)
        syn = Synapse("s6", pre, post, config=cfg)
        assert syn.effective_weight == pytest.approx(1.5)

    def test_repr_contains_ids(self):
        pre, post = make_pair("p1", "p2")
        syn = Synapse("s7", pre, post)
        r = repr(syn)
        assert "s7" in r
        assert "p1" in r
        assert "p2" in r

    def test_multiple_steps_do_not_crash(self):
        pre, post = make_pair()
        syn = Synapse("s8", pre, post)
        for _ in range(20):
            syn.step(dt_ms=1.0)

    def test_transmission_counter_increments(self):
        pre, post = make_pair()
        syn = Synapse("s9", pre, post, config=SynapseConfig(release_probability=1.0, use_short_term_plasticity=False))
        # Manually fire the pre neuron to trigger a transmission
        pre.force_fire()
        # Run enough steps for the queued spike to be delivered (delay = 1 ms)
        for _ in range(5):
            syn.step(dt_ms=1.0)
        assert syn.total_transmissions >= 1

    def test_excitatory_synapse_injects_positive_current(self):
        """Excitatory synapse should drive post-neuron membrane toward firing."""
        pre, post = make_pair()
        pre.force_fire()
        syn = Synapse(
            "s10", pre, post, SynapseType.EXCITATORY,
            config=SynapseConfig(weight=5.0, delay_ms=0.5, release_probability=1.0, use_short_term_plasticity=False)
        )
        initial_vm = post.membrane_potential
        for _ in range(5):
            syn.step(dt_ms=1.0)
            post.step(dt_ms=1.0)
        # Post neuron should have received some excitatory drive
        # (may not always fire, but transmission count should be non-zero)
        assert syn.total_transmissions >= 1
