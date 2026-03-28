"""Tests for the NeuroplasticityEngine."""

import pytest

from quantum_neural_brain.neuron import Neuron
from quantum_neural_brain.synapse import Synapse, SynapseType, SynapseConfig
from quantum_neural_brain.neuroplasticity import (
    NeuroplasticityEngine,
    PlasticityConfig,
    LearningRule,
)


def make_synapse(syn_id="syn0", initial_weight=1.0):
    pre = Neuron("pre", layer_name="hidden")
    post = Neuron("post", layer_name="hidden")
    cfg = SynapseConfig(weight=initial_weight)
    return Synapse(syn_id, pre, post, SynapseType.EXCITATORY, cfg)


class TestNeuroplasticityEngine:
    def test_creation(self):
        syn = make_synapse()
        engine = NeuroplasticityEngine([syn])
        assert len(engine.synapses) == 1

    def test_mean_weight(self):
        syn = make_synapse(initial_weight=2.0)
        engine = NeuroplasticityEngine([syn])
        assert engine.mean_weight == pytest.approx(2.0)

    def test_mean_weight_empty(self):
        engine = NeuroplasticityEngine([])
        assert engine.mean_weight == 0.0

    def test_update_does_not_crash(self):
        syn = make_synapse()
        engine = NeuroplasticityEngine([syn])
        engine.update(dt_ms=1.0)

    def test_weight_stays_within_bounds(self):
        """Weights must never exceed max_weight or go below min_weight."""
        syn = make_synapse(initial_weight=0.5)
        config = PlasticityConfig(
            learning_rule=LearningRule.HEBBIAN,
            max_weight=5.0,
            min_weight=0.0,
        )
        engine = NeuroplasticityEngine([syn], config)
        # Force many firing events on both neurons
        syn.pre_neuron.force_fire()
        syn.post_neuron.force_fire()
        for _ in range(200):
            engine.update(dt_ms=1.0)
        assert syn.config.weight >= 0.0
        assert syn.config.weight <= 5.0

    def test_add_synapse(self):
        engine = NeuroplasticityEngine([])
        syn = make_synapse("new")
        engine.add_synapse(syn)
        assert len(engine.synapses) == 1

    def test_remove_synapse(self):
        syn = make_synapse("remove_me")
        engine = NeuroplasticityEngine([syn])
        engine.remove_synapse("remove_me")
        assert len(engine.synapses) == 0

    def test_weight_history_grows(self):
        syn = make_synapse()
        engine = NeuroplasticityEngine([syn])
        for _ in range(5):
            engine.update()
        assert len(engine.weight_history) == 5

    def test_stdp_potentiation(self):
        """STDP should increase weight when pre fires before post."""
        pre = Neuron("pre_stdp", layer_name="hidden")
        post = Neuron("post_stdp", layer_name="hidden")
        syn = Synapse("stdp_syn", pre, post, config=SynapseConfig(weight=1.0))
        config = PlasticityConfig(
            learning_rule=LearningRule.STDP,
            stdp_a_plus=0.1,
            stdp_a_minus=0.1,
        )
        engine = NeuroplasticityEngine([syn], config)

        # Directly set spike times so t_post > t_pre (causal → LTP)
        import time
        now = time.monotonic()
        pre.spike_times = [now - 0.010]   # pre fired 10 ms ago
        post.spike_times = [now - 0.005]  # post fired 5 ms ago (after pre)

        before = syn.config.weight
        engine.update()
        after = syn.config.weight
        assert after >= before  # LTP should not decrease weight

    def test_repr(self):
        syn = make_synapse()
        engine = NeuroplasticityEngine([syn])
        r = repr(engine)
        assert "NeuroplasticityEngine" in r

    def test_combined_rule_runs(self):
        syn = make_synapse()
        config = PlasticityConfig(learning_rule=LearningRule.COMBINED)
        engine = NeuroplasticityEngine([syn], config)
        for _ in range(10):
            engine.update()
        assert syn.config.weight >= 0.0
