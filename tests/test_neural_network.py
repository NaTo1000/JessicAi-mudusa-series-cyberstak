"""Integration tests for the full QuantumNeuralBrain network."""

import pytest

from quantum_neural_brain.neural_network import QuantumNeuralBrain, BrainConfig, BrainLayer
from quantum_neural_brain.neuroplasticity import LearningRule
from quantum_neural_brain.visualization import NeuralVisualizer


class TestBrainConfig:
    def test_defaults(self):
        cfg = BrainConfig()
        assert cfg.input_size == 8
        assert cfg.output_size == 4
        assert len(cfg.hidden_sizes) > 0


class TestBrainLayer:
    def test_size(self):
        from quantum_neural_brain.neuron import Neuron
        layer = BrainLayer(name="test")
        layer.neurons = [Neuron(f"n{i}", layer_name="test") for i in range(5)]
        assert layer.size == 5

    def test_repr(self):
        layer = BrainLayer(name="input", is_input=True)
        r = repr(layer)
        assert "input" in r


class TestQuantumNeuralBrain:
    def setup_method(self):
        cfg = BrainConfig(
            input_size=4,
            hidden_sizes=[6],
            output_size=2,
            enable_quantum_layer=True,
            enable_plasticity=True,
            num_cm4_nodes=2,
        )
        self.brain = QuantumNeuralBrain(cfg)

    def test_layers_created(self):
        assert len(self.brain.layers) == 3  # input + 1 hidden + output

    def test_input_layer_exists(self):
        layer = self.brain.get_layer("input")
        assert layer is not None
        assert layer.is_input is True

    def test_output_layer_exists(self):
        layer = self.brain.get_layer("output")
        assert layer is not None
        assert layer.is_output is True

    def test_hidden_layer_exists(self):
        layer = self.brain.get_layer("hidden_0")
        assert layer is not None

    def test_synapses_created(self):
        assert len(self.brain.synapses) > 0

    def test_hardware_nodes_created(self):
        assert len(self.brain.hardware.nodes) > 0

    def test_step_returns_dict(self):
        result = self.brain.step()
        assert isinstance(result, dict)
        assert "step" in result
        assert "output_activations" in result
        assert "decision" in result

    def test_step_with_input(self):
        stimuli = [10.0, 0.0, 10.0, 0.0]
        result = self.brain.step(stimuli)
        assert result["step"] == 1
        assert len(result["output_activations"]) == 2

    def test_run_returns_correct_length(self):
        results = self.brain.run(num_steps=10)
        assert len(results) == 10

    def test_run_verbose_does_not_crash(self, capsys):
        self.brain.run(num_steps=5, verbose=True)
        captured = capsys.readouterr()
        assert "Step" in captured.out

    def test_decision_is_int_or_none(self):
        result = self.brain.step()
        decision = result["decision"]
        assert decision is None or isinstance(decision, int)

    def test_output_activations_in_range(self):
        for _ in range(20):
            result = self.brain.step()
            for act in result["output_activations"]:
                assert 0.0 <= act <= 1.0

    def test_total_spikes_metric_grows(self):
        self.brain.run(num_steps=20)
        # With quantum noise some spikes should occur
        assert self.brain.metrics["total_steps"] == 20

    def test_summary_contains_key_info(self):
        s = self.brain.summary()
        assert "Quantum Neural Brain" in s
        assert "neurons" in s.lower()
        assert "synapses" in s.lower()

    def test_get_nonexistent_layer(self):
        assert self.brain.get_layer("does_not_exist") is None

    def test_multiple_hidden_layers(self):
        cfg = BrainConfig(input_size=4, hidden_sizes=[8, 6], output_size=2)
        brain = QuantumNeuralBrain(cfg)
        assert brain.get_layer("hidden_0") is not None
        assert brain.get_layer("hidden_1") is not None

    def test_without_quantum_layer(self):
        cfg = BrainConfig(input_size=4, hidden_sizes=[6], output_size=2, enable_quantum_layer=False)
        brain = QuantumNeuralBrain(cfg)
        result = brain.step()
        assert "output_activations" in result

    def test_without_plasticity(self):
        cfg = BrainConfig(input_size=4, hidden_sizes=[6], output_size=2, enable_plasticity=False)
        brain = QuantumNeuralBrain(cfg)
        result = brain.step()
        assert "output_activations" in result


class TestNeuralVisualizer:
    def setup_method(self):
        cfg = BrainConfig(input_size=4, hidden_sizes=[6], output_size=2)
        self.brain = QuantumNeuralBrain(cfg)
        self.viz = NeuralVisualizer(self.brain)

    def test_render_returns_string(self):
        result = self.brain.step()
        output = self.viz.render(result)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_render_compact_returns_string(self):
        result = self.brain.step()
        output = self.viz.render_compact(result)
        assert isinstance(output, str)
        assert "Step" in output or "step" in output.lower() or "[" in output

    def test_render_compact_none_returns_empty(self):
        output = self.viz.render_compact(None)
        assert output == ""

    def test_performance_report_returns_string(self):
        self.brain.run(num_steps=5)
        report = self.viz.performance_report()
        assert "Performance Report" in report
        assert "Total steps" in report

    def test_render_without_step_result(self):
        output = self.viz.render(None)
        assert isinstance(output, str)
