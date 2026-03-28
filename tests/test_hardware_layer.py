"""Tests for the HardwareLayer and HardwareNode."""

import pytest

from quantum_neural_brain.neuron import Neuron
from quantum_neural_brain.hardware_layer import (
    HardwareLayer,
    HardwareNode,
    HardwareSpec,
    NodeType,
)


class TestHardwareNode:
    def setup_method(self):
        self.node = HardwareNode("node0", HardwareSpec(node_type=NodeType.CM4_COMPUTE))

    def test_initial_status(self):
        status = self.node.status()
        assert status["node_id"] == "node0"
        assert status["node_type"] == "CM4_COMPUTE"
        assert status["neurons"] == 0

    def test_dram_read_returns_positive_latency(self):
        latency = self.node.dram_read(1.0)
        assert latency > 0

    def test_dram_write_returns_positive_latency(self):
        latency = self.node.dram_write(1.0)
        assert latency > 0

    def test_nvme_persist_returns_non_negative(self):
        latency = self.node.nvme_persist({"key": "value"})
        assert latency >= 0

    def test_nvme_load_returns_positive_latency(self):
        latency = self.node.nvme_load(1024)
        assert latency >= 0

    def test_dram_read_counter_increments(self):
        self.node.dram_read()
        assert self.node.dram_reads == 1

    def test_dram_write_counter_increments(self):
        self.node.dram_write()
        assert self.node.dram_writes == 1

    def test_is_throttled_false_at_normal_temp(self):
        assert self.node.is_throttled is False

    def test_send_spike_message(self):
        target = HardwareNode("target", HardwareSpec())
        latency = self.node.send_spike_message(target, "n1", 0.0)
        assert latency > 0
        assert self.node.total_messages_sent == 1
        assert target.total_messages_received == 1

    def test_repr(self):
        r = repr(self.node)
        assert "node0" in r

    def test_effective_clock_throttled(self):
        # Push temp to throttle
        self.node.spec.operating_temp_c = self.node.spec.thermal_throttle_temp_c
        assert self.node.effective_clock_mhz < self.node.spec.clock_speed_mhz


class TestHardwareLayer:
    def test_create_default_array(self):
        layer = HardwareLayer()
        layer.create_default_array(num_cm4_nodes=4)
        assert "cm4_0" in layer.nodes
        assert "dram_0" in layer.nodes
        assert "nvme_0" in layer.nodes

    def test_get_existing_node(self):
        layer = HardwareLayer()
        layer.create_default_array(2)
        node = layer.get_node("cm4_0")
        assert node is not None
        assert node.node_id == "cm4_0"

    def test_get_missing_node_returns_none(self):
        layer = HardwareLayer()
        assert layer.get_node("nonexistent") is None

    def test_connect_nodes_creates_topology(self):
        layer = HardwareLayer()
        n1 = HardwareNode("a", HardwareSpec())
        n2 = HardwareNode("b", HardwareSpec())
        layer.add_node(n1)
        layer.add_node(n2)
        layer.connect_nodes("a", "b")
        assert "b" in layer._topology["a"]
        assert "a" in layer._topology["b"]

    def test_all_neurons_collects_across_nodes(self):
        layer = HardwareLayer()
        n1 = HardwareNode("x", HardwareSpec())
        n2 = HardwareNode("y", HardwareSpec())
        neuron_a = Neuron("na", layer_name="hidden")
        neuron_b = Neuron("nb", layer_name="hidden")
        n1.neurons.append(neuron_a)
        n2.neurons.append(neuron_b)
        layer.add_node(n1)
        layer.add_node(n2)
        all_n = layer.all_neurons()
        assert neuron_a in all_n
        assert neuron_b in all_n

    def test_cluster_status_returns_list(self):
        layer = HardwareLayer()
        layer.create_default_array(2)
        status = layer.cluster_status()
        assert isinstance(status, list)
        assert len(status) == 4  # 2 CM4 + 1 DRAM + 1 NVMe

    def test_persist_all_weights_runs_without_error(self):
        from quantum_neural_brain.synapse import Synapse, SynapseType, SynapseConfig
        layer = HardwareLayer()
        layer.create_default_array(2)
        pre = Neuron("pre", layer_name="hidden")
        post = Neuron("post", layer_name="hidden")
        syn = Synapse("s0", pre, post, SynapseType.EXCITATORY, SynapseConfig())
        layer.persist_all_weights([syn])  # Should not raise

    def test_repr(self):
        layer = HardwareLayer()
        layer.create_default_array(2)
        r = repr(layer)
        assert "HardwareLayer" in r
