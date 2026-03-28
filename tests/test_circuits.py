"""
tests/test_circuits.py – Unit tests for the quantum circuit simulation layer.

Run with:
    pytest tests/test_circuits.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from quantum_neural_brain.quantum_circuits import (
    QuantumCircuitBlock,
    QuantumInterferenceLayer,
)
from quantum_neural_brain.neural_mesh import (
    SynapticLayer,
    VertexMechanicsLayer,
    FabricMeshLayer,
    QuadBrainBlock,
    NeuralMeshNetwork,
)


# ---------------------------------------------------------------------------
# QuantumCircuitBlock tests
# ---------------------------------------------------------------------------

class TestQuantumCircuitBlock:
    @pytest.fixture
    def block(self):
        return QuantumCircuitBlock(num_qubits=3, depth=1, hidden_size=16)

    def test_output_shape(self, block):
        x = torch.randn(2, 16)
        out = block(x)
        assert out.shape == (2, 16)

    def test_deterministic_eval(self, block):
        block.eval()
        x = torch.randn(1, 16)
        with torch.no_grad():
            out1 = block(x)
            out2 = block(x)
        assert torch.allclose(out1, out2)

    def test_gradient_flow(self):
        block = QuantumCircuitBlock(num_qubits=2, depth=1, hidden_size=8)
        x = torch.randn(1, 8, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_different_batch_sizes(self, block):
        for batch in [1, 4, 8]:
            x = torch.randn(batch, 16)
            out = block(x)
            assert out.shape == (batch, 16)

    def test_no_nan_in_output(self, block):
        x = torch.randn(4, 16)
        out = block(x)
        assert not torch.isnan(out).any(), "NaN values in circuit block output"

    def test_small_input_padding(self):
        # hidden_size > 2^num_qubits → partial encoding
        block = QuantumCircuitBlock(num_qubits=2, depth=1, hidden_size=32)
        x = torch.randn(2, 32)
        out = block(x)
        assert out.shape == (2, 32)


# ---------------------------------------------------------------------------
# QuantumInterferenceLayer tests
# ---------------------------------------------------------------------------

class TestQuantumInterferenceLayer:
    @pytest.fixture
    def layer(self):
        return QuantumInterferenceLayer(
            num_blocks=3, num_qubits=2, depth=1, hidden_size=16
        )

    def test_output_shape_3d(self, layer):
        x = torch.randn(2, 5, 16)  # (batch, seq, hidden)
        out = layer(x)
        assert out.shape == (2, 5, 16)

    def test_output_shape_2d(self, layer):
        x = torch.randn(2, 16)  # (batch, hidden)
        out = layer(x)
        assert out.shape == (2, 16)

    def test_no_nan(self, layer):
        x = torch.randn(2, 5, 16)
        out = layer(x)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, layer):
        x = torch.randn(2, 4, 16, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# NeuralMesh component tests
# ---------------------------------------------------------------------------

class TestSynapticLayer:
    def test_shape(self):
        layer = SynapticLayer(32, 64)
        x = torch.randn(4, 32)
        out = layer(x)
        assert out.shape == (4, 64)

    def test_no_nan(self):
        layer = SynapticLayer(16, 16)
        x = torch.randn(8, 16)
        out = layer(x)
        assert not torch.isnan(out).any()


class TestVertexMechanicsLayer:
    def test_shape_3d(self):
        layer = VertexMechanicsLayer(hidden_size=32, vertex_dimensions=4)
        x = torch.randn(2, 6, 32)
        out = layer(x)
        assert out.shape == (2, 6, 32)

    def test_shape_2d(self):
        layer = VertexMechanicsLayer(hidden_size=32, vertex_dimensions=4)
        x = torch.randn(3, 32)
        out = layer(x)
        assert out.shape == (3, 32)

    def test_no_nan(self):
        layer = VertexMechanicsLayer(hidden_size=16, vertex_dimensions=4)
        x = torch.randn(2, 5, 16)
        out = layer(x)
        assert not torch.isnan(out).any()


class TestFabricMeshLayer:
    def test_shape(self):
        layer = FabricMeshLayer(hidden_size=32)
        x = torch.randn(2, 32)
        out = layer(x)
        assert out.shape == (2, 32)

    def test_no_nan(self):
        layer = FabricMeshLayer(hidden_size=16)
        x = torch.randn(3, 16)
        out = layer(x)
        assert not torch.isnan(out).any()

    def test_sequence_input(self):
        layer = FabricMeshLayer(hidden_size=16)
        x = torch.randn(2, 5, 16)
        out = layer(x)
        assert out.shape == (2, 5, 16)


class TestQuadBrainBlock:
    def test_depth_1(self):
        block = QuadBrainBlock(hidden_size=16, depth=1)
        x = torch.randn(2, 16)
        out = block(x)
        assert out.shape == (2, 16)

    def test_depth_2(self):
        block = QuadBrainBlock(hidden_size=16, depth=2)
        x = torch.randn(2, 16)
        out = block(x)
        assert out.shape == (2, 16)

    def test_sequence_input(self):
        block = QuadBrainBlock(hidden_size=16, depth=1)
        x = torch.randn(2, 5, 16)
        out = block(x)
        assert out.shape == (2, 5, 16)

    def test_hidden_size_not_divisible_by_4(self):
        with pytest.raises(AssertionError):
            QuadBrainBlock(hidden_size=10, depth=1)

    def test_no_nan(self):
        block = QuadBrainBlock(hidden_size=16, depth=1)
        x = torch.randn(3, 5, 16)
        out = block(x)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        block = QuadBrainBlock(hidden_size=16, depth=1)
        x = torch.randn(1, 4, 16, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestNeuralMeshNetwork:
    @pytest.fixture
    def mesh(self):
        return NeuralMeshNetwork(
            hidden_size=16,
            num_quad_brains=1,
            mesh_fabric_layers=1,
            vertex_dimensions=4,
        )

    def test_output_shape(self, mesh):
        x = torch.randn(2, 6, 16)
        out = mesh(x)
        assert out.shape == (2, 6, 16)

    def test_no_nan(self, mesh):
        x = torch.randn(2, 6, 16)
        out = mesh(x)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, mesh):
        x = torch.randn(1, 4, 16, requires_grad=True)
        out = mesh(x)
        out.sum().backward()
        assert x.grad is not None

    def test_invalid_hidden_size(self):
        # hidden_size not divisible by 4^num_quad_brains
        with pytest.raises(ValueError):
            NeuralMeshNetwork(hidden_size=10, num_quad_brains=1)

    def test_multi_fabric_layers(self):
        mesh = NeuralMeshNetwork(hidden_size=16, num_quad_brains=1, mesh_fabric_layers=3)
        x = torch.randn(1, 4, 16)
        out = mesh(x)
        assert out.shape == (1, 4, 16)
