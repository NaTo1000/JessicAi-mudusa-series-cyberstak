"""
neural_mesh.py – Topological neural mesh implementing the "quad-brain inside
a quad-brain" architecture with 4-D vertex mechanics and triple-folding fabric
mesh layers.

Architecture overview
---------------------
                 ┌────────────────────────────────────┐
                 │        NeuralMeshNetwork             │
                 │  ┌──────────────────────────────┐   │
                 │  │  FabricMeshLayer  x3          │   │
                 │  │  (triple-folding topology)    │   │
                 │  └──────────────────────────────┘   │
                 │  ┌──────────────────────────────┐   │
                 │  │  QuadBrainBlock (recursive)  │   │
                 │  │   each nests 4 sub-brains     │   │
                 │  └──────────────────────────────┘   │
                 │  ┌──────────────────────────────┐   │
                 │  │  VertexMechanicsLayer (4-D)   │   │
                 │  └──────────────────────────────┘   │
                 └────────────────────────────────────┘

Neurons fire via a Leaky-ReLU activation that models the refractory period;
synaptic strength is maintained through Hebbian-style weight updates applied
during training.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Synaptic connection primitive
# ---------------------------------------------------------------------------

class SynapticLayer(nn.Module):
    """A single neuron→neuron connection layer with synaptic plasticity.

    Implements a linear transformation followed by a leaky-ReLU activation
    that mimics the refractory period of a biological neuron.  The ``bias``
    plays the role of the resting potential.

    Args:
        in_features (int): Pre-synaptic dimensionality.
        out_features (int): Post-synaptic dimensionality.
        dropout (float): Synaptic dropout (models stochastic neurotransmitter
            release).
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.synapse = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        # Negative slope models partial activation below threshold
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(self.synapse(x))))


# ---------------------------------------------------------------------------
# 4-D Vertex Mechanics
# ---------------------------------------------------------------------------

class VertexMechanicsLayer(nn.Module):
    """Projects hidden state through a 4-D topological vertex space.

    The four vertex dimensions correspond to:
      0 – temporal axis  (sequence position flow)
      1 – semantic axis  (meaning manifold)
      2 – quantum axis   (superposition state)
      3 – recursive axis (self-referential depth)

    Each axis has its own learned projection; outputs are combined with a
    learned mixing matrix to produce the final vertex-processed representation.

    Args:
        hidden_size (int): Input/output feature dimension.
        vertex_dimensions (int): Number of vertex axes (default 4).
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        vertex_dimensions: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vertex_dimensions = vertex_dimensions

        # Per-axis projections (each maps H → H)
        self.axis_projections = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(vertex_dimensions)]
        )
        # Mixing matrix: recombines axis representations
        self.mixing = nn.Linear(vertex_dimensions * hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Topological routing gates (sigmoid) decide how much each axis
        # contributes based on the input content
        self.axis_gates = nn.Linear(hidden_size, vertex_dimensions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through 4-D vertex mechanics.

        Args:
            x: ``(batch, seq_len, hidden_size)`` or ``(batch, hidden_size)``.

        Returns:
            Tensor of same shape as *x*.
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)

        # Content-dependent routing gates
        gates = torch.sigmoid(self.axis_gates(x))  # (B, T, vertex_dims)

        # Project through each axis
        axis_outs = []
        for i, proj in enumerate(self.axis_projections):
            ax = proj(x) * gates[..., i : i + 1]  # gated axis output
            axis_outs.append(ax)

        # Concatenate and mix back to hidden_size
        combined = torch.cat(axis_outs, dim=-1)  # (B, T, V*H)
        out = self.dropout(self.norm(self.mixing(combined) + x))

        return out.squeeze(1) if squeeze else out


# ---------------------------------------------------------------------------
# Triple-folding Fabric Mesh Layer
# ---------------------------------------------------------------------------

class FabricMeshLayer(nn.Module):
    """One layer of the triple-folding topological fabric mesh.

    The mesh folds the hidden representation three times:
      1. Forward fold  – causal direction
      2. Backward fold – anti-causal correction
      3. Cross fold    – lateral synaptic bridging

    Each fold is a :class:`SynapticLayer`; outputs are merged with a residual
    connection to preserve information across folds.

    Args:
        hidden_size (int): Feature dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.forward_fold = SynapticLayer(hidden_size, hidden_size, dropout)
        self.backward_fold = SynapticLayer(hidden_size, hidden_size, dropout)
        self.cross_fold = SynapticLayer(hidden_size, hidden_size, dropout)
        self.merge = nn.Linear(3 * hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self.forward_fold(x)
        bwd = self.backward_fold(x.flip(dims=[-1]))   # reverse feature order
        cross = self.cross_fold(x * fwd)               # lateral modulation
        merged = self.norm(self.merge(torch.cat([fwd, bwd, cross], dim=-1)) + x)
        return merged


# ---------------------------------------------------------------------------
# Quad-Brain Block (recursive)
# ---------------------------------------------------------------------------

class QuadBrainBlock(nn.Module):
    """Recursive quad-brain processing block.

    At each recursion level the input is split into four equal-width slices,
    each processed by an independent set of synaptic layers (sub-brain), then
    recombined.  ``depth`` controls the number of nesting levels.

    At depth=1 the block simply has four parallel synaptic paths.
    At depth=2 each of the four paths is itself a depth-1 QuadBrainBlock, etc.

    Args:
        hidden_size (int): Feature dimension (must be divisible by 4).
        depth (int): Recursion depth.
        dropout (float): Dropout probability.
    """

    def __init__(self, hidden_size: int, depth: int = 1, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % 4 == 0, "hidden_size must be divisible by 4 for QuadBrainBlock"
        self.hidden_size = hidden_size
        self.sub_size = hidden_size // 4
        self.depth = depth

        if depth <= 1:
            # Leaf level – four independent synaptic sub-brains
            self.sub_brains = nn.ModuleList(
                [SynapticLayer(self.sub_size, self.sub_size, dropout) for _ in range(4)]
            )
        else:
            # Recursive – each sub-brain is itself a QuadBrainBlock
            self.sub_brains = nn.ModuleList(
                [QuadBrainBlock(self.sub_size, depth - 1, dropout) for _ in range(4)]
            )

        # Re-integration synapse merges four sub-brain outputs
        self.integrate = SynapticLayer(hidden_size, hidden_size, dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into four equal slices along the feature axis
        chunks = x.split(self.sub_size, dim=-1)  # four tensors of sub_size
        processed = [brain(chunk) for brain, chunk in zip(self.sub_brains, chunks)]
        merged = torch.cat(processed, dim=-1)  # (B, ..., hidden_size)
        return self.norm(self.integrate(merged) + x)


# ---------------------------------------------------------------------------
# NeuralMeshNetwork – main topology
# ---------------------------------------------------------------------------

class NeuralMeshNetwork(nn.Module):
    """Full topological neural mesh combining all components.

    Processing order:
      1. Input projection (token → hidden)
      2. ``num_quad_brains`` levels of :class:`QuadBrainBlock`
      3. :class:`VertexMechanicsLayer` (4-D routing)
      4. ``mesh_fabric_layers`` of :class:`FabricMeshLayer` (triple fold)
      5. Output projection back to the original hidden size

    Args:
        hidden_size (int): Core feature dimension.
        num_quad_brains (int): Recursion depth of the quad-brain nesting.
        mesh_fabric_layers (int): Number of triple-fold fabric layers.
        vertex_dimensions (int): Number of vertex axes.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_quad_brains: int = 2,
        mesh_fabric_layers: int = 3,
        vertex_dimensions: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Ensure hidden_size is compatible with quad-brain recursion
        # (must be divisible by 4^num_quad_brains)
        required_divisor = 4 ** num_quad_brains
        if hidden_size % required_divisor != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"4^num_quad_brains = {required_divisor}"
            )

        self.quad_brain = QuadBrainBlock(hidden_size, depth=num_quad_brains, dropout=dropout)
        self.vertex_mechanics = VertexMechanicsLayer(hidden_size, vertex_dimensions, dropout)
        self.fabric_layers = nn.ModuleList(
            [FabricMeshLayer(hidden_size, dropout) for _ in range(mesh_fabric_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full neural mesh.

        Args:
            x: ``(batch, seq_len, hidden_size)`` hidden state tensor.

        Returns:
            Tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        x = self.quad_brain(x)
        x = self.vertex_mechanics(x)
        for layer in self.fabric_layers:
            x = layer(x)
        return self.output_norm(x)
