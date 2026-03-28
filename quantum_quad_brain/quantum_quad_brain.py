"""
quantum_quad_brain.py
---------------------
Top-level ``QuantumQuadBrain`` class: the complete system wiring all
components into a single, easy-to-use API.

Architecture:

  ┌──────────────────────────────────────────────────┐
  │              QuantumQuadBrain                    │
  │                                                  │
  │  QuadBrainCore   ←──→  SupersymmetryEngine       │
  │      ↕                       ↕                   │
  │  TopologicalVertex4D  ←──→  TripleFoldingMesh    │
  │                      ↘  ↙                        │
  │              QuantumFabricLayer × N              │
  │                      ↓                           │
  │            QuantumInferencePipeline              │
  └──────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .inference import QuantumInferencePipeline, InferenceResult
from .quad_brain_core import QuadBrainCore
from .topological_vertex import TopologicalVertex4D
from .triple_fold_mesh import TripleFoldingMesh
from .supersymmetry_engine import SupersymmetryEngine
from .quantum_fabric import QuantumFabricLayer


class QuantumQuadBrain:
    """
    Quantum Quad-Brain System

    A unified interface to the full quad-layered quantum inference engine:

    * Four brain cores in quantum superposition (QuadBrainCore)
    * Infinite-scalability 4D topological vertex graph (TopologicalVertex4D)
    * Triple-folding fabric mesh (TripleFoldingMesh)
    * SUSY error-correction engine (SupersymmetryEngine)
    * Multi-tiered vaulted fabric layers (QuantumFabricLayer)
    * End-to-end inference pipeline (QuantumInferencePipeline)

    Parameters
    ----------
    state_dims : int
        Dimensionality of each brain unit's quantum state vector.
    topo_vertices : int
        Initial number of 4D topological vertices.
    mesh_rows, mesh_cols : int
        Shape of each triple-fold mesh layer grid.
    num_fabric_layers : int
        Number of stacked quantum fabric layers.
    """

    def __init__(
        self,
        state_dims: int = 64,
        topo_vertices: int = 16,
        mesh_rows: int = 8,
        mesh_cols: int = 8,
        num_fabric_layers: int = 4,
    ) -> None:
        self.state_dims = state_dims
        self.pipeline = QuantumInferencePipeline(
            state_dims=state_dims,
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
            num_fabric_layers=num_fabric_layers,
            topo_vertices=topo_vertices,
        )

    # ------------------------------------------------------------------
    # Convenience accessors to sub-systems
    # ------------------------------------------------------------------

    @property
    def quad_brain(self) -> QuadBrainCore:
        return self.pipeline.quad_brain

    @property
    def topology(self) -> TopologicalVertex4D:
        return self.pipeline.topology

    @property
    def mesh(self) -> TripleFoldingMesh:
        return self.pipeline.mesh

    @property
    def susy_engine(self) -> SupersymmetryEngine:
        return self.pipeline.susy_engine

    @property
    def fabric_layers(self) -> List[QuantumFabricLayer]:
        return self.pipeline.fabric_layers

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def think(
        self,
        input_data: Optional[np.ndarray] = None,
        noise_sigma: float = 0.02,
    ) -> InferenceResult:
        """
        Run one inference cycle — the system 'thinks'.

        Parameters
        ----------
        input_data : np.ndarray, optional
            Input stimulus vector.  If None, random noise is used.
        noise_sigma : float
            Environmental noise level (decoherence simulation).

        Returns
        -------
        InferenceResult
            Structured result with decision, probabilities, coherence metrics.
        """
        return self.pipeline.run_cycle(input_data=input_data, noise_sigma=noise_sigma)

    def think_batch(
        self,
        inputs: List[np.ndarray],
        noise_sigma: float = 0.02,
    ) -> List[InferenceResult]:
        """Process a batch of inputs sequentially."""
        return self.pipeline.run_batch(inputs, noise_sigma=noise_sigma)

    def expand_topology(self, new_vertices: int = 8) -> None:
        """
        Grow the 4D topological graph by adding new vertices.
        Demonstrates infinite scalability.
        """
        for _ in range(new_vertices):
            self.topology.add_vertex()

    def status(self) -> Dict[str, Any]:
        """Return a comprehensive status report of all subsystems."""
        p = self.pipeline
        return {
            "state_dims": self.state_dims,
            "cycle_count": p.cycle_count,
            "quad_brain": {
                "num_brains": p.quad_brain.NUM_BRAINS,
                "collective_state_norm": float(np.linalg.norm(p.quad_brain.get_collective_state())),
            },
            "topology": {
                "vertices": len(p.topology.vertices),
                "edges": len(p.topology.edges),
            },
            "mesh": {
                "layers": p.mesh.NUM_LAYERS,
                "fold_count": p.mesh.fold_count,
                "inter_layer_coherence": p.mesh.inter_layer_coherence(),
            },
            "susy_engine": {
                "coherence": p.susy_engine.coherence(),
                "corrections": p.susy_engine.correction_count,
                "susy_invariant": p.susy_engine.susy_invariant(),
            },
            "fabric_layers": [
                {
                    "id": fl.layer_id,
                    "fold_operations": fl.fold_operations,
                    "total_entropy": fl.total_entropy(),
                }
                for fl in p.fabric_layers
            ],
        }

    def __repr__(self) -> str:
        s = self.status()
        return (
            f"QuantumQuadBrain("
            f"state_dims={self.state_dims}, "
            f"cycles={s['cycle_count']}, "
            f"topo_vertices={s['topology']['vertices']}, "
            f"mesh_coherence={s['mesh']['inter_layer_coherence']:.4f}, "
            f"susy_coherence={s['susy_engine']['coherence']:.4f})"
        )
