"""
inference.py
------------
Quantum inference pipeline for the Quantum Quad-Brain System.

The pipeline connects the four brain units, the 4D topology, the triple-fold
mesh and the quantum fabric into a unified decision-making flow:

  Input → QuadBrainCore → SupersymmetryEngine (error-correct) →
  TopologicalVertex4D (propagate) → TripleFoldingMesh (fold) →
  QuantumFabricLayer (ingest) → Decision vector
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .quad_brain_core import QuadBrainCore
from .topological_vertex import TopologicalVertex4D
from .triple_fold_mesh import TripleFoldingMesh
from .supersymmetry_engine import SupersymmetryEngine
from .quantum_fabric import QuantumFabricLayer


class InferenceResult:
    """Container for a single inference cycle result."""

    def __init__(
        self,
        cycle: int,
        decision: int,
        probability: float,
        coherence: float,
        susy_invariant: float,
        mesh_coherence: float,
        fabric_entropy: float,
        raw_probs: np.ndarray,
    ) -> None:
        self.cycle = cycle
        self.decision = decision
        self.probability = probability
        self.coherence = coherence
        self.susy_invariant = susy_invariant
        self.mesh_coherence = mesh_coherence
        self.fabric_entropy = fabric_entropy
        self.raw_probs = raw_probs

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "decision": self.decision,
            "probability": round(self.probability, 6),
            "coherence": round(self.coherence, 6),
            "susy_invariant": round(self.susy_invariant, 8),
            "mesh_coherence": round(self.mesh_coherence, 6),
            "fabric_entropy": round(self.fabric_entropy, 4),
        }

    def __repr__(self) -> str:
        return (
            f"InferenceResult(cycle={self.cycle}, "
            f"decision={self.decision}, "
            f"prob={self.probability:.4f}, "
            f"coherence={self.coherence:.4f})"
        )


class QuantumInferencePipeline:
    """
    End-to-end quantum inference pipeline.

    Each call to ``run_cycle`` pushes an input vector through the full
    quad-brain stack and returns a structured ``InferenceResult``.
    """

    def __init__(
        self,
        state_dims: int = 64,
        mesh_rows: int = 8,
        mesh_cols: int = 8,
        num_fabric_layers: int = 4,
        topo_vertices: int = 16,
    ) -> None:
        self.state_dims = state_dims
        self.cycle_count = 0
        self.history: List[InferenceResult] = []

        # Core components
        self.quad_brain = QuadBrainCore(state_dims=state_dims)
        self.susy_engine = SupersymmetryEngine(state_dims=state_dims)
        self.topology = TopologicalVertex4D(initial_vertices=topo_vertices)
        self.mesh = TripleFoldingMesh(rows=mesh_rows, cols=mesh_cols)
        self.fabric_layers: List[QuantumFabricLayer] = [
            QuantumFabricLayer(layer_id=i, cluster_size=max(state_dims // 4, 4))
            for i in range(num_fabric_layers)
        ]

    # ------------------------------------------------------------------
    # Core pipeline step
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        input_data: Optional[np.ndarray] = None,
        noise_sigma: float = 0.02,
    ) -> InferenceResult:
        """Execute one full inference cycle."""

        # 1. Prepare input
        if input_data is None:
            rng = np.random.default_rng()
            input_data = rng.standard_normal(self.state_dims)
        input_vec = input_data.astype(complex)
        if np.linalg.norm(input_vec) > 0:
            input_vec /= np.linalg.norm(input_vec)

        # 2. Quad-brain parallel inference
        probs = self.quad_brain.parallel_inference(input_vec.real)

        # 3. SUSY error correction on the collective brain state
        collective = self.quad_brain.get_collective_state()
        self.susy_engine.inject_noise(sigma=noise_sigma)
        corrected = self.susy_engine.apply_to_state(collective)

        # 4. Topology: propagate amplitudes and rotate in 4D
        self.topology.propagate_amplitudes(steps=2)
        self.topology.rotate_all(plane=(0, 3), angle=0.05)  # wz-plane rotation
        topo_amps = self.topology.amplitude_vector()

        # 5. Triple-fold mesh: inject corrected state and fold
        mesh_input = corrected.real[: self.mesh.rows * self.mesh.cols].reshape(
            self.mesh.rows, self.mesh.cols
        )
        for layer in self.mesh.layers:
            layer.cells += 0.1 * mesh_input.astype(complex)
            layer.cells /= np.linalg.norm(layer.cells)
        self.mesh.triple_fold()
        self.mesh.diffuse_all()

        # 6. Quantum fabric: ingest and fold
        fabric_data = self.mesh.fabric_state()
        for fl in self.fabric_layers:
            fl.ingest(fabric_data)
            fl.fold()

        # 7. Final decision: combine probs with topology amplitudes
        n = min(len(probs), len(topo_amps))
        combined = probs[:n] * topo_amps[:n]
        if combined.sum() > 0:
            combined /= combined.sum()
        else:
            combined = probs[:n] / (probs[:n].sum() + 1e-12)

        decision = int(np.argmax(combined))
        probability = float(combined[decision]) if len(combined) > 0 else 0.0

        result = InferenceResult(
            cycle=self.cycle_count,
            decision=decision,
            probability=probability,
            coherence=self.susy_engine.coherence(),
            susy_invariant=self.susy_engine.susy_invariant(),
            mesh_coherence=self.mesh.inter_layer_coherence(),
            fabric_entropy=sum(fl.total_entropy() for fl in self.fabric_layers),
            raw_probs=probs,
        )
        self.history.append(result)
        self.cycle_count += 1
        return result

    def run_batch(
        self,
        inputs: List[np.ndarray],
        noise_sigma: float = 0.02,
    ) -> List[InferenceResult]:
        """Run multiple inference cycles."""
        return [self.run_cycle(inp, noise_sigma=noise_sigma) for inp in inputs]

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all inference cycles so far."""
        if not self.history:
            return {"cycles": 0}
        decisions = [r.decision for r in self.history]
        coherences = [r.coherence for r in self.history]
        return {
            "total_cycles": self.cycle_count,
            "decisions": decisions,
            "mean_coherence": float(np.mean(coherences)),
            "min_coherence": float(np.min(coherences)),
            "max_coherence": float(np.max(coherences)),
            "unique_decisions": len(set(decisions)),
        }

    def __repr__(self) -> str:
        return (
            f"QuantumInferencePipeline("
            f"state_dims={self.state_dims}, "
            f"cycles={self.cycle_count}, "
            f"fabric_layers={len(self.fabric_layers)})"
        )
