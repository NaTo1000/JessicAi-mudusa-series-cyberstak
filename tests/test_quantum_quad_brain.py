"""
Tests for the Quantum Quad-Brain System.
"""

import math
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# QuadBrainCore
# ---------------------------------------------------------------------------
from quantum_quad_brain.quad_brain_core import BrainUnit, QuadBrainCore


class TestBrainUnit:
    def test_state_normalised(self):
        u = BrainUnit(unit_id=0, state_dims=16, seed=1)
        assert math.isclose(np.linalg.norm(u.state), 1.0, abs_tol=1e-9)

    def test_hadamard_preserves_norm(self):
        u = BrainUnit(unit_id=1, state_dims=16, seed=2)
        u.apply_hadamard_layer()
        assert math.isclose(np.linalg.norm(u.state), 1.0, abs_tol=1e-6)

    def test_phase_kick_preserves_magnitude(self):
        u = BrainUnit(unit_id=2, state_dims=16, seed=3)
        before = np.abs(u.state).copy()
        u.phase_kick(math.pi / 4)
        np.testing.assert_allclose(np.abs(u.state), before, atol=1e-9)

    def test_measure_collapse_valid_index(self):
        u = BrainUnit(unit_id=3, state_dims=32, seed=4)
        for _ in range(10):
            idx = u.measure_collapse()
            assert 0 <= idx < 32

    def test_entanglement_symmetric(self):
        a = BrainUnit(unit_id=0, state_dims=16, seed=10)
        b = BrainUnit(unit_id=1, state_dims=16, seed=11)
        a.entangle_with(b)
        np.testing.assert_allclose(a.state, b.state, atol=1e-9)

    def test_inference_step_returns_probs(self):
        u = BrainUnit(unit_id=4, state_dims=16, seed=5)
        inp = np.ones(16, dtype=float)
        probs = u.inference_step(inp)
        assert probs.shape == (16,)
        assert probs.min() >= 0
        assert math.isclose(probs.sum(), 1.0, abs_tol=1e-6)


class TestQuadBrainCore:
    def test_num_brains(self):
        core = QuadBrainCore(state_dims=16)
        assert len(core.brains) == 4

    def test_parallel_inference_shape(self):
        core = QuadBrainCore(state_dims=16)
        inp = np.random.default_rng(0).standard_normal(16)
        probs = core.parallel_inference(inp)
        assert probs.shape == (16,)
        assert probs.min() >= 0

    def test_collective_decision_in_range(self):
        core = QuadBrainCore(state_dims=32)
        inp = np.ones(32)
        decision = core.collective_decision(inp)
        assert 0 <= decision < 32

    def test_collective_state_normalised(self):
        core = QuadBrainCore(state_dims=16)
        state = core.get_collective_state()
        assert math.isclose(np.linalg.norm(state), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# TopologicalVertex4D
# ---------------------------------------------------------------------------
from quantum_quad_brain.topological_vertex import Vertex4D, TopologicalVertex4D


class TestVertex4D:
    def test_coords_shape(self):
        v = Vertex4D(vid=0)
        assert v.coords.shape == (4,)

    def test_unit_amplitude(self):
        v = Vertex4D(vid=1)
        assert math.isclose(abs(v.amplitude), 1.0, abs_tol=1e-9)

    def test_rotation_preserves_norm(self):
        v = Vertex4D(vid=2)
        before = np.linalg.norm(v.coords)
        v.rotate_4d((0, 1), math.pi / 6)
        after = np.linalg.norm(v.coords)
        assert math.isclose(before, after, abs_tol=1e-9)

    def test_distance_to_self_zero(self):
        v = Vertex4D(vid=3)
        assert v.distance_to(v) == 0.0


class TestTopologicalVertex4D:
    def test_initial_vertex_count(self):
        t = TopologicalVertex4D(initial_vertices=8)
        assert len(t.vertices) == 8

    def test_edge_keys_ordered(self):
        t = TopologicalVertex4D(initial_vertices=4)
        for a, b in t.edges:
            assert a <= b

    def test_add_vertex_increases_count(self):
        t = TopologicalVertex4D(initial_vertices=4)
        t.add_vertex()
        assert len(t.vertices) == 5

    def test_propagate_amplitudes(self):
        t = TopologicalVertex4D(initial_vertices=8)
        # Record complex amplitudes before propagation
        before = np.array([t.vertices[v].amplitude for v in sorted(t.vertices)])
        t.propagate_amplitudes(steps=3)
        after = np.array([t.vertices[v].amplitude for v in sorted(t.vertices)])
        # Complex amplitudes (phases) should change after propagation
        assert not np.allclose(before, after)

    def test_coordinate_matrix_shape(self):
        t = TopologicalVertex4D(initial_vertices=6)
        mat = t.coordinate_matrix()
        assert mat.shape == (6, 4)


# ---------------------------------------------------------------------------
# TripleFoldingMesh
# ---------------------------------------------------------------------------
from quantum_quad_brain.triple_fold_mesh import TripleFoldingMesh, FoldDirection


class TestTripleFoldingMesh:
    def test_num_layers(self):
        m = TripleFoldingMesh(rows=4, cols=4)
        assert len(m.layers) == 3

    def test_layer_energy_positive(self):
        m = TripleFoldingMesh(rows=4, cols=4)
        for layer in m.layers:
            assert layer.energy() > 0

    def test_fold_increments_count(self):
        m = TripleFoldingMesh(rows=4, cols=4)
        m.fold(FoldDirection.FORWARD)
        assert m.fold_count == 1

    def test_triple_fold_increments_by_three(self):
        m = TripleFoldingMesh(rows=4, cols=4)
        m.triple_fold()
        assert m.fold_count == 3

    def test_fabric_state_length(self):
        m = TripleFoldingMesh(rows=4, cols=4)
        state = m.fabric_state()
        assert state.shape[0] == 3 * 4 * 4

    def test_coherence_in_range(self):
        m = TripleFoldingMesh(rows=4, cols=4)
        c = m.inter_layer_coherence()
        assert 0.0 <= c <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# SupersymmetryEngine
# ---------------------------------------------------------------------------
from quantum_quad_brain.supersymmetry_engine import SupersymmetryEngine


class TestSupersymmetryEngine:
    def test_initial_coherence_one(self):
        e = SupersymmetryEngine(state_dims=16)
        assert math.isclose(e.coherence(), 1.0, abs_tol=1e-9)

    def test_noise_reduces_coherence(self):
        e = SupersymmetryEngine(state_dims=16)
        e.inject_noise(sigma=0.5)
        # After noise injection the state is still normalised
        assert math.isclose(np.linalg.norm(e.state), 1.0, abs_tol=1e-6)

    def test_correction_returns_normalised(self):
        e = SupersymmetryEngine(state_dims=16)
        e.inject_noise(sigma=0.3)
        corrected, residual = e.correct_errors()
        assert math.isclose(np.linalg.norm(corrected), 1.0, abs_tol=1e-6)
        assert residual >= 0

    def test_correction_count_increments(self):
        e = SupersymmetryEngine(state_dims=16)
        e.correct_errors()
        e.correct_errors()
        assert e.correction_count == 2

    def test_apply_to_state(self):
        e = SupersymmetryEngine(state_dims=16)
        v = np.ones(16, dtype=complex)
        out = e.apply_to_state(v)
        assert out.shape == (16,)
        assert math.isclose(np.linalg.norm(out), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# QuantumFabricLayer
# ---------------------------------------------------------------------------
from quantum_quad_brain.quantum_fabric import DataCluster, VaultedContainer, QuantumFabricLayer


class TestDataCluster:
    def test_state_normalised(self):
        c = DataCluster(cluster_id=0, size=8)
        assert math.isclose(np.linalg.norm(c.state), 1.0, abs_tol=1e-9)

    def test_entropy_non_negative(self):
        c = DataCluster(cluster_id=1, size=8)
        assert c.entropy() >= 0

    def test_update_preserves_norm(self):
        c = DataCluster(cluster_id=2, size=8)
        c.update(np.ones(8))
        assert math.isclose(np.linalg.norm(c.state), 1.0, abs_tol=1e-6)


class TestQuantumFabricLayer:
    def test_initial_containers(self):
        fl = QuantumFabricLayer(layer_id=0, num_containers=4, clusters_per_container=2, cluster_size=8)
        assert len(fl.containers) == 4

    def test_fold_increments(self):
        fl = QuantumFabricLayer(layer_id=0, num_containers=4, clusters_per_container=2, cluster_size=8)
        fl.fold()
        assert fl.fold_operations == 1

    def test_re_fold_increments(self):
        fl = QuantumFabricLayer(layer_id=0, num_containers=4, clusters_per_container=2, cluster_size=8)
        fl.re_fold()
        assert fl.fold_operations == 1

    def test_ingest_accepts_arbitrary_data(self):
        fl = QuantumFabricLayer(layer_id=0, num_containers=2, clusters_per_container=2, cluster_size=8)
        data = np.random.default_rng(0).standard_normal(100)
        fl.ingest(data)  # should not raise

    def test_fabric_state_length(self):
        fl = QuantumFabricLayer(
            layer_id=0, num_containers=2, clusters_per_container=2, cluster_size=8
        )
        state = fl.fabric_state()
        # 2 containers × 2 clusters × 8 = 32
        assert state.shape[0] == 32


# ---------------------------------------------------------------------------
# QuantumInferencePipeline
# ---------------------------------------------------------------------------
from quantum_quad_brain.inference import QuantumInferencePipeline


class TestQuantumInferencePipeline:
    def test_single_cycle(self):
        p = QuantumInferencePipeline(state_dims=32, mesh_rows=4, mesh_cols=4, topo_vertices=8)
        result = p.run_cycle()
        assert result.cycle == 0
        assert 0 <= result.decision < 32
        assert 0 <= result.probability <= 1.0 + 1e-9

    def test_cycle_count_increments(self):
        p = QuantumInferencePipeline(state_dims=32, mesh_rows=4, mesh_cols=4, topo_vertices=8)
        for _ in range(5):
            p.run_cycle()
        assert p.cycle_count == 5

    def test_history_length(self):
        p = QuantumInferencePipeline(state_dims=32, mesh_rows=4, mesh_cols=4, topo_vertices=8)
        p.run_batch([np.ones(32)] * 3)
        assert len(p.history) == 3

    def test_summary_keys(self):
        p = QuantumInferencePipeline(state_dims=32, mesh_rows=4, mesh_cols=4, topo_vertices=8)
        p.run_cycle()
        s = p.summary()
        assert "total_cycles" in s
        assert "mean_coherence" in s


# ---------------------------------------------------------------------------
# QuantumQuadBrain (integration)
# ---------------------------------------------------------------------------
from quantum_quad_brain import QuantumQuadBrain


class TestQuantumQuadBrain:
    def test_think_returns_result(self):
        brain = QuantumQuadBrain(state_dims=32, topo_vertices=8, mesh_rows=4, mesh_cols=4)
        result = brain.think()
        assert result is not None
        assert 0 <= result.decision < 32

    def test_expand_topology(self):
        brain = QuantumQuadBrain(state_dims=32, topo_vertices=8, mesh_rows=4, mesh_cols=4)
        before = len(brain.topology.vertices)
        brain.expand_topology(new_vertices=4)
        assert len(brain.topology.vertices) == before + 4

    def test_status_structure(self):
        brain = QuantumQuadBrain(state_dims=32, topo_vertices=8, mesh_rows=4, mesh_cols=4)
        brain.think()
        s = brain.status()
        assert "quad_brain" in s
        assert "topology" in s
        assert "mesh" in s
        assert "susy_engine" in s
        assert "fabric_layers" in s

    def test_think_batch(self):
        brain = QuantumQuadBrain(state_dims=32, topo_vertices=8, mesh_rows=4, mesh_cols=4)
        inputs = [np.ones(32) * i for i in range(5)]
        results = brain.think_batch(inputs)
        assert len(results) == 5
