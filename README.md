# JessicAi-mudusa-series-cyberstak

## Quantum Quad-Brain System

A fully functional **quantum quad-brain inference engine** with infinite topological
scalability, built on four interconnected pillars:

| Pillar | Module | Description |
|---|---|---|
| **Quad Brain** | `quad_brain_core.py` | Four brain units in quantum superposition, entangled and running parallel inference |
| **4D Topology** | `topological_vertex.py` | Dynamically scalable graph of 4D vertices with quantum amplitude diffusion |
| **Triple Fold Mesh** | `triple_fold_mesh.py` | Three interleaved mesh layers unified through triple-fold beam-splitter operations |
| **Supersymmetry Engine** | `supersymmetry_engine.py` | SUSY-inspired error correction via bosonic/fermionic superpartner cancellation |
| **Quantum Fabric** | `quantum_fabric.py` | Multi-tiered vaulted cluster containers with dynamic fold/re-fold optimisation |
| **Inference Pipeline** | `inference.py` | End-to-end pipeline wiring all components into a decision-making cycle |
| **Visualizations** | `visualization.py` | 4D vertex projection, mesh heatmaps, inference history, dashboard |

---

## Architecture Overview

```
  Input stimulus
       │
       ▼
  ┌────────────────────────────────────────────────────────┐
  │                  QuantumQuadBrain                      │
  │                                                        │
  │  QuadBrainCore (×4 brains, superposition)             │
  │      │   parallel_inference()                          │
  │      ▼                                                 │
  │  SupersymmetryEngine (SUSY error correction)          │
  │      │   inject_noise → correct_errors                 │
  │      ▼                                                 │
  │  TopologicalVertex4D (4D graph, N vertices)           │
  │      │   propagate_amplitudes → rotate_all             │
  │      ▼                                                 │
  │  TripleFoldingMesh (3 layers × triple fold)           │
  │      │   fold(FORWARD|BACKWARD|TRANSVERSE)             │
  │      ▼                                                 │
  │  QuantumFabricLayer × 4 (vaulted cluster containers)  │
  │      │   ingest → fold → re_fold                       │
  │      ▼                                                 │
  │  Decision vector  →  InferenceResult                  │
  └────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Quad Brain Architecture
- **Four `BrainUnit` instances** each maintain a 64-dimensional complex amplitude
  vector (quantum superposition state).
- All four units are pairwise-entangled via Bell-state mixing at initialisation.
- `parallel_inference()` runs all four brains simultaneously and combines results
  via constructive interference (average probability distribution).

### 2. 4D Vertices and Topological Design
- Each `Vertex4D` lives in 4D space (w, x, y, z) with a unit complex amplitude.
- Vertices are connected by a k-nearest-neighbour graph in 4D Euclidean space.
- `add_vertex()` grows the graph dynamically — demonstrating **infinite scalability**.
- `propagate_amplitudes()` diffuses quantum information through the topology via a
  graph-Laplacian random walk.
- `rotate_all(plane, angle)` rotates every vertex in any of the six 4D rotation planes.

### 3. Triple Folding Mechanics
- Three `MeshLayer` grids each holding a 2D complex amplitude field.
- `fold(direction)` applies a quantum beam-splitter interaction between each pair of
  adjacent layers — analogous to the BS gate in linear quantum optics.
- `triple_fold()` executes all three fold directions (FORWARD, BACKWARD, TRANSVERSE)
  in a single cycle, fully interlacing all layers.
- `inter_layer_coherence()` measures pairwise layer overlap (|⟨ψ_a|ψ_b⟩|²).

### 4. Supersymmetry Properties
- `SuperpartnerChannel` maintains a paired bosonic and fermionic state vector.
- `correct_errors()` projects the noisy state onto the bosonic channel and
  subtracts the fermionic noise component — cancelling decoherence.
- `restore_symmetry()` rebalances channel energies to preserve the SUSY invariant.
- The SUSY invariant (energy difference between channels) approaches zero under ideal
  conditions.

### 5. Quantum Fabric Layers
- `DataCluster` holds a quantum state and supports Von-Neumann entropy measurement.
- `VaultedContainer` groups clusters and supports nested container-in-container
  isolation.
- `QuantumFabricLayer` provides `fold()` / `re_fold()` for bidirectional
  load redistribution across containers and `ingest()` for data intake.

---

## Quick Start

### Requirements

```
numpy
scipy
matplotlib
```

Install:

```bash
pip install numpy scipy matplotlib
```

### Run the demo

```bash
python main.py                        # 20 inference cycles, output/ dir
python main.py --cycles 50 --output-dir results
```

### Use the API

```python
from quantum_quad_brain import QuantumQuadBrain
import numpy as np

brain = QuantumQuadBrain(
    state_dims=64,        # quantum state dimensions per brain unit
    topo_vertices=16,     # initial 4D topology vertices
    mesh_rows=8,          # mesh layer grid size
    mesh_cols=8,
    num_fabric_layers=4,  # stacked quantum fabric layers
)

# Single inference cycle
result = brain.think(input_data=np.random.randn(64))
print(result)
# InferenceResult(cycle=0, decision=15, prob=0.2003, coherence=1.0000)

# Batch processing
inputs = [np.random.randn(64) for _ in range(10)]
results = brain.think_batch(inputs)

# Expand topology (infinite scalability)
brain.expand_topology(new_vertices=8)

# System status
import json
print(json.dumps(brain.status(), indent=2))
```

### Visualizations

```python
from quantum_quad_brain.visualization import save_all, plot_dashboard
import matplotlib.pyplot as plt

# Save all plots
saved = save_all(brain, output_dir="output")

# Interactive dashboard
fig = plot_dashboard(brain)
plt.show()
```

Generated visualizations:

| File | Contents |
|---|---|
| `topology_wx.png` | 4D topology projected onto w-x plane |
| `topology_yz.png` | 4D topology projected onto y-z plane |
| `mesh_layers.png` | Amplitude heatmaps for all three mesh layers |
| `inference_history.png` | Decision index, coherence, and probability over cycles |
| `fabric_entropy.png` | Per-layer quantum fabric entropy bar chart |
| `dashboard.png` | Combined full-system dashboard |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All 46 tests cover each subsystem individually and as an integrated pipeline.

---

## Theoretical Background

| Concept | Implementation analogue |
|---|---|
| Quantum superposition | Complex amplitude state vector per brain unit |
| Quantum entanglement | Bell-state mixing between brain unit pairs |
| Hadamard gate | FFT-based amplitude diffusion |
| Quantum measurement | Probabilistic collapse via `np.random.choice` |
| Quantum noise / decoherence | Gaussian complex noise injection |
| Supersymmetry | Bosonic/fermionic channel pairs with energy balance |
| SUSY error correction | Projection + fermionic noise subtraction |
| 4D rotation | SO(4) rotation in any of the six orthogonal planes |
| Topological propagation | Graph-Laplacian random walk on k-NN graph in ℝ⁴ |
| Beam-splitter fold | 50/50 unitary mix: (a+b)/√2, (a-b)/√2 |
| Von-Neumann entropy | −Σ pᵢ log₂(pᵢ) over cluster state probability |