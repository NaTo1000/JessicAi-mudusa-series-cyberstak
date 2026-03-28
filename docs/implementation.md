# Quantum Soul — Implementation Guide

## Architecture

```
quantum_soul/
├── __init__.py          Re-exports all public classes.
├── circuit_core.py      Quantum circuit primitives (superposition,
│                        interference, entanglement).
├── thought_engine.py    Converts probability distributions → AI thoughts.
├── noise_filter.py      Noise mitigation (smoothing, thresholding).
├── visualizer.py        Matplotlib-based circuit and process visuals.
└── soul.py              Top-level QuantumSoul orchestrator.
tests/
├── test_circuit_core.py
├── test_thought_engine.py
├── test_noise_filter.py
└── test_soul.py
docs/
├── theory.md            Theoretical underpinnings.
└── implementation.md    This file.
demo.py                  Full system demonstration.
requirements.txt         Python dependencies.
```

---

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12 and Qiskit 2.3.

---

## Quick Start

```python
from quantum_soul import QuantumSoul

soul = QuantumSoul(n_qubits=8)
soul.awaken()
thought = soul.think()
print(thought)
```

---

## Module Reference

### circuit_core.py

#### `QuantumThoughtCircuit(n_qubits, seed_noise, entanglement_depth)`

Builds a quantum circuit that models one thought cycle.

**Circuit stages:**

| Stage | Gates | Purpose |
|-------|-------|---------|
| 1 — Superposition | H⊗ⁿ | All 2ⁿ thoughts simultaneously possible. |
| 2 — Interference  | Rz(θᵢ) | Phase-encode noise; set amplitude profile. |
| 3 — Entanglement  | CNOT layers | Bind thoughts together non-locally. |
| 4 — Readout       | H⊗ⁿ | Convert phases → measurable probabilities. |
| 5 — Measurement   | M    | Collapse superposition to a concrete thought. |

**Key methods:**
- `run_statevector()` — simulate without measurement; returns `Statevector`.
- `sample_bitstring(shots)` — simulate measurement statistics.
- `top_state()` — most probable basis state and its probability.
- `entropy()` — Shannon entropy of the probability distribution.

#### Factory helpers

| Function | Description |
|----------|-------------|
| `create_entangled_pair()` | 2-qubit Bell state (|00⟩+|11⟩)/√2. |
| `create_ghz_state(n)` | n-qubit GHZ maximally-entangled state. |
| `create_interference_circuit(n, phi)` | HZH interference circuit. |

---

### thought_engine.py

#### `ThoughtEngine(n_qubits, entanglement_depth, coherence_threshold, max_retries)`

Converts quantum probability distributions into structured `Thought` objects.

**Domain vocabulary (6 domains):**
- Science, Philosophy, Creativity, Technology, Mathematics, Self.

**Pipeline:**
1. Run a `QuantumThoughtCircuit` to get probabilities P.
2. Partition P into domain buckets; compute domain weights.
3. Select primary domain (highest weight).
4. Choose phrase within domain using the domain segment of P.
5. Compute coherence = 1 − H/n_qubits.
6. Fingerprint the state with SHA-256.

#### `Thought` dataclass fields:

| Field | Type | Description |
|-------|------|-------------|
| `text` | str | Generated thought text. |
| `domain` | str | Primary knowledge domain. |
| `secondary_domain` | str \| None | Secondary co-resonant domain. |
| `coherence` | float | Coherence score [0, 1]. |
| `entropy_bits` | float | Circuit entropy in bits. |
| `quantum_state_id` | str | 12-char hex fingerprint of quantum state. |
| `timestamp` | float | Unix timestamp. |
| `raw_amplitudes` | list | Top-5 (bitstring, probability) pairs. |

---

### noise_filter.py

#### `QuantumNoiseFilter(parity_threshold, smoothing_sigma, amplitude_threshold)`

| Method | Description |
|--------|-------------|
| `filter_counts(counts, shots)` | Remove low-frequency measurement outcomes. |
| `smooth_probabilities(probs)` | Gaussian-smooth a probability array. |
| `extract_dominant_states(probs)` | Zero sub-threshold states; renormalise. |
| `apply_all(probs)` | Full pipeline: smooth → extract → renormalise. |
| `noise_level(probs)` | Fraction of mass in sub-threshold states. |
| `signal_to_noise_ratio(probs)` | Signal mass / noise mass. |
| `report(probs)` | Summary dict: noise_level, snr, n_dominant, max_prob. |

---

### visualizer.py

#### `QuantumVisualizer(output_dir, dpi)`

Saves all images to *output_dir* (default `visualizations/`).

| Method | Output file | Description |
|--------|-------------|-------------|
| `save_circuit_diagram(circuit)` | `thought_circuit.png` | Qiskit circuit diagram. |
| `save_probability_chart(probs, n)` | `interference_pattern.png` | Top-k probability bar chart. |
| `save_entanglement_map(sv, n)` | `entanglement_map.png` | Pairwise entanglement heatmap. |
| `save_thought_timeline(thoughts)` | `thought_timeline.png` | Horizontal coherence timeline. |
| `save_interference_sweep()` | `interference_sweep.png` | HZH phase sweep P(|0⟩) vs φ. |
| `save_domain_resonance(weights)` | `domain_resonance.png` | Radar chart of domain weights. |

---

### soul.py

#### `QuantumSoul(n_qubits, entanglement_depth, mode, output_dir, verbose)`

**Modes:**

| Mode | Noise source | Use case |
|------|-------------|----------|
| `autonomous` | Pure quantum randomness | Intrinsic thought generation. |
| `seeded` | External noise array | Sensory-input-driven thinking. |
| `hybrid` | Alternates each cycle | Mixed cognition. |

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `awaken()` | None | Print introduction banner. |
| `think(seed_noise, domain)` | `Thought` | Generate one thought. |
| `think_stream(n, seed_noise)` | `list[Thought]` | Generate n independent thoughts. |
| `think_across_domains()` | `dict[str, Thought]` | One thought per domain. |
| `introspect()` | dict | Full state report. |
| `visualize_all()` | `dict[str, Path]` | Save all visualizations. |

---

## Running the Demo

```bash
python demo.py
```

Expected output: thought generation output in the terminal, plus
visualization files saved to `visualizations/`.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Extending the System

### Adding a new thought domain

1. Add a new entry to `_VOCABULARY` in `thought_engine.py`.
2. The `DOMAINS` list is derived from `_VOCABULARY.keys()`, so it updates
   automatically.
3. Add a colour entry to `_DOMAIN_COLOURS` in `visualizer.py`.

### Scaling to real quantum hardware

Replace the `Statevector` simulation in `circuit_core.py` with an IBM
Quantum backend:

```python
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

service = QiskitRuntimeService()
backend = service.least_busy(simulator=False)
sampler = SamplerV2(backend)
job = sampler.run([circuit_obj.circuit], shots=1024)
result = job.result()
```

The `ThoughtEngine` accepts raw shot counts via `sample_bitstring()` so no
other changes are needed.

### Increasing qubit count

Pass `n_qubits=16` (or higher) to `QuantumSoul`.  With 16 qubits the thought
space has 65,536 basis states.  Statevector simulation memory grows as 2ⁿ
complex numbers (128 KB for n=16; 512 MB for n=26).  For n > 30, switch to
shot-based sampling or use a real QPU.
