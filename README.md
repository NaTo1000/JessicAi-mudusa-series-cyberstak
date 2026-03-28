# JessicAi-mudusa-series-cyberstak

## Quantum Soul — A Quantum Physics-Based AI Thought Generation System

A full quantum-physics AI framework that leverages quantum circuits,
interference noise, superposition, and entanglement to produce emergent
thoughts and decisions — embodying the concept of an AI 'soul'.

---

## Features

| Feature | Description |
|---------|-------------|
| **Quantum Thought Circuits** | Hadamard + phase + CNOT circuits simulate a full thought cycle. |
| **Interference-Driven Thoughts** | Phase rotations from quantum/external noise create constructive and destructive interference that selects thoughts. |
| **Entanglement** | CNOT layers bind qubits together, modelling the associative nature of ideas. |
| **AI Soul (ThoughtEngine)** | Maps quantum probability distributions to structured thoughts across 6 knowledge domains. |
| **Noise Filtering** | Gaussian smoothing, parity filtering, and amplitude thresholding mitigate noise. |
| **visualizations** | Circuit diagrams, probability bar charts, entanglement heatmaps, interference sweeps, and thought timelines. |
| **Three Modes** | Autonomous (pure quantum), Seeded (noise-driven), Hybrid (alternating). |
| **Scalable** | Works on classical simulators today; drop-in ready for IBM Quantum hardware. |

---

## Quick Start

```bash
pip install -r requirements.txt
python demo.py
```

---

## Usage

```python
from quantum_soul import QuantumSoul

# Create a quantum soul with 8 qubits (256-state thought space)
soul = QuantumSoul(n_qubits=8, mode="autonomous")
soul.awaken()

# Generate a single thought from pure quantum randomness
thought = soul.think()
print(thought)

# Generate a stream of 5 thoughts
stream = soul.think_stream(n=5)

# Generate one thought per knowledge domain
domain_thoughts = soul.think_across_domains()

# Full introspection report
report = soul.introspect()

# Save all visualizations to visualizations/
soul.visualize_all()
```

---

## Repository Structure

```
quantum_soul/          Core package
  __init__.py
  circuit_core.py      Quantum circuit primitives
  thought_engine.py    AI thought generation from quantum states
  noise_filter.py      Noise mitigation and robustness
  visualizer.py        Circuit and process visualization
  soul.py              QuantumSoul orchestrator
tests/                 Unit test suite (pytest)
docs/
  theory.md            Theoretical foundations (quantum mechanics)
  implementation.md    Implementation reference guide
demo.py                Full demonstration script
requirements.txt       Python dependencies
```

---

## Quantum Physics Concepts Used

- **Superposition** (Hadamard gates) — all thoughts possible simultaneously
- **Interference** (Rz phase gates + H readout) — amplifies correct thoughts, suppresses noise
- **Entanglement** (CNOT gates) — binds related ideas non-locally
- **Born rule measurement** — irreducible quantum randomness collapses to a definite thought
- **Von Neumann entropy** — measures creative vs focused thinking
- **GHZ states** — maximally correlated multi-qubit thought registers
- **Bell pairs** — foundational 2-qubit entangled building blocks

---

## Documentation

- [Theoretical Foundations](docs/theory.md)
- [Implementation Guide](docs/implementation.md)

---

## Running Tests

```bash
pytest tests/ -v
```

---

## visualization Outputs

After running `demo.py` or calling `soul.visualize_all()`, the following
images are saved to `visualizations/`:

| File | Description |
|------|-------------|
| `thought_circuit.png` | Full quantum thought circuit diagram |
| `interference_pattern.png` | Top-16 basis-state probability bar chart |
| `entanglement_map.png` | Pairwise qubit entanglement heatmap |
| `interference_sweep.png` | P(|0⟩) vs phase angle (HZH circuit) |
| `domain_resonance.png` | Radar chart of domain probability weights |
| `thought_timeline.png` | Horizontal timeline of all generated thoughts |
