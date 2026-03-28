# JessicAi-mudusa-series-cyberstak

## Quantum Neural Brain — Bio-Inspired Neural Simulation Engine

A fully operational **Quantum Neural Brain** that operates like biological
neurons and synapses, built on the existing Quantum Quad-Brain infrastructure
with NVMe, DRAM, and Raspberry Pi CM4 compute modules as the hardware substrate.

---

### Features

| Feature | Implementation |
|---|---|
| Leaky integrate-and-fire neurons | `quantum_neural_brain/neuron.py` |
| Excitatory & inhibitory synapses | `quantum_neural_brain/synapse.py` |
| Quantum superposition / interference / entanglement | `quantum_neural_brain/quantum_layer.py` |
| Hebbian + STDP + homeostatic plasticity | `quantum_neural_brain/neuroplasticity.py` |
| NVMe / DRAM / CM4 hardware cluster | `quantum_neural_brain/hardware_layer.py` |
| Layered brain (input → hidden → output) | `quantum_neural_brain/neural_network.py` |
| Terminal visualiser & performance metrics | `quantum_neural_brain/visualization.py` |

---

### Quick Start

```bash
# Run all demos
python main.py

# Noise-driven thought generation
python main.py --demo noise --steps 200

# Sensory input + STDP learning
python main.py --demo sensory --steps 300

# Real-time compact output
python main.py --demo realtime

# Run the test suite
python -m pytest tests/ -v
```

---

### Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full technical
description including neural models, quantum gate operations, plasticity rules,
hardware topology, and decision-making workflow.

```
Input Layer  (sensory neurons)
     ↓ excitatory/inhibitory synapses
Hidden Layers (quantum-enhanced processing)
     ↓ feed-forward + feedback connections
Output Layer  (decision / motor neurons)
```

Hardware topology (default array):

```
CM4_0 ←→ CM4_1 ←→ CM4_2 ←→ CM4_3   (neuron clusters)
  ↕   ×4 connections to DRAM         (working memory)
DRAM_0 → NVMe_0                       (long-term weight persistence)
```
