# Quantum Neural Brain — Architecture & Methodology

## Overview

The **Quantum Neural Brain** is a bio-inspired neural simulation engine that
models information processing in the style of biological neurons and synapses.
It extends the classical leaky integrate-and-fire (LIF) neuron model with
quantum-mechanical enhancements, distributed hardware simulation, and adaptive
synaptic plasticity.

The system is designed to run on a distributed cluster of Raspberry Pi CM4
compute modules backed by DRAM caches and NVMe persistent storage —
mirroring the way the biological brain distributes function across
specialised regions.

---

## Module Architecture

```
quantum_neural_brain/
├── __init__.py          — Public API surface
├── neuron.py            — Biological neuron model (LIF + quantum noise)
├── synapse.py           — Synaptic transmission (excitatory / inhibitory)
├── quantum_layer.py     — Quantum gates, superposition, interference
├── neuroplasticity.py   — Hebbian, STDP, and homeostatic learning
├── hardware_layer.py    — NVMe / DRAM / CM4 compute node abstraction
├── neural_network.py    — Full layered brain assembly
└── visualization.py     — Terminal-based visualiser and performance metrics

tests/
├── test_neuron.py
├── test_synapse.py
├── test_quantum_layer.py
├── test_neuroplasticity.py
├── test_hardware_layer.py
└── test_neural_network.py

main.py                  — Demo entry point (three scenarios)
```

---

## 1. Neuronal Architecture (`neuron.py`)

Each `Neuron` implements the **leaky integrate-and-fire** (LIF) model:

```
dV/dt = (1/τ) * [ g_leak * (V_rest − V) + I_syn + I_quantum_noise ]
```

| Parameter | Biological Equivalent | Default |
|---|---|---|
| `resting_potential` | Neuronal resting state | −70 mV |
| `threshold_potential` | Action potential threshold | −55 mV |
| `peak_potential` | Spike peak | +40 mV |
| `reset_potential` | Post-spike hyperpolarisation | −75 mV |
| `refractory_period_ms` | Absolute refractory period | 2 ms |
| `membrane_time_constant_ms` | RC time constant (τ) | 20 ms |
| `quantum_noise_amplitude` | Ion-channel shot noise | 0.5 mV |

**Neuron States:** `RESTING → INTEGRATING → FIRING → REFRACTORY → RESTING`

Quantum noise is modelled as zero-mean Gaussian perturbations to the membrane
potential, representing stochastic ion-channel gating (shot noise).

---

## 2. Synaptic Transmission (`synapse.py`)

Synapses implement a **conductance-based** model:

```
I_psc = g(t) * (E_rev − V_post)
```

where `g(t)` decays exponentially after each vesicle release event.

**Synapse Types:**
- **Excitatory (AMPA/NMDA-like):** `E_rev ≈ 0 mV` → depolarises post-neuron
- **Inhibitory (GABA-like):** `E_rev ≈ −70 mV` → hyperpolarises post-neuron

**Short-Term Plasticity (Tsodyks-Markram model):**
- Synaptic resources (`x`) deplete with each release and recover exponentially
- Use fraction (`u`) facilitates (Ca²⁺-like) then decays to baseline

**Quantum Release Probability:**
Each vesicle release event is gated by a Bernoulli trial with probability
`P_r`, simulating the probabilistic quantum nature of neurotransmitter release.

---

## 3. Quantum Layer (`quantum_layer.py`)

The quantum layer wraps classical neurons with a **qubit-inspired** state
representation, where each neuron occupies a two-dimensional Hilbert space:

- **|0⟩** — Resting state
- **|1⟩** — Firing state

### Quantum Gates Applied Each Timestep

| Gate | Effect |
|---|---|
| **Hadamard (H)** | Creates equal superposition of resting / firing |
| **Phase Rotation R(θ)** | Encodes membrane potential as quantum phase |
| **Controlled-NOT (CX)** | Creates entanglement between neuron pairs |

### Interference

The external signal (sensory input or noise) is encoded as a complex phase
and interfered with the current quantum state:

```
P_fire = |ψ_1 + 0.3 * e^(iφ_signal)|² / Z
```

Constructive interference amplifies near-threshold signals; destructive
interference suppresses them — creating rich probabilistic dynamics.

### Entanglement

Selected neuron pairs are connected via CX gates. When the control neuron
collapses into `|1⟩`, the target qubit flips — causing correlated firing
with a 60% co-fire probability, analogous to synchronised neural oscillations.

---

## 4. Neuroplasticity (`neuroplasticity.py`)

### Hebbian Learning
```
Δw = η * r_pre * r_post − λ * w
```
Weights grow when both neurons are active simultaneously and decay slowly
to prevent saturation.

### Spike-Timing-Dependent Plasticity (STDP)
```
Δw = A+ * exp(−Δt / τ+)   if t_post > t_pre  (LTP — causal)
Δw = −A- * exp(Δt / τ-)   if t_post < t_pre  (LTD — anti-causal)
```
The sign and magnitude of weight change depends on the temporal ordering
of pre- and post-synaptic spikes — matching experimental biology.

### Homeostatic Scaling
```
w_new = w * (1 + η_h * (r_target − r_mean))
```
Globally scales all weights to keep the network's average firing rate near
a target, preventing runaway excitation or silence.

---

## 5. Hardware Integration (`hardware_layer.py`)

| Hardware Component | Neural Role | Simulated Spec |
|---|---|---|
| Raspberry Pi CM4 | Neuron cluster processor | 4-core @ 1800 MHz, 8 GB RAM |
| DRAM (DDR4-3200) | Working memory / short-term state | 64 GB, 25.6 GB/s bandwidth |
| NVMe SSD (PCIe Gen 4) | Long-term weight persistence | 7 GB/s read, 6.5 GB/s write |

**Default Topology:**
```
CM4_0 ←→ CM4_1 ←→ CM4_2 ←→ CM4_3
  ↕         ↕         ↕         ↕
              DRAM_0
                 ↕
              NVMe_0
```

The 4 CM4 nodes form a ring and all connect to the shared DRAM cache, which
persists long-term synaptic weights to NVMe.

Thermal throttling is simulated: when node temperature exceeds 95% of the
85°C TDP limit, the effective clock speed drops to 60%, slowing integration.

---

## 6. Layered Brain Architecture (`neural_network.py`)

```
Input Layer  (sensory neurons, lower threshold)
     ↓ feed-forward synapses
Hidden Layer 0  (association cortex analogy)
     ↓ feed-forward synapses
Hidden Layer 1  (deep processing analogy)
     ↓ feed-forward synapses
Output Layer  (decision / motor neurons)
     ↑ feedback synapses (weight × 0.3)
```

**Decision Making:**
Winner-take-all across output neuron activations, with a minimum activation
threshold of 0.3 to prevent noise-driven decisions.

---

## 7. Visualisation (`visualization.py`)

Terminal-native, zero-dependency rendering:

- **Firing map** — `●` fired, `R` refractory, `○` integrating, `·` resting
- **Output bar chart** — Unicode block characters for activation levels
- **Weight histogram** — 10-bucket ASCII histogram
- **Sparkline** — Mean weight trend over training history
- **Hardware panel** — Per-node temperature, throttle status, I/O counters
- **Performance report** — Elapsed time, spike rate, per-output distribution

---

## Running the System

```bash
# All three demos
python main.py

# Noise-driven thought generation only
python main.py --demo noise --steps 200

# Sensory input + STDP learning
python main.py --demo sensory --steps 300

# Real-time compact mode
python main.py --demo realtime --steps 100

# Run the test suite
python -m pytest tests/ -v
```

---

## Biological Decision-Making Workflow Example

```
Sensory stimulus arrives at input layer
    → Input neurons receive excitatory current injection
    → Membrane potentials rise toward threshold
    → Some neurons fire (action potential)
    ↓
Spikes propagate through excitatory/inhibitory synapses
    → Conductance-based PSCs perturb hidden layer potentials
    → Quantum interference modulates firing probability
    → Entangled neuron pairs show correlated activity
    ↓
Hidden layers integrate across ~50 ms window
    → STDP strengthens causal pathways
    → Homeostasis prevents saturation
    ↓
Output neurons cross threshold
    → Winner-take-all decision emitted
    → Decision fed back to last hidden layer (thalamo-cortical loop)
    ↓
Synaptic weights persist to NVMe for long-term memory
```

---

## Technology Stack

| Technology | Role |
|---|---|
| Python 3.9+ | Core simulation runtime |
| Standard library only (`math`, `random`, `cmath`, `time`, `dataclasses`) | No heavy dependencies |
| `pytest` | Test framework |
| Raspberry Pi CM4 (hardware target) | Distributed compute |
| NVMe SSD | Weight persistence |
| DDR4 DRAM | State caching |
