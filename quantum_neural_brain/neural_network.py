"""
neural_network.py — Quantum Neural Brain: layered network architecture.

Assembles neurons, synapses, quantum layer, neuroplasticity, and hardware
into a complete bio-inspired brain:

  Input Layer  →  Hidden Layer(s)  →  Output Layer
       ↑                ↓
   Sensory input    Decision output

The network can operate in two modes:
- `step()`: advance one discrete timestep (suitable for simulations)
- `run()`: continuous loop until a stop condition

Architecture inspired by:
- Cortical columns (layered processing)
- Thalamo-cortical loops (feedback connections)
- Basal ganglia (decision gating)
"""

import time
import random
import math
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from .neuron import Neuron, NeuronConfig, NeuronState
from .synapse import Synapse, SynapseType, SynapseConfig
from .quantum_layer import QuantumLayer
from .neuroplasticity import NeuroplasticityEngine, PlasticityConfig, LearningRule
from .hardware_layer import HardwareLayer, HardwareNode, HardwareSpec, NodeType


@dataclass
class BrainLayer:
    """A single processing layer within the Quantum Neural Brain."""
    name: str
    neurons: List[Neuron] = field(default_factory=list)
    is_input: bool = False
    is_output: bool = False

    @property
    def size(self) -> int:
        return len(self.neurons)

    def fired_neurons(self) -> List[Neuron]:
        return [n for n in self.neurons if n.total_spikes > 0]

    def __repr__(self) -> str:
        return f"BrainLayer(name={self.name!r}, neurons={self.size})"


@dataclass
class BrainConfig:
    """Configuration for the Quantum Neural Brain network."""
    # Layer sizes
    input_size: int = 8
    hidden_sizes: List[int] = field(default_factory=lambda: [16, 16])
    output_size: int = 4

    # Connectivity
    excitatory_ratio: float = 0.8       # Fraction of synapses that are excitatory
    connection_probability: float = 0.4  # Probability of forming a connection
    feedback_connections: bool = True    # Enable output→hidden feedback

    # Quantum
    enable_quantum_layer: bool = True
    entanglement_fraction: float = 0.1  # Fraction of neuron pairs that are entangled
    quantum_noise_level: float = 0.15   # Interference noise amplitude

    # Plasticity
    enable_plasticity: bool = True
    learning_rule: LearningRule = LearningRule.COMBINED

    # Hardware
    num_cm4_nodes: int = 4

    # Timing
    dt_ms: float = 1.0                  # Simulation timestep
    steps_per_decision: int = 50        # Steps before reading output


class QuantumNeuralBrain:
    """
    Complete bio-inspired Quantum Neural Brain.

    Integrates:
    - Layered neuron architecture (input / hidden / output)
    - Weighted synaptic connections (excitatory & inhibitory)
    - Quantum processing layer (superposition, interference, entanglement)
    - Neuroplasticity engine (Hebbian + STDP + homeostatic)
    - Distributed hardware simulation (CM4, DRAM, NVMe)
    """

    def __init__(self, config: Optional[BrainConfig] = None):
        self.config = config or BrainConfig()
        self._step_count: int = 0

        # Build the brain
        self.layers: List[BrainLayer] = []
        self.synapses: List[Synapse] = []
        self.hardware = HardwareLayer()

        self._build_layers()
        self._build_synapses()
        self._build_quantum_layer()
        self._build_plasticity_engine()
        self._build_hardware()

        # Performance metrics
        self.metrics: Dict[str, Any] = {
            "total_steps": 0,
            "total_spikes": 0,
            "decisions_made": 0,
            "mean_weight_history": [],
            "output_activation_history": [],
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, input_stimuli: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Advance the brain by one timestep (dt_ms).

        Args:
            input_stimuli: Optional list of current values (pA) for input neurons.
                           Length must match the input layer size.

        Returns:
            A dictionary with firing state, output activations, and metrics.
        """
        dt = self.config.dt_ms

        # 1. Inject sensory input into the input layer
        if input_stimuli is not None:
            self._inject_input(input_stimuli)
        else:
            self._inject_noise_input()

        # 2. Step all synapses (propagate spikes through the network)
        for synapse in self.synapses:
            synapse.step(dt)

        # 3. Step all neurons (integrate and fire)
        fired = []
        for layer in self.layers:
            for neuron in layer.neurons:
                did_fire = neuron.step(dt)
                if did_fire:
                    fired.append(neuron.neuron_id)

        # 4. Quantum layer processing
        quantum_fired = []
        if self.config.enable_quantum_layer and self.quantum_layer is not None:
            quantum_fired = self.quantum_layer.step(dt, self.config.quantum_noise_level)

        # 5. Apply neuroplasticity
        if self.config.enable_plasticity:
            self.plasticity_engine.update(dt)

        # 6. Read output layer activations
        output_activations = self._read_output()

        # 7. Update metrics
        self._step_count += 1
        spike_count = len(fired) + len(quantum_fired)
        self.metrics["total_steps"] = self._step_count
        self.metrics["total_spikes"] += spike_count
        self.metrics["output_activation_history"].append(output_activations)
        if len(self.metrics["output_activation_history"]) > 200:
            self.metrics["output_activation_history"] = self.metrics["output_activation_history"][-200:]

        return {
            "step": self._step_count,
            "fired_neurons": fired + quantum_fired,
            "spike_count": spike_count,
            "output_activations": output_activations,
            "decision": self._make_decision(output_activations),
            "plasticity_mean_weight": self.plasticity_engine.mean_weight,
        }

    def run(
        self,
        num_steps: int = 100,
        input_stimuli_sequence: Optional[List[Optional[List[float]]]] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run the brain for num_steps timesteps.

        Args:
            num_steps: Number of simulation steps.
            input_stimuli_sequence: List of input vectors, one per step.
                                    Use None entries for noise-driven steps.
            verbose: If True, print progress every 10 steps.

        Returns:
            List of step result dictionaries.
        """
        results = []
        for i in range(num_steps):
            stimuli = None
            if input_stimuli_sequence is not None and i < len(input_stimuli_sequence):
                stimuli = input_stimuli_sequence[i]
            result = self.step(stimuli)
            results.append(result)
            if verbose and i % 10 == 0:
                print(
                    f"  Step {i:4d}/{num_steps} | "
                    f"Spikes: {result['spike_count']:3d} | "
                    f"Decision: {result['decision']} | "
                    f"Avg weight: {result['plasticity_mean_weight']:.4f}"
                )
        self.metrics["decisions_made"] += num_steps
        return results

    def get_layer(self, name: str) -> Optional[BrainLayer]:
        """Retrieve a layer by name."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def summary(self) -> str:
        """Return a human-readable network summary."""
        lines = ["=" * 60, "Quantum Neural Brain — Network Summary", "=" * 60]
        lines.append(f"  Total neurons  : {sum(l.size for l in self.layers)}")
        lines.append(f"  Total synapses : {len(self.synapses)}")
        lines.append(f"  Quantum layer  : {self.quantum_layer}")
        lines.append(f"  Plasticity     : {self.plasticity_engine}")
        lines.append(f"  Hardware       : {self.hardware}")
        lines.append("")
        for layer in self.layers:
            tag = ""
            if layer.is_input:
                tag = " [INPUT]"
            elif layer.is_output:
                tag = " [OUTPUT]"
            lines.append(f"  Layer '{layer.name}'{tag}: {layer.size} neurons")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_layers(self) -> None:
        """Construct all neuron layers."""
        cfg = self.config

        # Input layer
        input_layer = BrainLayer(name="input", is_input=True)
        for i in range(cfg.input_size):
            neuron = Neuron(
                neuron_id=f"in_{i}",
                layer_name="input",
                config=NeuronConfig(threshold_potential=-50.0),  # Lower threshold for sensory input
            )
            input_layer.neurons.append(neuron)
        self.layers.append(input_layer)

        # Hidden layers
        for h_idx, h_size in enumerate(cfg.hidden_sizes):
            hidden_layer = BrainLayer(name=f"hidden_{h_idx}")
            for i in range(h_size):
                neuron = Neuron(
                    neuron_id=f"h{h_idx}_{i}",
                    layer_name=f"hidden_{h_idx}",
                )
                hidden_layer.neurons.append(neuron)
            self.layers.append(hidden_layer)

        # Output layer
        output_layer = BrainLayer(name="output", is_output=True)
        for i in range(cfg.output_size):
            neuron = Neuron(
                neuron_id=f"out_{i}",
                layer_name="output",
                config=NeuronConfig(threshold_potential=-52.0),
            )
            output_layer.neurons.append(neuron)
        self.layers.append(output_layer)

    def _build_synapses(self) -> None:
        """
        Connect layers with weighted synapses (feed-forward + optional feedback).

        Each neuron in layer L can connect to neurons in layer L+1 with
        probability p = connection_probability.
        """
        for layer_idx in range(len(self.layers) - 1):
            pre_layer = self.layers[layer_idx]
            post_layer = self.layers[layer_idx + 1]
            self._connect_layers(pre_layer, post_layer)

        # Optional feedback: output → last hidden layer
        if self.config.feedback_connections and len(self.layers) >= 3:
            output_layer = self.layers[-1]
            last_hidden = self.layers[-2]
            self._connect_layers(output_layer, last_hidden, weight_scale=0.3)

    def _connect_layers(
        self,
        pre_layer: BrainLayer,
        post_layer: BrainLayer,
        weight_scale: float = 1.0,
    ) -> None:
        """Create synaptic connections between two layers."""
        syn_count = 0
        for pre_n in pre_layer.neurons:
            for post_n in post_layer.neurons:
                if random.random() > self.config.connection_probability:
                    continue

                syn_type = (
                    SynapseType.EXCITATORY
                    if random.random() < self.config.excitatory_ratio
                    else SynapseType.INHIBITORY
                )
                weight = random.gauss(1.0, 0.3) * weight_scale
                weight = max(0.05, weight)

                syn_cfg = SynapseConfig(
                    weight=weight,
                    delay_ms=random.uniform(0.5, 3.0),
                    release_probability=random.uniform(0.6, 0.95),
                )
                synapse = Synapse(
                    synapse_id=f"syn_{syn_count}_{pre_n.neuron_id}_{post_n.neuron_id}",
                    pre_neuron=pre_n,
                    post_neuron=post_n,
                    synapse_type=syn_type,
                    config=syn_cfg,
                )
                self.synapses.append(synapse)
                syn_count += 1

    def _build_quantum_layer(self) -> None:
        """Set up the quantum processing layer over all neurons."""
        if not self.config.enable_quantum_layer:
            self.quantum_layer = None
            return

        all_neurons = []
        for layer in self.layers:
            all_neurons.extend(layer.neurons)

        # Create entangled pairs among hidden neurons
        hidden_neurons = [n for n in all_neurons if n.layer_name.startswith("hidden")]
        num_pairs = max(1, int(len(hidden_neurons) * self.config.entanglement_fraction))
        indices = list(range(len(all_neurons)))
        entanglement_pairs = []
        for _ in range(num_pairs):
            i, j = random.sample(range(len(hidden_neurons)), 2)
            # Map hidden neuron indices to global neuron indices
            global_i = all_neurons.index(hidden_neurons[i])
            global_j = all_neurons.index(hidden_neurons[j])
            entanglement_pairs.append((global_i, global_j))

        self.quantum_layer = QuantumLayer(
            neurons=all_neurons,
            entanglement_pairs=entanglement_pairs,
        )

    def _build_plasticity_engine(self) -> None:
        """Attach the neuroplasticity engine."""
        plasticity_cfg = PlasticityConfig(learning_rule=self.config.learning_rule)
        self.plasticity_engine = NeuroplasticityEngine(
            synapses=self.synapses,
            config=plasticity_cfg,
        )

    def _build_hardware(self) -> None:
        """Distribute neurons across the hardware node array."""
        self.hardware.create_default_array(self.config.num_cm4_nodes)

        all_neurons = []
        for layer in self.layers:
            all_neurons.extend(layer.neurons)

        # Distribute neurons round-robin across CM4 nodes
        cm4_nodes = [
            n for n in self.hardware.nodes.values()
            if n.spec.node_type == NodeType.CM4_COMPUTE
        ]
        if cm4_nodes:
            for idx, neuron in enumerate(all_neurons):
                cm4_nodes[idx % len(cm4_nodes)].neurons.append(neuron)

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    def _inject_input(self, stimuli: List[float]) -> None:
        """Drive input neurons with external current."""
        input_layer = self.get_layer("input")
        if input_layer is None:
            return
        for i, neuron in enumerate(input_layer.neurons):
            if i < len(stimuli):
                neuron.receive_input(float(stimuli[i]))

    def _inject_noise_input(self) -> None:
        """Drive input layer with random noise (thought generation from noise)."""
        input_layer = self.get_layer("input")
        if input_layer is None:
            return
        for neuron in input_layer.neurons:
            noise_current = random.gauss(0.0, 5.0)
            neuron.receive_input(noise_current)

    def _read_output(self) -> List[float]:
        """
        Read activation level of each output neuron.

        Returns a normalised list in [0, 1] based on membrane potential.
        """
        output_layer = self.get_layer("output")
        if output_layer is None:
            return []
        activations = []
        for neuron in output_layer.neurons:
            vm = neuron.membrane_potential
            resting = neuron.config.resting_potential
            threshold = neuron.config.threshold_potential
            span = threshold - resting
            activation = (vm - resting) / span if span > 0 else 0.0
            activation = max(0.0, min(1.0, activation))
            activations.append(activation)
        return activations

    def _make_decision(self, output_activations: List[float]) -> Optional[int]:
        """
        Winner-take-all decision: return the index of the most active output neuron.
        Returns None if all outputs are below a 0.3 activation threshold.
        """
        if not output_activations:
            return None
        max_val = max(output_activations)
        if max_val < 0.3:
            return None
        return output_activations.index(max_val)
