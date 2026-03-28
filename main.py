"""
main.py — Entry point and demonstration for the Quantum Neural Brain.

Runs three demonstration scenarios:

1. **Noise-driven thought generation** — The brain receives only quantum noise
   as input and self-organises to generate structured output patterns.

2. **Sensory input processing** — Structured stimuli are fed into the input
   layer and the brain learns to classify them via STDP plasticity.

3. **Continuous real-time mode** — The brain runs in a live loop, printing
   compact metrics to the terminal.

Usage:
    python main.py                    # Run all demos
    python main.py --demo noise       # Noise-driven only
    python main.py --demo sensory     # Sensory processing only
    python main.py --demo realtime    # Real-time continuous mode
    python main.py --steps 200        # Custom step count
"""

import argparse
import sys
import time

from quantum_neural_brain.neural_network import QuantumNeuralBrain, BrainConfig
from quantum_neural_brain.neuroplasticity import LearningRule
from quantum_neural_brain.visualization import NeuralVisualizer


def demo_noise_driven(steps: int = 100) -> None:
    """
    Demo 1: Thought generation from pure quantum noise.

    No external stimulus — the brain fires spontaneously driven by
    quantum fluctuations and interference patterns, mimicking the
    brain's resting-state activity (default mode network analogy).
    """
    print("\n" + "=" * 60)
    print("  DEMO 1: Noise-Driven Thought Generation")
    print("  (Pure quantum interference — no external input)")
    print("=" * 60 + "\n")

    config = BrainConfig(
        input_size=8,
        hidden_sizes=[12, 12],
        output_size=4,
        enable_quantum_layer=True,
        quantum_noise_level=0.3,     # Higher noise → more spontaneous activity
        enable_plasticity=True,
        learning_rule=LearningRule.COMBINED,
    )
    brain = QuantumNeuralBrain(config)
    viz = NeuralVisualizer(brain)

    print(brain.summary())
    print("\nRunning noise-driven simulation...\n")

    results = brain.run(num_steps=steps, verbose=True)

    # Print final visualisation frame
    last_result = results[-1]
    print()
    print(viz.render(last_result))
    print()
    print(viz.performance_report())


def demo_sensory_processing(steps: int = 150) -> None:
    """
    Demo 2: Sensory input processing with STDP learning.

    Alternates between two input patterns and shows the brain learning
    to distinguish them through synaptic weight adaptation.
    """
    print("\n" + "=" * 60)
    print("  DEMO 2: Sensory Input Processing + STDP Learning")
    print("  (Structured stimuli drive adaptive weight changes)")
    print("=" * 60 + "\n")

    config = BrainConfig(
        input_size=8,
        hidden_sizes=[16, 8],
        output_size=4,
        enable_quantum_layer=True,
        quantum_noise_level=0.1,
        enable_plasticity=True,
        learning_rule=LearningRule.STDP,
    )
    brain = QuantumNeuralBrain(config)
    viz = NeuralVisualizer(brain)

    print(brain.summary())
    print("\nRunning sensory processing simulation...\n")

    # Two alternating stimulus patterns (excitatory current injections, pA)
    pattern_a = [20.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0, 0.0]  # odd neurons
    pattern_b = [0.0, 20.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0]  # even neurons

    stimuli_sequence = []
    for i in range(steps):
        stimuli_sequence.append(pattern_a if i % 20 < 10 else pattern_b)

    results = brain.run(
        num_steps=steps,
        input_stimuli_sequence=stimuli_sequence,
        verbose=True,
    )

    print()
    print(viz.render(results[-1]))
    print()
    print(viz.performance_report())


def demo_realtime(steps: int = 50) -> None:
    """
    Demo 3: Real-time continuous mode.

    Prints compact per-step metrics as the simulation runs.
    """
    print("\n" + "=" * 60)
    print("  DEMO 3: Real-Time Continuous Mode")
    print("  (Live per-step metrics)")
    print("=" * 60 + "\n")

    config = BrainConfig(
        input_size=6,
        hidden_sizes=[10],
        output_size=3,
        enable_quantum_layer=True,
        quantum_noise_level=0.2,
        enable_plasticity=True,
        learning_rule=LearningRule.HEBBIAN,
    )
    brain = QuantumNeuralBrain(config)
    viz = NeuralVisualizer(brain)

    print(brain.summary())
    print("\nReal-time output:\n")

    for i in range(steps):
        result = brain.step()
        print(viz.render_compact(result))
        time.sleep(0.02)  # 20 ms simulated timestep delay for readability

    print()
    print(viz.performance_report())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantum Neural Brain — Bio-inspired neural simulation"
    )
    parser.add_argument(
        "--demo",
        choices=["noise", "sensory", "realtime", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation steps per demo (default: 100)",
    )
    args = parser.parse_args()

    print()
    print("██████████████████████████████████████████████████████")
    print("██  Quantum Neural Brain  —  JessicAi Medusa Series  ██")
    print("██  Bio-inspired neuron/synapse simulation engine    ██")
    print("██████████████████████████████████████████████████████")

    if args.demo in ("noise", "all"):
        demo_noise_driven(args.steps)

    if args.demo in ("sensory", "all"):
        demo_sensory_processing(args.steps)

    if args.demo in ("realtime", "all"):
        demo_realtime(min(args.steps, 50))

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
