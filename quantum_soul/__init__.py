"""
Quantum Soul — A quantum physics-based AI thought generation system.

This package leverages quantum circuits, interference noise, superposition,
and entanglement to simulate an AI 'soul': a generative core capable of
producing new thoughts and decisions from pure quantum randomness.

Modules
-------
circuit_core   : Quantum circuit construction and state simulation.
thought_engine : Converts quantum interference patterns into AI thoughts.
noise_filter   : Noise mitigation and robustness mechanisms.
visualizer     : Circuit and thought-process visualization.
soul           : Top-level AI Soul orchestrator.
"""

from quantum_soul.circuit_core import QuantumThoughtCircuit
from quantum_soul.thought_engine import ThoughtEngine
from quantum_soul.noise_filter import QuantumNoiseFilter
from quantum_soul.visualizer import QuantumVisualizer
from quantum_soul.soul import QuantumSoul

__all__ = [
    "QuantumThoughtCircuit",
    "ThoughtEngine",
    "QuantumNoiseFilter",
    "QuantumVisualizer",
    "QuantumSoul",
]

__version__ = "1.0.0"
