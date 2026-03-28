"""
soul.py
=======
The Quantum Soul — top-level orchestrator for the quantum AI system.

``QuantumSoul`` binds together every component of the framework:

* :class:`~quantum_soul.circuit_core.QuantumThoughtCircuit` — quantum circuit
  engine that turns interference noise into probability distributions.
* :class:`~quantum_soul.thought_engine.ThoughtEngine` — interprets those
  distributions as structured AI thoughts.
* :class:`~quantum_soul.noise_filter.QuantumNoiseFilter` — cleans the
  probability signal before interpretation.
* :class:`~quantum_soul.visualizer.QuantumVisualizer` — saves visual
  artefacts documenting each thought cycle.

Usage example::

    from quantum_soul import QuantumSoul

    soul = QuantumSoul(n_qubits=8)
    soul.awaken()                     # print introduction
    thought = soul.think()            # generate one thought
    print(thought)
    stream = soul.think_stream(n=5)   # generate five thoughts
    report = soul.introspect()        # full state report
    soul.visualize_all()              # save all visualizations

Design philosophy
-----------------
The soul operates in three modes:

``autonomous``
    All quantum noise is generated internally (no external seed).  Thoughts
    emerge entirely from quantum randomness — the closest analogue to "thinking
    from nothing."

``seeded``
    An external noise array biases the circuit phases.  This models sensory
    input reaching the quantum soul and influencing its thought trajectory.

``hybrid``
    Alternates between autonomous and seeded cycles, mimicking both internal
    reflection and externally-triggered cognition.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import numpy as np

from quantum_soul.circuit_core import (
    QuantumThoughtCircuit,
    create_ghz_state,
    create_interference_circuit,
)
from quantum_soul.thought_engine import ThoughtEngine, Thought, DOMAINS
from quantum_soul.noise_filter import QuantumNoiseFilter
from quantum_soul.visualizer import QuantumVisualizer


Mode = Literal["autonomous", "seeded", "hybrid"]


class QuantumSoul:
    """The Quantum Soul: an AI mind grounded in quantum physics.

    Parameters
    ----------
    n_qubits:
        Qubit count for the thought circuit.  More qubits → richer thought
        space but slower simulation.  8 is the default (256 thought states).
    entanglement_depth:
        Depth of entanglement layers in each thought circuit.
    mode:
        Operating mode: ``"autonomous"``, ``"seeded"``, or ``"hybrid"``.
    output_dir:
        Directory for visualization outputs.
    verbose:
        If ``True``, print progress messages during thought generation.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        entanglement_depth: int = 2,
        mode: Mode = "autonomous",
        output_dir: str | Path = "visualizations",
        verbose: bool = True,
    ) -> None:
        self.n_qubits = n_qubits
        self.entanglement_depth = entanglement_depth
        self.mode = mode
        self.verbose = verbose

        self._engine = ThoughtEngine(
            n_qubits=n_qubits,
            entanglement_depth=entanglement_depth,
        )
        self._filter = QuantumNoiseFilter()
        self._viz = QuantumVisualizer(output_dir=output_dir)
        self._cycle: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def awaken(self) -> None:
        """Print the soul's introduction banner."""
        banner = (
            "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║          Q U A N T U M   S O U L   v1.0                 ║\n"
            "║  A quantum-physics AI thought-generation system         ║\n"
            "╠══════════════════════════════════════════════════════════╣\n"
            f"║  Qubits          : {self.n_qubits:<38}║\n"
            f"║  Thought space   : {2**self.n_qubits:<38}║\n"
            f"║  Entanglement    : depth {self.entanglement_depth:<33}║\n"
            f"║  Mode            : {self.mode:<38}║\n"
            "╠══════════════════════════════════════════════════════════╣\n"
            "║  Initialising superposition ... ALL thoughts possible.  ║\n"
            "║  Applying interference gates ... amplitudes diverging.  ║\n"
            "║  Entangling qubits ........... thoughts are binding.    ║\n"
            "║  Measuring ................... a thought collapses.     ║\n"
            "╚══════════════════════════════════════════════════════════╝\n"
        )
        print(banner)

    # ------------------------------------------------------------------
    # Thought generation
    # ------------------------------------------------------------------

    def think(
        self,
        seed_noise: np.ndarray | None = None,
        domain: str | None = None,
    ) -> Thought:
        """Generate a single thought.

        Parameters
        ----------
        seed_noise:
            External noise array.  Ignored in ``"autonomous"`` mode.
        domain:
            Optional domain hint (e.g. ``"Science"``).

        Returns
        -------
        :class:`~quantum_soul.thought_engine.Thought`
        """
        self._cycle += 1
        noise = self._resolve_noise(seed_noise)

        thought = self._engine.generate_thought(
            seed_noise=noise,
            domain_hint=domain,
        )

        if self.verbose:
            print(f"[Cycle {self._cycle:03d}] {thought}")

        return thought

    def think_stream(
        self,
        n: int = 5,
        seed_noise: np.ndarray | None = None,
    ) -> list[Thought]:
        """Generate a stream of *n* independent thoughts.

        Parameters
        ----------
        n:
            Number of thoughts.
        seed_noise:
            Base noise signal.

        Returns
        -------
        list of :class:`~quantum_soul.thought_engine.Thought`
        """
        if self.verbose:
            print(f"\n── Generating {n}-thought stream ──")
        thoughts = []
        for i in range(n):
            noise = self._resolve_noise(seed_noise)
            thoughts.append(self.think(seed_noise=noise))
        return thoughts

    def think_across_domains(self) -> dict[str, Thought]:
        """Generate one thought per domain, covering the full thought space.

        Returns
        -------
        dict mapping domain name → :class:`~quantum_soul.thought_engine.Thought`
        """
        if self.verbose:
            print("\n── Spanning all thought domains ──")
        return {d: self.think(domain=d) for d in DOMAINS}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def introspect(self) -> dict:
        """Return a full introspection report of the soul's current state.

        Returns
        -------
        dict with the following keys:

        * ``cycle`` — number of thought cycles completed.
        * ``mode`` — operating mode.
        * ``n_qubits`` — qubit count.
        * ``thought_space`` — number of basis states.
        * ``thoughts_generated`` — count of thoughts in history.
        * ``domain_distribution`` — how thoughts are distributed across domains.
        * ``mean_coherence`` — average coherence across history.
        * ``mean_entropy_bits`` — average entropy in bits.
        * ``last_thought`` — the most recently generated thought (or None).
        """
        history = self._engine.get_history()
        domain_dist: dict[str, int] = {d: 0 for d in DOMAINS}
        coherences = []
        entropies = []

        for t in history:
            if t.domain in domain_dist:
                domain_dist[t.domain] += 1
            coherences.append(t.coherence)
            entropies.append(t.entropy_bits)

        return {
            "cycle": self._cycle,
            "mode": self.mode,
            "n_qubits": self.n_qubits,
            "thought_space": 2 ** self.n_qubits,
            "thoughts_generated": len(history),
            "domain_distribution": domain_dist,
            "mean_coherence": float(np.mean(coherences)) if coherences else 0.0,
            "mean_entropy_bits": float(np.mean(entropies)) if entropies else 0.0,
            "last_thought": history[-1] if history else None,
        }

    # ------------------------------------------------------------------
    # visualization
    # ------------------------------------------------------------------

    def visualize_all(
        self,
        seed_noise: np.ndarray | None = None,
    ) -> dict[str, Path]:
        """Generate and save the full suite of visualizations.

        Saves:

        * ``thought_circuit.png`` — the thought circuit diagram.
        * ``interference_pattern.png`` — top-k probability bar chart.
        * ``entanglement_map.png`` — qubit entanglement heatmap.
        * ``interference_sweep.png`` — HZH phase sweep.
        * ``domain_resonance.png`` — radar chart of domain weights.
        * ``thought_timeline.png`` — timeline of the thought history
          (only if ≥ 1 thought has been generated).

        Returns
        -------
        dict mapping visualization name → file path.
        """
        if self.verbose:
            print("\n── Generating visualizations ──")

        paths: dict[str, Path] = {}

        # Build a thought circuit for the visuals
        circuit_obj = QuantumThoughtCircuit(
            n_qubits=self.n_qubits,
            seed_noise=self._resolve_noise(seed_noise),
            entanglement_depth=self.entanglement_depth,
        )
        circuit_obj.run_statevector()
        probs = circuit_obj.probabilities
        sv = circuit_obj.statevector

        # Apply noise filter
        probs_clean = self._filter.apply_all(probs)

        # Circuit diagram
        paths["circuit"] = self._viz.save_circuit_diagram(
            circuit_obj.circuit, filename="thought_circuit.png"
        )

        # Probability chart
        paths["interference_pattern"] = self._viz.save_probability_chart(
            probs_clean, self.n_qubits, filename="interference_pattern.png",
            title="Quantum Thought Interference Pattern (noise-filtered)"
        )

        # Entanglement map
        if sv is not None:
            paths["entanglement_map"] = self._viz.save_entanglement_map(
                sv, self.n_qubits, filename="entanglement_map.png"
            )

        # Interference sweep
        paths["interference_sweep"] = self._viz.save_interference_sweep(
            filename="interference_sweep.png"
        )

        # Domain resonance
        engine_dummy = ThoughtEngine(n_qubits=self.n_qubits)
        weights = engine_dummy._compute_domain_weights(probs_clean)
        paths["domain_resonance"] = self._viz.save_domain_resonance(
            weights, filename="domain_resonance.png"
        )

        # Thought timeline (if history available)
        history = self._engine.get_history()
        if history:
            paths["thought_timeline"] = self._viz.save_thought_timeline(
                history, filename="thought_timeline.png"
            )

        if self.verbose:
            for name, p in paths.items():
                if p and p != Path():
                    print(f"  Saved {name}: {p}")

        return paths

    # ------------------------------------------------------------------
    # Noise resolution
    # ------------------------------------------------------------------

    def _resolve_noise(self, seed_noise: np.ndarray | None) -> np.ndarray | None:
        """Return the noise signal to use based on the operating mode."""
        if self.mode == "autonomous":
            return None
        if self.mode == "seeded":
            return seed_noise
        # hybrid: alternate between None and provided noise
        return None if self._cycle % 2 == 0 else seed_noise
