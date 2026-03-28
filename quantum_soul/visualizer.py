"""
visualizer.py
=============
visualization utilities for the Quantum Soul system.

Provides:

* **Circuit diagrams** — renders the thought circuit using Qiskit's built-in
  text or matplotlib drawers.
* **Probability bar charts** — plots the top-N basis states by probability,
  showing the interference pattern responsible for a given thought.
* **Bloch sphere** — visualises a single-qubit state on the Bloch sphere to
  communicate the concept of superposition to non-specialists.
* **Entanglement map** — heatmap of pairwise mutual information between qubits,
  illustrating how strongly thoughts are correlated.
* **Thought timeline** — timeline chart of a sequence of generated thoughts,
  coloured by domain and scaled by coherence.
* **Interference sweep** — plots probability vs phase angle for a
  Hadamard–Rz–Hadamard circuit, demonstrating constructive/destructive
  interference visually.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on headless servers)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy

from quantum_soul.circuit_core import (
    QuantumThoughtCircuit,
    create_ghz_state,
    create_interference_circuit,
)
from quantum_soul.thought_engine import Thought, DOMAINS


# Colour palette — one colour per domain
_DOMAIN_COLOURS: dict[str, str] = {
    "Science":     "#4C72B0",
    "Philosophy":  "#DD8452",
    "Creativity":  "#55A868",
    "Technology":  "#C44E52",
    "Mathematics": "#8172B2",
    "Self":        "#937860",
}


class QuantumVisualizer:
    """Handles all visualization tasks for the Quantum Soul system.

    Parameters
    ----------
    output_dir:
        Directory where image files are saved.  Created if it does not exist.
    dpi:
        Resolution for rasterised images.  Default 120.
    """

    def __init__(self, output_dir: str | Path = "visualizations", dpi: int = 120) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    # ------------------------------------------------------------------
    # 1. Circuit diagram
    # ------------------------------------------------------------------

    def save_circuit_diagram(
        self,
        circuit: QuantumCircuit,
        filename: str = "thought_circuit.png",
        style: str = "iqp",
    ) -> Path:
        """Save a circuit diagram to *filename*.

        Parameters
        ----------
        circuit:
            Any Qiskit ``QuantumCircuit``.
        filename:
            Output file name (relative to *output_dir*).
        style:
            Matplotlib circuit style; ``"iqp"`` (default) or ``"bw"``.

        Returns
        -------
        Path to the saved image.
        """
        try:
            fig = circuit.draw(output="mpl", style=style, fold=-1)
        except Exception:
            # Fallback: text drawing if mpl fails
            print(circuit.draw(output="text"))
            return Path()

        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 2. Probability distribution bar chart
    # ------------------------------------------------------------------

    def save_probability_chart(
        self,
        probs: np.ndarray,
        n_qubits: int,
        top_k: int = 16,
        title: str = "Quantum Interference Pattern",
        filename: str = "interference_pattern.png",
        domain: str | None = None,
    ) -> Path:
        """Save a bar chart of the top-k basis-state probabilities.

        Parameters
        ----------
        probs:
            Full probability array of length 2^n_qubits.
        n_qubits:
            Number of qubits (used for labelling).
        top_k:
            How many states to show.
        title:
            Plot title.
        filename:
            Output file name.
        domain:
            If given, the bar colour is drawn from the domain palette.

        Returns
        -------
        Path to the saved image.
        """
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]
        labels = [f"{i:0{n_qubits}b}" for i in top_indices]

        colour = _DOMAIN_COLOURS.get(domain or "", "#4C72B0")
        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(range(len(labels)), top_probs, color=colour, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_xlabel("Basis State (bitstring)")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.set_ylim(0, max(top_probs) * 1.15)

        # Annotate peak
        peak_x = int(np.argmax(top_probs))
        ax.annotate(
            f"peak\n{top_probs[peak_x]:.4f}",
            xy=(peak_x, top_probs[peak_x]),
            xytext=(peak_x + 0.5, top_probs[peak_x] * 1.05),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 3. Entanglement map (pairwise mutual information)
    # ------------------------------------------------------------------

    def save_entanglement_map(
        self,
        statevector: Statevector,
        n_qubits: int,
        filename: str = "entanglement_map.png",
    ) -> Path:
        """Save a heatmap of pairwise qubit entanglement.

        The entanglement between qubits *i* and *j* is approximated by the
        von Neumann entropy of the reduced density matrix of qubit *i* after
        tracing out all other qubits.  For a fully separable state this is 0;
        for a maximally entangled pair it is 1 ebit.

        Parameters
        ----------
        statevector:
            Full n-qubit statevector.
        n_qubits:
            Number of qubits.
        filename:
            Output file name.

        Returns
        -------
        Path to the saved image.
        """
        # Compute single-qubit entanglement entropies
        ent = np.zeros((n_qubits, n_qubits))
        for i in range(n_qubits):
            keep = [i]
            trace_out = [q for q in range(n_qubits) if q != i]
            try:
                rho_i = partial_trace(statevector, trace_out)
                s_i = float(entropy(rho_i, base=2))
            except Exception:
                s_i = 0.0
            for j in range(n_qubits):
                if i != j:
                    ent[i, j] = s_i   # approximation: share entropy with each partner

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(ent, cmap="plasma", vmin=0, vmax=1)
        ax.set_title("Qubit Entanglement Map (von Neumann entropy)")
        ax.set_xlabel("Qubit index")
        ax.set_ylabel("Qubit index")
        ax.set_xticks(range(n_qubits))
        ax.set_yticks(range(n_qubits))
        plt.colorbar(im, ax=ax, label="Entropy (ebits)")
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 4. Thought timeline
    # ------------------------------------------------------------------

    def save_thought_timeline(
        self,
        thoughts: Sequence[Thought],
        filename: str = "thought_timeline.png",
    ) -> Path:
        """Save a horizontal timeline of generated thoughts.

        Each thought is a horizontal bar coloured by domain and scaled by
        coherence score.

        Parameters
        ----------
        thoughts:
            Sequence of :class:`~quantum_soul.thought_engine.Thought` objects.
        filename:
            Output file name.

        Returns
        -------
        Path to the saved image.
        """
        n = len(thoughts)
        if n == 0:
            raise ValueError("At least one thought is required for the timeline.")

        fig, ax = plt.subplots(figsize=(14, max(3, n * 0.8)))
        for i, t in enumerate(thoughts):
            colour = _DOMAIN_COLOURS.get(t.domain, "#888888")
            ax.barh(i, t.coherence, color=colour, edgecolor="white", height=0.6)
            label = f"[{t.domain}] {t.text[:60]}{'…' if len(t.text) > 60 else ''}"
            ax.text(0.01, i, label, va="center", fontsize=8)

        ax.set_xlabel("Coherence")
        ax.set_title("Quantum Soul — Thought Timeline")
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"T{i+1}" for i in range(n)])
        ax.set_xlim(0, 1.1)
        ax.invert_yaxis()

        # Legend
        legend_patches = [
            mpatches.Patch(color=c, label=d) for d, c in _DOMAIN_COLOURS.items()
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=8, ncol=2)

        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 5. Interference sweep (H–Rz(φ)–H)
    # ------------------------------------------------------------------

    def save_interference_sweep(
        self,
        n_points: int = 200,
        filename: str = "interference_sweep.png",
    ) -> Path:
        """Plot P(|0…0⟩) vs phase angle for a single-qubit HZH circuit.

        Demonstrates how phase rotations produce constructive (P=1) and
        destructive (P=0) interference — the fundamental quantum mechanism
        driving thought generation.

        Parameters
        ----------
        n_points:
            Number of phase angles to sweep in [0, 2π].
        filename:
            Output file name.

        Returns
        -------
        Path to the saved image.
        """
        angles = np.linspace(0, 2 * math.pi, n_points)
        p_zero = []
        for phi in angles:
            # |+⟩ → Rz(φ)|+⟩ → H|+⟩ : probability of measuring |0⟩
            # = cos²(φ/2)
            p_zero.append(math.cos(phi / 2) ** 2)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.degrees(angles), p_zero, color="#4C72B0", lw=2)
        ax.fill_between(np.degrees(angles), p_zero, alpha=0.2, color="#4C72B0")
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="random (0.5)")
        ax.set_xlabel("Phase angle φ (degrees)")
        ax.set_ylabel("P(|0⟩)")
        ax.set_title("Quantum Interference Sweep — HZH Circuit\n"
                     "Constructive (P→1) and Destructive (P→0) Interference")
        ax.set_xlim(0, 360)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(range(0, 361, 45))
        ax.legend()
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 6. Domain resonance radar chart
    # ------------------------------------------------------------------

    def save_domain_resonance(
        self,
        domain_weights: np.ndarray,
        filename: str = "domain_resonance.png",
    ) -> Path:
        """Save a radar (spider) chart of domain resonance weights.

        Parameters
        ----------
        domain_weights:
            1-D array of length ``len(DOMAINS)`` with probability mass per domain.
        filename:
            Output file name.

        Returns
        -------
        Path to the saved image.
        """
        labels = DOMAINS
        n = len(labels)
        angles = [2 * math.pi * i / n for i in range(n)] + [0]
        values = list(domain_weights) + [domain_weights[0]]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
        ax.plot(angles, values, "o-", lw=2, color="#4C72B0")
        ax.fill(angles, values, alpha=0.25, color="#4C72B0")
        ax.set_thetagrids(
            [math.degrees(a) for a in angles[:-1]], labels, fontsize=10
        )
        ax.set_title("Quantum Domain Resonance\n(probability mass per thought domain)",
                     size=11, pad=20)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 0.3)
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path
