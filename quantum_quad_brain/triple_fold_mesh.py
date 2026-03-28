"""
triple_fold_mesh.py
-------------------
Triple-folding fabric mesh for distributed quantum processing layers.

Three independent mesh layers are interleaved through fold operations that
exchange information between them.  Each fold "stitches" two adjacent layers
together, enabling cross-layer quantum coherence and distributing computational
load across the full fabric.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

import numpy as np


class FoldDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    TRANSVERSE = "transverse"


class MeshLayer:
    """
    A single 2D quantum mesh layer modelled as a grid of complex amplitude cells.
    """

    def __init__(self, layer_id: int, rows: int = 8, cols: int = 8) -> None:
        self.layer_id = layer_id
        self.rows = rows
        self.cols = cols
        rng = np.random.default_rng(layer_id * 7)
        raw = rng.standard_normal((rows, cols)) + 1j * rng.standard_normal((rows, cols))
        self.cells: np.ndarray = raw / np.linalg.norm(raw)

    # ------------------------------------------------------------------
    # Mesh operations
    # ------------------------------------------------------------------

    def apply_phase_gate(self, phase: float) -> None:
        """Apply a uniform phase rotation to all cells."""
        self.cells *= np.exp(1j * phase)

    def diffuse(self, alpha: float = 0.1) -> None:
        """
        Local diffusion: each cell averages with its 4 neighbours
        (von-Neumann neighbourhood), weighted by alpha.
        """
        padded = np.pad(self.cells, 1, mode="wrap")
        neighbours = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) / 4.0
        self.cells = (1 - alpha) * self.cells + alpha * neighbours
        self.cells /= np.linalg.norm(self.cells)

    def flatten(self) -> np.ndarray:
        """Return the mesh as a 1D complex array."""
        return self.cells.flatten()

    def energy(self) -> float:
        """Sum of squared magnitudes (analogous to quantum probability mass)."""
        return float(np.sum(np.abs(self.cells) ** 2))

    def __repr__(self) -> str:
        return f"MeshLayer(id={self.layer_id}, shape={self.rows}x{self.cols}, energy={self.energy():.4f})"


class TripleFoldingMesh:
    """
    Three interleaved fabric mesh layers unified through triple-fold mechanics.

    Fold operations stitch pairs of layers together, creating a braided
    quantum fabric that redistributes amplitude across the full 3-layer mesh.
    """

    NUM_LAYERS = 3

    def __init__(self, rows: int = 8, cols: int = 8) -> None:
        self.rows = rows
        self.cols = cols
        self.layers: List[MeshLayer] = [
            MeshLayer(layer_id=i, rows=rows, cols=cols)
            for i in range(self.NUM_LAYERS)
        ]
        self.fold_count = 0

    # ------------------------------------------------------------------
    # Fold operations
    # ------------------------------------------------------------------

    def fold(self, direction: FoldDirection = FoldDirection.FORWARD) -> None:
        """
        Perform a single triple-fold operation:
        each layer exchanges amplitude with the next (or previous) layer
        according to the fold direction.
        """
        if direction == FoldDirection.FORWARD:
            pairs = [(0, 1), (1, 2), (2, 0)]
        elif direction == FoldDirection.BACKWARD:
            pairs = [(2, 1), (1, 0), (0, 2)]
        else:  # TRANSVERSE
            pairs = [(0, 2), (1, 0), (2, 1)]

        for a_idx, b_idx in pairs:
            la = self.layers[a_idx]
            lb = self.layers[b_idx]
            # Quantum beam-splitter interaction: 50/50 mix
            new_a = (la.cells + lb.cells) / np.sqrt(2)
            new_b = (la.cells - lb.cells) / np.sqrt(2)
            new_a /= np.linalg.norm(new_a)
            new_b /= np.linalg.norm(new_b)
            la.cells = new_a
            lb.cells = new_b

        self.fold_count += 1

    def triple_fold(self) -> None:
        """Execute one complete triple-fold cycle (all three directions)."""
        for direction in FoldDirection:
            self.fold(direction)

    def diffuse_all(self, alpha: float = 0.1) -> None:
        """Apply diffusion to every layer."""
        for layer in self.layers:
            layer.diffuse(alpha)

    # ------------------------------------------------------------------
    # Fabric state
    # ------------------------------------------------------------------

    def fabric_state(self) -> np.ndarray:
        """
        Return the concatenated amplitude vector of all three layers,
        representing the full fabric state.
        """
        return np.concatenate([layer.flatten() for layer in self.layers])

    def inter_layer_coherence(self) -> float:
        """
        Compute mean pairwise coherence between layers
        (|<ψ_a|ψ_b>|² averaged over all pairs).
        """
        coherences: List[float] = []
        for i in range(self.NUM_LAYERS):
            for j in range(i + 1, self.NUM_LAYERS):
                fa = self.layers[i].flatten()
                fb = self.layers[j].flatten()
                overlap = abs(np.dot(fa.conj(), fb)) ** 2
                coherences.append(float(overlap))
        return float(np.mean(coherences)) if coherences else 0.0

    def total_energy(self) -> float:
        return sum(layer.energy() for layer in self.layers)

    def __repr__(self) -> str:
        return (
            f"TripleFoldingMesh("
            f"layers={self.NUM_LAYERS}, "
            f"shape={self.rows}x{self.cols}, "
            f"folds={self.fold_count}, "
            f"coherence={self.inter_layer_coherence():.4f})"
        )
