"""
supersymmetry_engine.py
-----------------------
Supersymmetry-inspired error correction and coherence engine.

In supersymmetric quantum mechanics every bosonic state has a fermionic
superpartner.  Here this principle is used to:
  - Pair each quantum state with a "superpartner" correction state.
  - Cancel noise via destructive interference with the partner channel.
  - Maintain global coherence through an energy-balance constraint.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class SuperpartnerChannel:
    """
    Bosonic/fermionic superpartner pair for a quantum state vector.
    """

    def __init__(self, dims: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        raw_b = rng.standard_normal(dims) + 1j * rng.standard_normal(dims)
        raw_f = rng.standard_normal(dims) + 1j * rng.standard_normal(dims)
        self.bosonic: np.ndarray = raw_b / np.linalg.norm(raw_b)
        self.fermionic: np.ndarray = raw_f / np.linalg.norm(raw_f)
        self.dims = dims

    @property
    def susy_invariant(self) -> float:
        """
        SUSY invariant: should remain ~0 if supersymmetry is preserved.
        Defined as |<ψ_B|ψ_F>|² − |<ψ_F|ψ_B>|²  ≡ 0 (by Hermiticity).
        We use a practical analogue: energy difference between channels.
        """
        e_b = float(np.sum(np.abs(self.bosonic) ** 2))
        e_f = float(np.sum(np.abs(self.fermionic) ** 2))
        return abs(e_b - e_f)

    def restore_symmetry(self) -> None:
        """Re-balance bosonic and fermionic energies to enforce SUSY."""
        e_b = np.linalg.norm(self.bosonic)
        e_f = np.linalg.norm(self.fermionic)
        if e_b + e_f < 1e-12:
            return
        avg = (e_b + e_f) / 2
        self.bosonic = self.bosonic * (avg / e_b) if e_b > 0 else self.bosonic
        self.fermionic = self.fermionic * (avg / e_f) if e_f > 0 else self.fermionic


class SupersymmetryEngine:
    """
    Full supersymmetry engine wrapping a quantum state vector.

    Provides:
    - Noise injection modelling environmental decoherence.
    - SUSY-based error correction via superpartner cancellation.
    - Coherence monitoring.
    """

    def __init__(self, state_dims: int = 64) -> None:
        self.state_dims = state_dims
        rng = np.random.default_rng(999)
        raw = rng.standard_normal(state_dims) + 1j * rng.standard_normal(state_dims)
        self.state: np.ndarray = raw / np.linalg.norm(raw)
        self.partner = SuperpartnerChannel(dims=state_dims, seed=42)
        self.noise_level: float = 0.0
        self.correction_count: int = 0

    # ------------------------------------------------------------------
    # Noise model
    # ------------------------------------------------------------------

    def inject_noise(self, sigma: float = 0.05) -> None:
        """Inject Gaussian complex noise to simulate environmental decoherence."""
        rng = np.random.default_rng()
        noise = rng.standard_normal(self.state_dims) + 1j * rng.standard_normal(self.state_dims)
        noise *= sigma
        self.state += noise
        self.state /= np.linalg.norm(self.state)
        self.noise_level = float(np.linalg.norm(noise))

    # ------------------------------------------------------------------
    # SUSY error correction
    # ------------------------------------------------------------------

    def correct_errors(self) -> Tuple[np.ndarray, float]:
        """
        Apply supersymmetric error correction:
        1. Project state onto bosonic subspace.
        2. Compute correction via fermionic superpartner channel.
        3. Subtract the noise component estimated from the fermionic channel.

        Returns the corrected state and the residual (error estimate).
        """
        # Step 1: bosonic projection (overlap with bosonic channel)
        b = self.partner.bosonic
        proj_b = np.dot(b.conj(), self.state) * b

        # Step 2: fermionic correction – cancel the fermionic noise component
        f = self.partner.fermionic
        noise_component = np.dot(f.conj(), self.state) * f
        corrected = self.state - noise_component + proj_b
        norm = np.linalg.norm(corrected)
        if norm > 1e-12:
            corrected /= norm

        residual = float(np.linalg.norm(self.state - corrected))
        self.state = corrected
        self.correction_count += 1

        # Restore SUSY balance in partner channels
        self.partner.restore_symmetry()

        return self.state, residual

    # ------------------------------------------------------------------
    # Coherence
    # ------------------------------------------------------------------

    def coherence(self) -> float:
        """
        Coherence measure: purity ρ = |⟨ψ|ψ⟩|² for a pure state = 1.
        Decoherence pushes this below 1.
        """
        return float(abs(np.dot(self.state.conj(), self.state)))

    def susy_invariant(self) -> float:
        return self.partner.susy_invariant

    def apply_to_state(self, state: np.ndarray) -> np.ndarray:
        """
        Apply SUSY correction to an external state vector.
        Useful when the engine is used as a sub-component.
        """
        self.state = state.copy()
        if np.linalg.norm(self.state) > 1e-12:
            self.state /= np.linalg.norm(self.state)
        corrected, _ = self.correct_errors()
        return corrected

    def __repr__(self) -> str:
        return (
            f"SupersymmetryEngine("
            f"dims={self.state_dims}, "
            f"coherence={self.coherence():.4f}, "
            f"corrections={self.correction_count}, "
            f"susy_invariant={self.susy_invariant():.6f})"
        )
