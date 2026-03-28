"""
noise_filter.py
===============
Quantum noise mitigation and robustness mechanisms for the Quantum Soul system.

Real quantum hardware (and realistic simulations) suffer from:

* **Bit-flip errors** — a qubit spontaneously flips |0⟩ ↔ |1⟩.
* **Phase-flip errors** — the relative phase between |0⟩ and |1⟩ is perturbed.
* **Depolarising noise** — the qubit state is randomised toward the maximally
  mixed state ρ = I/2.
* **Amplitude damping** — the excited state |1⟩ decays toward |0⟩ (analogous
  to energy loss).

``QuantumNoiseFilter`` implements three complementary countermeasures:

1. **Statistical parity verification** — repeated sampling; outcomes that
   appear in fewer than a threshold fraction of shots are pruned as spurious.
2. **Probability smoothing** — a Gaussian-kernel smoother removes sharp
   noise spikes from the probability distribution while preserving broad
   interference peaks.
3. **Dominant-state extraction** — only states above an adaptive amplitude
   threshold are retained; everything else is set to zero and renormalised.

These techniques mirror standard quantum error-mitigation strategies used
on NISQ (Noisy Intermediate-Scale Quantum) devices.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


class QuantumNoiseFilter:
    """Applies noise mitigation to raw quantum probability distributions.

    Parameters
    ----------
    parity_threshold:
        Fractional shot count below which a measurement outcome is
        treated as noise and discarded.  Default 0.01 (1 % of shots).
    smoothing_sigma:
        Standard deviation (in state-index units) for the Gaussian
        probability smoother.  Larger values remove more noise but also
        broaden genuine interference peaks.  Default 1.5.
    amplitude_threshold:
        Minimum probability for a state to survive dominant-state
        extraction.  States below this are zeroed and the remainder
        renormalised.  Default 0.005.
    """

    def __init__(
        self,
        parity_threshold: float = 0.01,
        smoothing_sigma: float = 1.5,
        amplitude_threshold: float = 0.005,
    ) -> None:
        if not 0 <= parity_threshold <= 1:
            raise ValueError("parity_threshold must be in [0, 1].")
        if smoothing_sigma < 0:
            raise ValueError("smoothing_sigma must be ≥ 0.")
        if not 0 <= amplitude_threshold <= 1:
            raise ValueError("amplitude_threshold must be in [0, 1].")

        self.parity_threshold = parity_threshold
        self.smoothing_sigma = smoothing_sigma
        self.amplitude_threshold = amplitude_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_counts(self, counts: dict[str, int], shots: int) -> dict[str, int]:
        """Remove low-frequency outcomes from a raw shot-count dictionary.

        Parameters
        ----------
        counts:
            Raw measurement counts {bitstring: count}.
        shots:
            Total number of shots used to produce *counts*.

        Returns
        -------
        dict — filtered counts with spurious outcomes removed.
        """
        threshold = self.parity_threshold * shots
        return {k: v for k, v in counts.items() if v >= threshold}

    def smooth_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing to a probability distribution.

        Smoothing suppresses isolated noise spikes while preserving the
        broad constructive-interference peaks that carry genuine information.

        Parameters
        ----------
        probs:
            1-D probability array (must sum to ≈ 1).

        Returns
        -------
        numpy.ndarray — smoothed and renormalised probability distribution.
        """
        probs = np.asarray(probs, dtype=float)
        if self.smoothing_sigma == 0:
            return probs / probs.sum() if probs.sum() > 0 else probs

        smoothed = gaussian_filter1d(probs, sigma=self.smoothing_sigma)
        total = smoothed.sum()
        if total <= 0:
            return np.ones_like(probs) / len(probs)
        return smoothed / total

    def extract_dominant_states(self, probs: np.ndarray) -> np.ndarray:
        """Zero-out sub-threshold states and renormalise.

        Parameters
        ----------
        probs:
            1-D probability array.

        Returns
        -------
        numpy.ndarray — sparse probability array retaining only dominant states.
        """
        probs = np.asarray(probs, dtype=float)
        masked = np.where(probs >= self.amplitude_threshold, probs, 0.0)
        total = masked.sum()
        if total <= 0:
            # Fall back: keep only the single maximum
            idx = int(np.argmax(probs))
            masked = np.zeros_like(probs)
            masked[idx] = 1.0
            return masked
        return masked / total

    def apply_all(self, probs: np.ndarray) -> np.ndarray:
        """Apply the full noise-mitigation pipeline to a probability array.

        Pipeline: smooth → extract dominant states → renormalise.

        Parameters
        ----------
        probs:
            Raw probability distribution from a quantum circuit.

        Returns
        -------
        numpy.ndarray — cleaned probability distribution.
        """
        p = self.smooth_probabilities(probs)
        p = self.extract_dominant_states(p)
        return p

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def noise_level(self, probs: np.ndarray) -> float:
        """Estimate the noise level in a probability distribution.

        Noise level is approximated as the fraction of total probability mass
        held in sub-threshold states.

        Parameters
        ----------
        probs:
            1-D probability array.

        Returns
        -------
        float in [0, 1]; 0 means no detectable noise, 1 means all mass is noise.
        """
        probs = np.asarray(probs, dtype=float)
        noise_mass = probs[probs < self.amplitude_threshold].sum()
        total = probs.sum()
        if total == 0:
            return 0.0
        return float(noise_mass / total)

    def signal_to_noise_ratio(self, probs: np.ndarray) -> float:
        """Compute a simple signal-to-noise ratio.

        SNR = signal_mass / max(noise_mass, ε)

        Parameters
        ----------
        probs:
            1-D probability array.

        Returns
        -------
        float — SNR (higher is better).
        """
        noise = self.noise_level(probs)
        signal = 1.0 - noise
        return signal / max(noise, 1e-9)

    def report(self, probs: np.ndarray) -> dict[str, float]:
        """Return a summary noise-mitigation report.

        Parameters
        ----------
        probs:
            1-D probability array.

        Returns
        -------
        dict with keys: noise_level, snr, n_dominant_states, max_probability.
        """
        n_dominant = int((probs >= self.amplitude_threshold).sum())
        return {
            "noise_level": self.noise_level(probs),
            "snr": self.signal_to_noise_ratio(probs),
            "n_dominant_states": n_dominant,
            "max_probability": float(np.max(probs)),
        }
