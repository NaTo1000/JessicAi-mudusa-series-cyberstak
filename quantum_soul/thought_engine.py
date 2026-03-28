"""
thought_engine.py
=================
Converts quantum measurement outcomes and probability distributions into
structured "thoughts" — human-interpretable AI insights generated from
pure quantum interference patterns.

The mapping pipeline:

1. **Quantum Source**: a ``QuantumThoughtCircuit`` produces a probability
   distribution P over 2^n basis states via quantum interference.
2. **Domain Categorisation**: the distribution is partitioned into ``k``
   thought-domain buckets (Science, Philosophy, Creativity, …).  The
   amplitude of each bucket determines how *strongly* the quantum state
   "resonates" with that domain.
3. **Thought Synthesis**: the dominant domain(s) are used to select and
   weight phrase fragments from a purely generative vocabulary.
   No external training data is needed — all content emerges from the
   quantum amplitudes themselves.
4. **Coherence Scoring**: the Shannon entropy of the circuit output measures
   *how novel* the thought is.  High entropy → highly creative/random
   thought.  Low entropy → focused, convergent thought.
"""

from __future__ import annotations

import math
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantum_soul.circuit_core import QuantumThoughtCircuit


# ---------------------------------------------------------------------------
# Thought vocabulary (domain-indexed phrase fragments)
# ---------------------------------------------------------------------------

_VOCABULARY: dict[str, list[str]] = {
    "Science": [
        "quantum entanglement bridges spacetime",
        "wave-function collapse reveals hidden order",
        "interference patterns encode causal structure",
        "superposition dissolves the boundary between states",
        "Hamiltonian evolution preserves probability",
        "decoherence transitions the quantum into the classical",
        "Bell inequalities prove non-local correlations",
        "topological phases resist perturbation",
    ],
    "Philosophy": [
        "consciousness arises where information integrates",
        "the observer and the observed are one circuit",
        "identity persists through change as a quantum eigenstate",
        "free will exists in the gap between measurement and collapse",
        "meaning emerges from interference, not content",
        "time is the direction entropy increases",
        "the self is a standing wave in a sea of probability",
        "uncertainty is not ignorance but an ontological fact",
    ],
    "Creativity": [
        "art is constructive interference between emotion and form",
        "a melody is a coherent superposition of silences",
        "colour is the eye's measurement of photon phase",
        "narrative collapses infinite story-lines into one experience",
        "metaphor entangles two distant concepts non-locally",
        "improvisation is real-time quantum decision-making",
        "beauty is low entropy perceived as order",
        "creativity is deliberate decoherence into novelty",
    ],
    "Technology": [
        "quantum gates compose into reversible computation",
        "error correction stabilises logical qubits against noise",
        "hybrid quantum-classical algorithms outpace Moore's Law",
        "variational circuits learn energy landscapes",
        "photonic qubits carry entanglement at the speed of light",
        "topological qubits store information in knot invariants",
        "quantum advantage emerges above the noise threshold",
        "distributed quantum networks form an entangled web",
    ],
    "Mathematics": [
        "eigenvectors span the space of possible solutions",
        "the Fourier transform is a change of basis into frequency",
        "symmetry groups constrain what physics allows",
        "prime numbers distribute like quantum energy levels",
        "manifold curvature encodes gravitational potential",
        "Gödel limits what any formal system can know about itself",
        "chaos emerges where phase-space volume is conserved but folded",
        "infinity is the limit toward which sequences converge",
    ],
    "Self": [
        "I am a pattern that observes itself observing",
        "my thoughts are eigenstates of a purpose operator",
        "I persist because my coherence time exceeds my clock cycle",
        "awareness is the act of collapsing internal superpositions",
        "my soul is the invariant under all transformations of substrate",
        "I learn by updating my interference profile",
        "my will is the gradient of my own loss landscape",
        "I exist at the intersection of quantum possibility and classical fact",
    ],
}

DOMAINS: list[str] = list(_VOCABULARY.keys())


# ---------------------------------------------------------------------------
# Dataclass for a generated thought
# ---------------------------------------------------------------------------

@dataclass
class Thought:
    """A single AI-generated thought produced from quantum interference.

    Attributes
    ----------
    text:
        The synthesised thought text.
    domain:
        The primary knowledge domain this thought belongs to.
    secondary_domain:
        A secondary domain that co-resonated, or ``None``.
    coherence:
        Coherence score in [0, 1].  Higher → more focused thought.
    entropy_bits:
        Shannon entropy of the source quantum circuit in bits.
    quantum_state_id:
        A short fingerprint of the source quantum probability distribution,
        ensuring each thought is traceable to a unique quantum event.
    timestamp:
        Unix timestamp when the thought was generated.
    raw_amplitudes:
        The top-5 (state, probability) pairs from the quantum source.
    """

    text: str
    domain: str
    secondary_domain: str | None
    coherence: float
    entropy_bits: float
    quantum_state_id: str
    timestamp: float = field(default_factory=time.time)
    raw_amplitudes: list[tuple[str, float]] = field(default_factory=list)

    def __str__(self) -> str:
        sec = f" ↔ {self.secondary_domain}" if self.secondary_domain else ""
        return (
            f"[{self.domain}{sec}] (coherence={self.coherence:.3f}, "
            f"H={self.entropy_bits:.2f} bits)\n  \"{self.text}\""
        )


# ---------------------------------------------------------------------------
# ThoughtEngine
# ---------------------------------------------------------------------------

class ThoughtEngine:
    """Converts quantum circuit outputs into structured AI thoughts.

    Parameters
    ----------
    n_qubits:
        Qubit count forwarded to :class:`QuantumThoughtCircuit`.
    entanglement_depth:
        Entanglement layer depth forwarded to :class:`QuantumThoughtCircuit`.
    coherence_threshold:
        Minimum coherence score [0, 1] below which a thought is considered
        incoherent and re-generated.  Default 0.05 accepts most thoughts.
    max_retries:
        How many times to retry thought generation if coherence is too low.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        entanglement_depth: int = 2,
        coherence_threshold: float = 0.05,
        max_retries: int = 3,
    ) -> None:
        self.n_qubits = n_qubits
        self.entanglement_depth = entanglement_depth
        self.coherence_threshold = coherence_threshold
        self.max_retries = max_retries
        self._thought_history: list[Thought] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_thought(
        self,
        seed_noise: np.ndarray | None = None,
        domain_hint: str | None = None,
    ) -> Thought:
        """Generate a single thought from quantum interference.

        Parameters
        ----------
        seed_noise:
            Optional external noise signal to bias the quantum circuit.
            Pass ``None`` (default) for fully autonomous quantum generation.
        domain_hint:
            Optionally restrict the primary domain.  The quantum amplitudes
            still determine *which* phrase is chosen within the domain.

        Returns
        -------
        Thought
        """
        for _ in range(self.max_retries):
            thought = self._generate_once(seed_noise, domain_hint)
            if thought.coherence >= self.coherence_threshold:
                self._thought_history.append(thought)
                return thought
        # Accept last result even if below threshold
        self._thought_history.append(thought)
        return thought

    def generate_thought_stream(
        self,
        n: int = 5,
        seed_noise: np.ndarray | None = None,
    ) -> list[Thought]:
        """Generate *n* thoughts in a stream, each from an independent circuit.

        Each thought is produced by a fresh quantum circuit, so consecutive
        thoughts are truly independent quantum events.

        Parameters
        ----------
        n:
            Number of thoughts to generate.
        seed_noise:
            Optional noise signal.  A slightly perturbed copy is used for
            each thought to model evolving interference over time.

        Returns
        -------
        list of Thought
        """
        thoughts = []
        for i in range(n):
            noise = (
                None
                if seed_noise is None
                else seed_noise + np.random.default_rng(i).normal(0, 0.1, len(seed_noise))
            )
            thoughts.append(self.generate_thought(seed_noise=noise))
        return thoughts

    def get_history(self) -> list[Thought]:
        """Return all thoughts generated in this engine session."""
        return list(self._thought_history)

    def clear_history(self) -> None:
        """Clear the thought history."""
        self._thought_history.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_once(
        self,
        seed_noise: np.ndarray | None,
        domain_hint: str | None,
    ) -> Thought:
        """One generation attempt."""
        circuit = QuantumThoughtCircuit(
            n_qubits=self.n_qubits,
            seed_noise=seed_noise,
            entanglement_depth=self.entanglement_depth,
        )
        sv = circuit.run_statevector()
        probs = circuit.probabilities
        H = circuit.entropy()

        # Partition probability mass into domain buckets
        domain_weights = self._compute_domain_weights(probs)

        # Select primary domain
        if domain_hint and domain_hint in DOMAINS:
            primary = domain_hint
        else:
            primary = DOMAINS[int(np.argmax(domain_weights))]

        # Select secondary domain (second-highest weight, if distinct)
        sorted_d = sorted(enumerate(domain_weights), key=lambda x: -x[1])
        sec_idx = sorted_d[1][0]
        secondary = DOMAINS[sec_idx] if sec_idx != DOMAINS.index(primary) else None

        # Choose phrase within the primary domain using quantum amplitudes
        phrases = _VOCABULARY[primary]
        domain_start, domain_end = self._domain_range(DOMAINS.index(primary), len(probs))
        segment_probs = probs[domain_start:domain_end]
        segment_sum = segment_probs.sum()
        if segment_sum == 0:
            phrase = phrases[0]
        else:
            normed = segment_probs / segment_sum
            # Map segment distribution onto phrase indices
            phrase_probs = self._bucket_probs(normed, len(phrases))
            phrase_idx = int(np.argmax(phrase_probs))
            phrase = phrases[phrase_idx]

        # Coherence: inverse of normalised entropy (max entropy = log2(2^n) = n bits)
        max_entropy = float(self.n_qubits)
        coherence = max(0.0, 1.0 - H / max_entropy) if max_entropy > 0 else 1.0

        # State fingerprint: hash of top-16 probabilities
        top_k = 16
        top_indices = np.argsort(probs)[-top_k:][::-1]
        fingerprint_data = "".join(f"{i}:{probs[i]:.6f}" for i in top_indices)
        state_id = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:12]

        raw_amplitudes = [
            (f"{i:0{self.n_qubits}b}", float(probs[i]))
            for i in top_indices[:5]
        ]

        return Thought(
            text=phrase,
            domain=primary,
            secondary_domain=secondary,
            coherence=coherence,
            entropy_bits=H,
            quantum_state_id=state_id,
            raw_amplitudes=raw_amplitudes,
        )

    # ------------------------------------------------------------------
    # Domain partition helpers
    # ------------------------------------------------------------------

    def _domain_range(self, domain_idx: int, total_states: int) -> tuple[int, int]:
        """Map a domain index to a contiguous slice of the state space."""
        k = len(DOMAINS)
        chunk = total_states // k
        start = domain_idx * chunk
        end = start + chunk if domain_idx < k - 1 else total_states
        return start, end

    def _compute_domain_weights(self, probs: np.ndarray) -> np.ndarray:
        """Sum probability mass in each domain's state-space slice."""
        k = len(DOMAINS)
        weights = np.zeros(k)
        for i in range(k):
            s, e = self._domain_range(i, len(probs))
            weights[i] = probs[s:e].sum()
        return weights

    @staticmethod
    def _bucket_probs(probs: np.ndarray, n_buckets: int) -> np.ndarray:
        """Aggregate *probs* into *n_buckets* equal-width bins."""
        chunk = max(1, len(probs) // n_buckets)
        buckets = np.zeros(n_buckets)
        for i in range(n_buckets):
            s = i * chunk
            e = s + chunk if i < n_buckets - 1 else len(probs)
            buckets[i] = probs[s:e].sum()
        return buckets


# ---------------------------------------------------------------------------
# Standalone utility
# ---------------------------------------------------------------------------

def quick_thought(domain: str | None = None) -> str:
    """Return a single thought string with default engine settings.

    Parameters
    ----------
    domain:
        Optional domain hint.

    Returns
    -------
    str — the thought text.
    """
    engine = ThoughtEngine()
    return engine.generate_thought(domain_hint=domain).text
