"""
synapse.py — Synaptic transmission between neurons.

Models both excitatory (AMPA/NMDA-like) and inhibitory (GABA-like) synapses
with:
- Weighted conductance-based transmission
- Neurotransmitter release probability (quantum uncertainty)
- Short-term synaptic depression and facilitation
- Synaptic delay simulation
"""

import random
import math
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

from .neuron import Neuron


class SynapseType(Enum):
    """Classification of synaptic connection."""
    EXCITATORY = auto()   # Depolarising (AMPA/NMDA-like), increases Vm
    INHIBITORY = auto()   # Hyperpolarising (GABA-like), decreases Vm


@dataclass
class SynapseConfig:
    """Tunable parameters for a synapse."""
    weight: float = 1.0                  # Synaptic efficacy (arbitrary units)
    delay_ms: float = 1.0                # Axonal conduction + synaptic delay
    release_probability: float = 0.8    # Pr — quantal release probability
    reversal_potential_mV: float = 0.0  # Erev; excitatory ≈ 0, inhib ≈ -70
    decay_time_constant_ms: float = 5.0  # EPSC/IPSC decay tau
    # Short-term plasticity (Tsodyks-Markram model)
    use_short_term_plasticity: bool = True
    depression_factor: float = 0.1      # Resource depletion rate
    facilitation_factor: float = 0.01   # Ca²⁺ facilitation rate
    recovery_time_constant_ms: float = 200.0  # Resource recovery tau


class Synapse:
    """
    Conductance-based synapse with stochastic vesicle release.

    Transmits spikes from a pre-synaptic neuron to a post-synaptic neuron,
    computing the post-synaptic current (PSC) injected each timestep.
    """

    def __init__(
        self,
        synapse_id: str,
        pre_neuron: Neuron,
        post_neuron: Neuron,
        synapse_type: SynapseType = SynapseType.EXCITATORY,
        config: Optional[SynapseConfig] = None,
    ):
        self.synapse_id = synapse_id
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.synapse_type = synapse_type
        self.config = config or SynapseConfig()

        # Ensure the reversal potential default matches synapse type
        if config is None:
            if synapse_type == SynapseType.INHIBITORY:
                self.config.reversal_potential_mV = -70.0
            else:
                self.config.reversal_potential_mV = 0.0

        # Conductance state (nS) — decays exponentially between spikes
        self._conductance: float = 0.0

        # Short-term plasticity state variables (Tsodyks-Markram)
        self._resource_fraction: float = 1.0   # x: fraction of resources available
        self._use_fraction: float = self.config.release_probability  # u: utilisation

        # Pending spike queue: list of (arrival_time_ms) relative to a local clock
        self._pending_spikes: list = []
        self._local_clock_ms: float = 0.0

        # Metrics
        self.total_transmissions: int = 0
        self.failed_releases: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, dt_ms: float = 1.0) -> float:
        """
        Advance the synapse by dt_ms and inject current into the post neuron.

        Returns the post-synaptic current (pA) injected this step.
        """
        self._local_clock_ms += dt_ms

        # Check if the pre-synaptic neuron just fired
        if self.pre_neuron.is_firing or self._pre_fired_recently():
            self._enqueue_spike()

        # Deliver any spikes that have crossed their delay
        self._deliver_due_spikes()

        # Decay conductance
        tau = self.config.decay_time_constant_ms
        self._conductance *= math.exp(-dt_ms / tau)

        # Recover short-term plasticity resources
        if self.config.use_short_term_plasticity:
            self._recover_resources(dt_ms)

        # Compute and inject post-synaptic current
        psc = self._compute_psc()
        if psc != 0.0:
            self.post_neuron.receive_input(psc)

        return psc

    @property
    def effective_weight(self) -> float:
        """Current weight including short-term plasticity modulation."""
        if self.config.use_short_term_plasticity:
            return self.config.weight * self._resource_fraction * self._use_fraction
        return self.config.weight

    def set_weight(self, new_weight: float) -> None:
        """Update synaptic weight (called by neuroplasticity engine)."""
        self.config.weight = max(0.0, new_weight)

    def __repr__(self) -> str:
        return (
            f"Synapse(id={self.synapse_id!r}, "
            f"{self.pre_neuron.neuron_id}→{self.post_neuron.neuron_id}, "
            f"type={self.synapse_type.name}, w={self.config.weight:.3f})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pre_fired_recently(self) -> bool:
        """Detect a new pre-synaptic spike within this timestep."""
        if not self.pre_neuron.spike_times:
            return False
        last_spike = self.pre_neuron.spike_times[-1]
        return (time.monotonic() - last_spike) < 0.002  # within 2 ms

    def _enqueue_spike(self) -> None:
        """Add an incoming spike to the delay queue."""
        arrival_ms = self._local_clock_ms + self.config.delay_ms
        # Avoid double-queueing within same timestep
        if not self._pending_spikes or abs(self._pending_spikes[-1] - arrival_ms) > 0.1:
            self._pending_spikes.append(arrival_ms)

    def _deliver_due_spikes(self) -> None:
        """Process all spikes that have reached their delivery time."""
        due = [t for t in self._pending_spikes if t <= self._local_clock_ms]
        remaining = [t for t in self._pending_spikes if t > self._local_clock_ms]
        self._pending_spikes = remaining

        for _ in due:
            self._release_neurotransmitter()

    def _release_neurotransmitter(self) -> None:
        """
        Stochastic vesicle release.

        With probability Pr (release_probability), a quantum of
        neurotransmitter is released, increasing post-synaptic conductance.
        """
        self.total_transmissions += 1

        # Short-term plasticity: update utilisation and resources
        if self.config.use_short_term_plasticity:
            self._use_fraction += self.config.facilitation_factor * (1.0 - self._use_fraction)
            released_fraction = self._use_fraction * self._resource_fraction
            self._resource_fraction -= released_fraction
            release_pr = min(1.0, max(0.0, released_fraction))
        else:
            release_pr = self.config.release_probability

        # Quantum probabilistic release
        if random.random() < release_pr:
            self._conductance += self.effective_weight
        else:
            self.failed_releases += 1

    def _recover_resources(self, dt_ms: float) -> None:
        """Exponential recovery of depleted synaptic resources."""
        tau_rec = self.config.recovery_time_constant_ms
        self._resource_fraction += (1.0 - self._resource_fraction) * (1.0 - math.exp(-dt_ms / tau_rec))
        # Facilitation variable decays back to baseline
        self._use_fraction += (self.config.release_probability - self._use_fraction) * (
            1.0 - math.exp(-dt_ms / tau_rec)
        )

    def _compute_psc(self) -> float:
        """
        Ohm's-law post-synaptic current: I = g * (Vm - Erev).

        Sign convention:
          - Excitatory (Erev ≈ 0 mV): current is inward (positive pA) when Vm < 0
          - Inhibitory (Erev ≈ -70 mV): current is outward (negative pA) when Vm > -70
        """
        vm = self.post_neuron.membrane_potential
        erev = self.config.reversal_potential_mV
        current = self._conductance * (erev - vm)
        return current
