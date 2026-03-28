"""
neuron.py — Biological neuron model with quantum-enhanced firing mechanics.

Models the Hodgkin-Huxley inspired integrate-and-fire neuron, where:
- Each neuron accumulates membrane potential from synaptic inputs.
- When potential crosses a threshold, an action potential (spike) fires.
- After firing, the neuron enters a refractory period.
- Quantum superposition allows probabilistic pre-firing based on noise.
"""

import math
import random
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional


class NeuronState(Enum):
    """Possible states of a biological neuron."""
    RESTING = auto()       # Baseline membrane potential
    INTEGRATING = auto()   # Accumulating synaptic input
    FIRING = auto()        # Action potential spike in progress
    REFRACTORY = auto()    # Post-spike recovery period
    QUANTUM_SUPERPOSITION = auto()  # Probabilistic pre-fire state


@dataclass
class NeuronConfig:
    """Configuration parameters for a neuron, inspired by biology."""
    resting_potential: float = -70.0      # mV — typical neuron resting potential
    threshold_potential: float = -55.0   # mV — action potential threshold
    peak_potential: float = 40.0         # mV — peak spike amplitude
    reset_potential: float = -75.0       # mV — post-spike hyperpolarization
    refractory_period_ms: float = 2.0    # ms — absolute refractory period
    membrane_time_constant_ms: float = 20.0  # ms — RC time constant (tau)
    leak_conductance: float = 0.1        # nS — passive membrane leak
    quantum_noise_amplitude: float = 0.5  # mV — quantum fluctuation amplitude
    max_firing_rate_hz: float = 200.0    # Hz — biological upper bound


class Neuron:
    """
    Bio-inspired quantum neuron.

    Implements a leaky integrate-and-fire (LIF) model with:
    - Deterministic integration of synaptic potentials
    - Quantum-noise perturbation for probabilistic firing
    - Refractory period enforcement
    - Spike-time recording for STDP learning
    """

    def __init__(
        self,
        neuron_id: str,
        config: Optional[NeuronConfig] = None,
        layer_name: str = "hidden",
    ):
        self.neuron_id = neuron_id
        self.layer_name = layer_name
        self.config = config or NeuronConfig()

        # Membrane state
        self.membrane_potential: float = self.config.resting_potential
        self.state: NeuronState = NeuronState.RESTING

        # Timing
        self._last_fire_time: float = -math.inf  # seconds since epoch
        self._refractory_end_time: float = -math.inf

        # Spike history (timestamps in seconds)
        self.spike_times: List[float] = []

        # Accumulated input current this timestep (pA)
        self._input_current: float = 0.0

        # Metrics
        self.total_spikes: int = 0
        self.activation_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def receive_input(self, current_pA: float) -> None:
        """Accumulate synaptic current into this neuron's input buffer."""
        self._input_current += current_pA
        if self.state == NeuronState.RESTING and current_pA != 0:
            self.state = NeuronState.INTEGRATING

    def step(self, dt_ms: float = 1.0) -> bool:
        """
        Advance the neuron by dt_ms milliseconds.

        Returns True if an action potential was fired this step.
        """
        now = time.monotonic()

        # Enforce refractory period
        if self.state == NeuronState.REFRACTORY:
            if now < self._refractory_end_time:
                self._input_current = 0.0
                return False
            else:
                self.state = NeuronState.RESTING
                self.membrane_potential = self.config.reset_potential

        # Apply quantum noise (simulates thermal and quantum fluctuations)
        quantum_noise = self._quantum_fluctuation()

        # Leaky integration: dV/dt = (-V_leak + I_syn) / tau
        dt_s = dt_ms / 1000.0
        tau = self.config.membrane_time_constant_ms / 1000.0
        leak = (self.config.resting_potential - self.membrane_potential) * self.config.leak_conductance
        dv = (dt_s / tau) * (leak + self._input_current + quantum_noise)
        self.membrane_potential += dv

        # Clamp membrane potential to physiological range
        self.membrane_potential = max(
            self.config.reset_potential - 5.0,
            min(self.membrane_potential, self.config.peak_potential + 5.0),
        )

        # Reset input buffer
        self._input_current = 0.0
        self.activation_count += 1

        # Check firing threshold
        if self.membrane_potential >= self.config.threshold_potential:
            return self._fire(now)

        # Update state
        if abs(self.membrane_potential - self.config.resting_potential) < 1.0:
            self.state = NeuronState.RESTING
        else:
            self.state = NeuronState.INTEGRATING

        return False

    def force_fire(self) -> bool:
        """Externally trigger an action potential (e.g., sensory input node)."""
        now = time.monotonic()
        if self.state == NeuronState.REFRACTORY and now < self._refractory_end_time:
            return False
        return self._fire(now)

    def reset(self) -> None:
        """Reset neuron to resting state."""
        self.membrane_potential = self.config.resting_potential
        self.state = NeuronState.RESTING
        self._input_current = 0.0
        self._last_fire_time = -math.inf
        self._refractory_end_time = -math.inf

    @property
    def is_firing(self) -> bool:
        return self.state == NeuronState.FIRING

    @property
    def is_refractory(self) -> bool:
        return self.state == NeuronState.REFRACTORY

    @property
    def firing_rate_hz(self) -> float:
        """Estimate instantaneous firing rate from recent spike history."""
        recent_window_s = 1.0
        now = time.monotonic()
        recent_spikes = [t for t in self.spike_times if now - t <= recent_window_s]
        return float(len(recent_spikes))

    def __repr__(self) -> str:
        return (
            f"Neuron(id={self.neuron_id!r}, layer={self.layer_name!r}, "
            f"Vm={self.membrane_potential:.1f}mV, state={self.state.name}, "
            f"spikes={self.total_spikes})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fire(self, now: float) -> bool:
        """Execute action potential and enter refractory period."""
        self.membrane_potential = self.config.peak_potential
        self.state = NeuronState.FIRING
        self.total_spikes += 1
        self._last_fire_time = now
        self._refractory_end_time = now + (self.config.refractory_period_ms / 1000.0)
        self.spike_times.append(now)

        # Trim spike history to last 2 seconds
        cutoff = now - 2.0
        self.spike_times = [t for t in self.spike_times if t >= cutoff]

        # Transition to refractory after recording spike
        self.state = NeuronState.REFRACTORY
        return True

    def _quantum_fluctuation(self) -> float:
        """
        Generate quantum noise representing sub-threshold membrane fluctuations.

        Uses a Gaussian distribution scaled by the configured amplitude.
        In a biological neuron, this corresponds to stochastic ion-channel
        gating events that create shot noise.
        """
        amplitude = self.config.quantum_noise_amplitude
        return random.gauss(0.0, amplitude)
