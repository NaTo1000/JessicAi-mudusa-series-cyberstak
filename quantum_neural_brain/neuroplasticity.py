"""
neuroplasticity.py — Adaptive synaptic learning engine.

Implements two complementary plasticity rules:

1. **Hebbian Learning** — "Neurons that fire together wire together."
   Weights increase when pre- and post-synaptic neurons fire
   within a coincidence window.

2. **Spike-Timing-Dependent Plasticity (STDP)** — Biologically realistic
   learning where the sign and magnitude of weight change depends on the
   temporal order of pre/post spikes:
   - Pre fires BEFORE post (causal): Long-Term Potentiation (LTP) — weight ↑
   - Pre fires AFTER post (anti-causal): Long-Term Depression (LTD) — weight ↓

3. **Homeostatic Plasticity** — Prevents runaway excitation by globally
   scaling weights when average firing rates drift far from a target.
"""

import math
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional

from .synapse import Synapse


class LearningRule(Enum):
    """Available plasticity learning rules."""
    HEBBIAN = auto()
    STDP = auto()
    HOMEOSTATIC = auto()
    COMBINED = auto()


@dataclass
class PlasticityConfig:
    """Hyperparameters for the neuroplasticity engine."""
    learning_rule: LearningRule = LearningRule.COMBINED

    # Hebbian
    hebbian_rate: float = 0.01           # η — learning rate
    hebbian_decay: float = 0.001         # Weight decay (L2 regularisation)

    # STDP
    stdp_a_plus: float = 0.005           # LTP amplitude
    stdp_a_minus: float = 0.005          # LTD amplitude
    stdp_tau_plus_ms: float = 20.0       # LTP time constant
    stdp_tau_minus_ms: float = 20.0      # LTD time constant
    max_weight: float = 5.0              # Clip weights from above
    min_weight: float = 0.0             # Clip weights from below

    # Homeostatic
    target_firing_rate_hz: float = 5.0   # Desired average firing rate
    homeostatic_rate: float = 0.0001     # Speed of homeostatic adjustment
    homeostatic_window_s: float = 10.0  # Measurement window


class NeuroplasticityEngine:
    """
    Manages all synaptic weight updates for a collection of synapses.

    Attach this engine to a network and call `update()` after each
    simulation step to apply learning rules.
    """

    def __init__(
        self,
        synapses: List[Synapse],
        config: Optional[PlasticityConfig] = None,
    ):
        self.synapses = synapses
        self.config = config or PlasticityConfig()
        self._step_count: int = 0
        self._weight_history: List[float] = []  # Track mean weight over time

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, dt_ms: float = 1.0) -> None:
        """Apply the configured learning rule to all managed synapses."""
        rule = self.config.learning_rule

        for synapse in self.synapses:
            dw = 0.0

            if rule in (LearningRule.HEBBIAN, LearningRule.COMBINED):
                dw += self._hebbian_delta(synapse)

            if rule in (LearningRule.STDP, LearningRule.COMBINED):
                dw += self._stdp_delta(synapse)

            if dw != 0.0:
                new_w = synapse.config.weight + dw
                new_w = max(self.config.min_weight, min(self.config.max_weight, new_w))
                synapse.set_weight(new_w)

        # Homeostatic scaling (less frequent — every 100 steps)
        if rule in (LearningRule.HOMEOSTATIC, LearningRule.COMBINED):
            if self._step_count % 100 == 0:
                self._apply_homeostatic_scaling()

        self._step_count += 1
        self._record_mean_weight()

    def add_synapse(self, synapse: Synapse) -> None:
        """Dynamically add a new synapse to the managed set."""
        self.synapses.append(synapse)

    def remove_synapse(self, synapse_id: str) -> None:
        """Remove a synapse from management by its ID."""
        self.synapses = [s for s in self.synapses if s.synapse_id != synapse_id]

    @property
    def mean_weight(self) -> float:
        if not self.synapses:
            return 0.0
        return sum(s.config.weight for s in self.synapses) / len(self.synapses)

    @property
    def weight_history(self) -> List[float]:
        return list(self._weight_history)

    def __repr__(self) -> str:
        return (
            f"NeuroplasticityEngine(synapses={len(self.synapses)}, "
            f"rule={self.config.learning_rule.name}, "
            f"mean_weight={self.mean_weight:.4f})"
        )

    # ------------------------------------------------------------------
    # Learning rules
    # ------------------------------------------------------------------

    def _hebbian_delta(self, synapse: Synapse) -> float:
        """
        Hebbian weight update: Δw = η * pre_activity * post_activity − decay * w

        Activity is approximated by the neuron's recent firing rate (Hz).
        """
        pre_rate = synapse.pre_neuron.firing_rate_hz
        post_rate = synapse.post_neuron.firing_rate_hz
        eta = self.config.hebbian_rate
        decay = self.config.hebbian_decay

        delta = eta * pre_rate * post_rate - decay * synapse.config.weight
        return delta

    def _stdp_delta(self, synapse: Synapse) -> float:
        """
        Spike-Timing-Dependent Plasticity weight update.

        For each pre-spike, look at all post-spikes to compute Δw:
          If t_post > t_pre  (post fires after pre): LTP — Δw = A+ * exp(-Δt / τ+)
          If t_post < t_pre  (post fires before pre): LTD — Δw = -A- * exp(-Δt / τ-)

        Uses the recorded spike times from both neurons.
        """
        pre_spikes = synapse.pre_neuron.spike_times
        post_spikes = synapse.post_neuron.spike_times

        if not pre_spikes or not post_spikes:
            return 0.0

        delta = 0.0
        a_plus = self.config.stdp_a_plus
        a_minus = self.config.stdp_a_minus
        tau_plus = self.config.stdp_tau_plus_ms / 1000.0    # convert to seconds
        tau_minus = self.config.stdp_tau_minus_ms / 1000.0

        # Only consider the most recent spike from each neuron to keep it fast
        t_pre = pre_spikes[-1]
        t_post = post_spikes[-1]
        dt = t_post - t_pre  # seconds

        if dt > 0:
            # Post fires after pre → LTP
            delta += a_plus * math.exp(-dt / tau_plus)
        elif dt < 0:
            # Post fires before pre → LTD
            delta -= a_minus * math.exp(dt / tau_minus)  # dt is negative, exp positive

        return delta

    def _apply_homeostatic_scaling(self) -> None:
        """
        Synaptic scaling: globally normalise weights to keep average
        firing rates near the target.

        When neurons fire too fast → scale weights down.
        When neurons fire too slow → scale weights up.
        """
        # Gather unique neurons from all managed synapses
        neurons = {}
        for s in self.synapses:
            neurons[s.pre_neuron.neuron_id] = s.pre_neuron
            neurons[s.post_neuron.neuron_id] = s.post_neuron

        if not neurons:
            return

        mean_rate = sum(n.firing_rate_hz for n in neurons.values()) / len(neurons)
        target = self.config.target_firing_rate_hz
        rate_error = target - mean_rate  # positive → need more firing

        scale = 1.0 + self.config.homeostatic_rate * rate_error

        for synapse in self.synapses:
            new_w = synapse.config.weight * scale
            new_w = max(self.config.min_weight, min(self.config.max_weight, new_w))
            synapse.set_weight(new_w)

    def _record_mean_weight(self) -> None:
        """Store mean weight for trend analysis."""
        self._weight_history.append(self.mean_weight)
        if len(self._weight_history) > 1000:
            self._weight_history = self._weight_history[-1000:]
