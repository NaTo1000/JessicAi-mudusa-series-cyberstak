"""
visualization.py — Real-time neural visualization and performance metrics.

Provides:
- ASCII art firing pattern display (terminal-friendly)
- Layer activation bar charts
- Synaptic weight heatmap (text-based)
- Performance metric tables (latency, firing rate, plasticity trends)
- Exportable metrics report (plain text)
"""

import math
import time
from typing import List, Dict, Optional, Any

from .neuron import Neuron, NeuronState
from .synapse import Synapse, SynapseType
from .neural_network import QuantumNeuralBrain


class NeuralVisualizer:
    """
    Terminal-based visualiser for the Quantum Neural Brain.

    Renders live firing patterns, activation levels, and hardware metrics
    without any external dependencies (pure Python, no matplotlib required).
    """

    BAR_WIDTH = 20       # Width of activation bar chart bars
    WEIGHT_COLS = 10     # Number of synapse columns in the weight heatmap

    def __init__(self, brain: QuantumNeuralBrain):
        self.brain = brain
        self._frame_count: int = 0
        self._start_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # High-level render
    # ------------------------------------------------------------------

    def render(self, step_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Render a full visualisation frame as a multi-line string.

        Includes firing map, output activations, hardware status, and metrics.
        """
        self._frame_count += 1
        lines = []

        lines.append(self._header())
        lines.append("")
        lines.append(self._firing_map(step_result))
        lines.append("")
        lines.append(self._output_activations(step_result))
        lines.append("")
        lines.append(self._synapse_weight_summary())
        lines.append("")
        lines.append(self._hardware_status())
        lines.append("")
        lines.append(self._performance_metrics(step_result))
        lines.append(self._footer())

        return "\n".join(lines)

    def render_compact(self, step_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Compact single-line render suitable for continuous simulation output.
        """
        if step_result is None:
            return ""
        step = step_result.get("step", 0)
        spikes = step_result.get("spike_count", 0)
        decision = step_result.get("decision")
        weight = step_result.get("plasticity_mean_weight", 0.0)
        activations = step_result.get("output_activations", [])
        act_str = " ".join(f"{a:.2f}" for a in activations)
        return (
            f"[{step:05d}] spikes={spikes:3d} | "
            f"outputs=[{act_str}] | "
            f"decision={str(decision):>4s} | "
            f"w̄={weight:.4f}"
        )

    def performance_report(self) -> str:
        """Generate a summary performance report."""
        elapsed = time.monotonic() - self._start_time
        metrics = self.brain.metrics
        total_steps = metrics.get("total_steps", 0)
        total_spikes = metrics.get("total_spikes", 0)
        history = metrics.get("output_activation_history", [])

        lines = [
            "=" * 60,
            "  Quantum Neural Brain — Performance Report",
            "=" * 60,
            f"  Elapsed time      : {elapsed:.2f}s",
            f"  Total steps       : {total_steps}",
            f"  Total spikes      : {total_spikes}",
            f"  Steps per second  : {total_steps / elapsed:.1f}" if elapsed > 0 else "  Steps per second  : N/A",
            f"  Mean firing rate  : {total_spikes / (total_steps or 1):.2f} spikes/step",
            "",
            "  Plasticity (mean weight trend):",
            self._weight_trend_chart(),
            "",
            "  Output activation distribution:",
            self._output_distribution(history),
            "",
            "  Hardware cluster status:",
        ]
        for node_status in self.brain.hardware.cluster_status():
            lines.append(
                f"    [{node_status['node_type']:12s}] {node_status['node_id']:8s} "
                f"| {node_status['neurons']:3d} neurons "
                f"| temp={node_status['temperature_c']:.1f}°C "
                f"| {'THROTTLED' if node_status['throttled'] else 'OK':9s}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Individual panel renderers
    # ------------------------------------------------------------------

    def _header(self) -> str:
        elapsed = time.monotonic() - self._start_time
        return (
            f"╔══════════════════════════════════════════════════════╗\n"
            f"║   ⚛  Quantum Neural Brain  —  t={elapsed:6.2f}s  frame={self._frame_count:4d} ║\n"
            f"╚══════════════════════════════════════════════════════╝"
        )

    def _footer(self) -> str:
        return "─" * 60

    def _firing_map(self, step_result: Optional[Dict[str, Any]]) -> str:
        """
        Render a grid showing which neurons fired this step.

        ● = fired this step
        ○ = active (membrane above resting)
        · = resting
        R = refractory
        """
        fired_set = set(step_result.get("fired_neurons", [])) if step_result else set()
        lines = ["  Neuronal firing map:"]

        for layer in self.brain.layers:
            tag = ""
            if layer.is_input:
                tag = " [INPUT] "
            elif layer.is_output:
                tag = " [OUTPUT]"
            row = f"  {layer.name:12s}{tag}│ "
            for neuron in layer.neurons:
                if neuron.neuron_id in fired_set:
                    row += "● "
                elif neuron.state == NeuronState.REFRACTORY:
                    row += "R "
                elif neuron.state in (NeuronState.INTEGRATING, NeuronState.QUANTUM_SUPERPOSITION):
                    row += "○ "
                else:
                    row += "· "
            lines.append(row)
        return "\n".join(lines)

    def _output_activations(self, step_result: Optional[Dict[str, Any]]) -> str:
        """Bar chart of output layer activations."""
        activations = []
        if step_result:
            activations = step_result.get("output_activations", [])
        if not activations:
            output_layer = self.brain.get_layer("output")
            if output_layer:
                activations = [0.0] * output_layer.size

        lines = ["  Output activations:"]
        for i, val in enumerate(activations):
            bar_len = int(val * self.BAR_WIDTH)
            bar = "█" * bar_len + "░" * (self.BAR_WIDTH - bar_len)
            lines.append(f"  out_{i} │{bar}│ {val:.3f}")
        return "\n".join(lines)

    def _synapse_weight_summary(self) -> str:
        """Summarise the distribution of synaptic weights."""
        synapses = self.brain.synapses
        if not synapses:
            return "  Synapses: (none)"

        weights = [s.config.weight for s in synapses]
        exc_w = [s.config.weight for s in synapses if s.synapse_type == SynapseType.EXCITATORY]
        inh_w = [s.config.weight for s in synapses if s.synapse_type == SynapseType.INHIBITORY]

        mean_w = sum(weights) / len(weights)
        min_w = min(weights)
        max_w = max(weights)

        lines = [
            f"  Synaptic weights ({len(synapses)} total):",
            f"  Mean={mean_w:.3f}  Min={min_w:.3f}  Max={max_w:.3f}",
            f"  Excitatory: {len(exc_w):3d}  Inhibitory: {len(inh_w):3d}",
        ]

        # Mini histogram (10 buckets)
        bucket_count = 10
        bucket_size = (max_w - min_w + 1e-9) / bucket_count
        buckets = [0] * bucket_count
        for w in weights:
            idx = min(bucket_count - 1, int((w - min_w) / bucket_size))
            buckets[idx] += 1
        max_bucket = max(buckets) or 1
        hist_line = "  Weight dist: │"
        for b in buckets:
            height = int(b / max_bucket * 8)
            hist_line += "▁▂▃▄▅▆▇█"[height - 1] if height > 0 else " "
        hist_line += "│"
        lines.append(hist_line)
        return "\n".join(lines)

    def _hardware_status(self) -> str:
        """Compact hardware cluster status."""
        lines = ["  Hardware nodes:"]
        for status in self.brain.hardware.cluster_status():
            throttle_flag = " ⚠ THROTTLED" if status["throttled"] else ""
            lines.append(
                f"  {status['node_id']:8s} [{status['node_type']:12s}] "
                f"neurons={status['neurons']:2d} "
                f"T={status['temperature_c']:.0f}°C"
                f"{throttle_flag}"
            )
        return "\n".join(lines)

    def _performance_metrics(self, step_result: Optional[Dict[str, Any]]) -> str:
        """Key performance indicators from the current step."""
        if step_result is None:
            return "  Metrics: (no data)"

        plasticity = self.brain.plasticity_engine
        quantum = self.brain.quantum_layer

        lines = ["  Performance metrics:"]
        lines.append(f"  Step            : {step_result.get('step', 0)}")
        lines.append(f"  Spike count     : {step_result.get('spike_count', 0)}")
        lines.append(f"  Decision        : {step_result.get('decision')}")
        lines.append(f"  Mean syn weight : {step_result.get('plasticity_mean_weight', 0.0):.4f}")
        if quantum is not None:
            lines.append(f"  Quantum collapses (total): {quantum.total_quantum_collapses}")
            lines.append(f"  Entangled firings (total): {quantum.entangled_firing_events}")
        return "\n".join(lines)

    def _weight_trend_chart(self) -> str:
        """ASCII sparkline of mean weight history."""
        history = self.brain.plasticity_engine.weight_history
        if not history:
            return "    (no data)"
        # Downsample to 40 points
        step = max(1, len(history) // 40)
        samples = history[::step][-40:]
        if not samples:
            return "    (no data)"
        min_h = min(samples)
        max_h = max(samples)
        span = max_h - min_h or 1e-9
        blocks = " ▁▂▃▄▅▆▇█"
        line = "    │"
        for val in samples:
            idx = int((val - min_h) / span * 8)
            line += blocks[min(8, idx)]
        line += f"│ {samples[-1]:.4f}"
        return line

    def _output_distribution(self, history: List[List[float]]) -> str:
        """Summary of output activation over recent history."""
        if not history:
            return "    (no data)"
        n_outputs = len(history[0]) if history else 0
        lines = []
        for i in range(n_outputs):
            vals = [h[i] for h in history if i < len(h)]
            if not vals:
                continue
            mean_val = sum(vals) / len(vals)
            lines.append(f"    out_{i}: mean={mean_val:.3f}")
        return "\n".join(lines) if lines else "    (no data)"
