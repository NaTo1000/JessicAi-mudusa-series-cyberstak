"""
Metrics Visualizer
==================
ASCII-based benchmarking visualizer for the Quantum Quad-Brain array.
Renders histograms, bar charts, and sparklines to the terminal without
requiring any external plotting libraries.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


_BAR_CHARS = "▏▎▍▌▋▊▉█"
_SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
MAX_NVME_THROUGHPUT_GBPS = 28.0   # 4× PCIe Gen 4 NVMe theoretical peak


def _sparkline(values: Sequence[float], width: int = 40) -> str:
    """Convert a sequence of floats into a single-row sparkline string."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    result = []
    for v in values[-width:]:
        idx = int((v - lo) / span * (len(_SPARKLINE_CHARS) - 1))
        result.append(_SPARKLINE_CHARS[idx])
    return "".join(result)


def _hbar(value: float, max_value: float, width: int = 30) -> str:
    """Render a single horizontal bar scaled to *width* characters."""
    if max_value == 0:
        return " " * width
    filled = min(width, int(round(value / max_value * width)))
    return "█" * filled + "░" * (width - filled)


class MetricsVisualizer:
    """
    Terminal visualizer for array performance metrics.

    Usage
    -----
    >>> viz = MetricsVisualizer()
    >>> print(viz.render_dashboard(quad_brain, monitor))
    """

    LINE_WIDTH = 72

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_dashboard(self, quad_brain: Any, monitor: Any) -> str:
        """
        Render a full-page ASCII dashboard.

        Parameters
        ----------
        quad_brain : QuantumQuadBrain
            The running compute array.
        monitor : PerformanceMonitor
            The associated performance monitor.
        """
        system = quad_brain.system_summary()
        perf = monitor.snapshot()

        sections = [
            self._header("QUANTUM QUAD-BRAIN · PERFORMANCE DASHBOARD"),
            self._uptime_section(perf),
            self._cluster_section(system),
            self._storage_section(system),
            self._cache_section(system),
            self._brain_section(system),
            self._energy_section(perf, monitor),
            self._footer(),
        ]
        return "\n".join(sections)

    def render_brain_comparison(self, brain_results: Dict[str, Any]) -> str:
        """Render a side-by-side comparison of the four brain results."""
        lines = [self._header("QUAD-BRAIN CONSENSUS VOTE")]
        if "error" in brain_results:
            lines.append(f"  ERROR: {brain_results['error']}")
        else:
            votes = brain_results.get("brain_votes", {})
            max_conf = max((v["confidence"] for v in votes.values()), default=1.0)
            for bid, vote in votes.items():
                bar = _hbar(vote["confidence"], max_conf, width=24)
                lines.append(
                    f"  {bid:8s}  class={vote['class']:4d}  "
                    f"[{bar}]  {vote['confidence']:.4f}"
                )
            lines.append("")
            lines.append(
                f"  CONSENSUS  class={brain_results.get('consensus_class', '?'):4}  "
                f"confidence={brain_results.get('consensus_confidence', 0):.4f}"
            )
        lines.append(self._divider())
        return "\n".join(lines)

    def render_sparklines(self, monitor: Any, metrics: Optional[List[str]] = None) -> str:
        """Render sparkline time-series for selected metrics."""
        metrics = metrics or ["cpu_utilisation", "inference_latency_ms", "cache_hit_ratio"]
        lines = [self._header("METRIC TRENDS (LAST 40 SAMPLES)")]
        history = list(monitor._history)
        for metric in metrics:
            values = [getattr(s, metric, 0.0) for s in history]
            spark = _sparkline(values)
            latest = values[-1] if values else 0.0
            lines.append(f"  {metric:28s}  {spark}  {latest:.3f}")
        lines.append(self._divider())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _header(self, title: str) -> str:
        pad = max(0, self.LINE_WIDTH - len(title) - 4)
        left = pad // 2
        right = pad - left
        return (
            f"\n{'═' * self.LINE_WIDTH}\n"
            f"  {'─' * left} {title} {'─' * right}\n"
            f"{'═' * self.LINE_WIDTH}"
        )

    def _divider(self) -> str:
        return "─" * self.LINE_WIDTH

    def _footer(self) -> str:
        return f"\n{'═' * self.LINE_WIDTH}\n"

    def _uptime_section(self, perf: Dict[str, Any]) -> str:
        curr = perf.get("current", {})
        return (
            f"\n  Uptime: {perf.get('uptime_s', 0):>8.1f} s    "
            f"Samples: {perf.get('samples_collected', 0):>5d}    "
            f"CPU: {curr.get('cpu_utilisation', 0):>5.1f}%    "
            f"RAM: {curr.get('ram_utilisation', 0):>5.1f}%"
        )

    def _cluster_section(self, system: Dict[str, Any]) -> str:
        cluster = system.get("cluster", {})
        total = cluster.get("total_nodes", 0)
        online = cluster.get("online_nodes", 0)
        avail = cluster.get("available_nodes", 0)
        bar = _hbar(online, total or 1, width=20)
        return (
            f"\n  CM4 CLUSTER\n"
            f"  Nodes online:   [{bar}]  {online}/{total}\n"
            f"  Available:      {avail:>3d}    "
            f"Tasks done: {cluster.get('total_tasks_completed', 0):>6d}\n"
            f"  Avg CPU: {cluster.get('avg_cpu_utilisation', 0):>5.1f}%   "
            f"Avg RAM: {cluster.get('avg_ram_utilisation', 0):>5.1f}%"
        )

    def _storage_section(self, system: Dict[str, Any]) -> str:
        storage = system.get("storage", {})
        tp = storage.get("throughput_gbps", 0.0)
        bar = _hbar(tp, MAX_NVME_THROUGHPUT_GBPS, width=20)   # 4× PCIe Gen4 NVMe max
        return (
            f"\n  NVMe STORAGE\n"
            f"  Throughput:  [{bar}]  {tp:.2f} GB/s\n"
            f"  Read ops:  {storage.get('total_reads', 0):>6d}    "
            f"Write ops: {storage.get('total_writes', 0):>6d}\n"
            f"  Healthy devices: {storage.get('healthy_devices', 0)}/{storage.get('devices', 0)}    "
            f"Errors: {storage.get('errors', 0):>4d}"
        )

    def _cache_section(self, system: Dict[str, Any]) -> str:
        cache = system.get("cache", {})
        hr = cache.get("hit_ratio", 0.0)
        bar = _hbar(hr, 1.0, width=20)
        util = cache.get("utilisation_pct", 0.0)
        util_bar = _hbar(util, 100.0, width=20)
        return (
            f"\n  DRAM CACHE\n"
            f"  Hit ratio:   [{bar}]  {hr:.4f}\n"
            f"  Utilisation: [{util_bar}]  {util:.1f}%\n"
            f"  Entries: {cache.get('entries', 0):>6d}    "
            f"Capacity: {cache.get('capacity_gb', 0):.1f} GB"
        )

    def _brain_section(self, system: Dict[str, Any]) -> str:
        brains = system.get("brains", {})
        lines = ["\n  QUANTUM BRAINS"]
        for bid, info in brains.items():
            latency = info.get("avg_latency_ms", 0.0)
            n = info.get("n_qubits", 0)
            runs = info.get("inferences_run", 0)
            bar = _hbar(latency, 5000.0, width=16)
            lines.append(
                f"  {bid:8s}  qubits={n:>2d}  runs={runs:>5d}  "
                f"latency=[{bar}] {latency:>8.2f} ms"
            )
        return "\n".join(lines)

    def _energy_section(self, perf: Dict[str, Any], monitor: Any) -> str:
        curr = perf.get("current", {})
        power = curr.get("power_watts", 0.0)
        eff = perf.get("energy_efficiency", 0.0)
        return (
            f"\n  ENERGY\n"
            f"  Estimated power: {power:>6.1f} W    "
            f"Efficiency: {eff:.4f} inferences/W"
        )
