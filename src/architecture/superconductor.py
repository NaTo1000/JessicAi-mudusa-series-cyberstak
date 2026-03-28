"""
Superconductor Layer Simulation
=================================
Models the 1-billion superconducting layers that provide computational
acceleration and heat efficiency in the JessicAI AGI architecture.

Each ``SuperconductorLayer`` tracks:
  - Simulated thermal load (Kelvin above critical temperature).
  - Computational throughput (operations per simulated tick).
  - Resistance state (superconducting ↔ normal).

The ``SuperconductorArray`` manages the full ensemble of layers using
a sliding-window design identical to the ``Tesseract`` so that the
conceptual billion-layer depth does not exhaust real memory.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


BILLION = 1_000_000_000

# Physical constants (illustrative values for simulation)
_CRITICAL_TEMP_K = 135.0       # High-temperature superconductor critical temp
_ROOM_TEMP_K = 293.15


class ConductionState(Enum):
    SUPERCONDUCTING = auto()   # zero resistance
    TRANSITIONING = auto()     # near critical temperature
    NORMAL = auto()            # resistive


@dataclass
class LayerMetrics:
    """Accumulated performance metrics for a single superconductor layer."""
    total_ops: int = 0
    total_heat_joules: float = 0.0
    peak_throughput: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class SuperconductorLayer:
    """
    A single superconducting computational layer.

    Attributes
    ----------
    logical_index:
        Absolute position within the billion-layer array.
    temperature_k:
        Current simulated temperature in Kelvin.
    throughput:
        Operations per simulated tick (arbitrary units).
    state:
        Current conduction state.
    """

    logical_index: int
    layer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    temperature_k: float = _CRITICAL_TEMP_K * 0.9   # starts cold
    throughput: float = 0.0
    state: ConductionState = ConductionState.SUPERCONDUCTING
    metrics: LayerMetrics = field(default_factory=LayerMetrics)
    _created_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Simulation tick
    # ------------------------------------------------------------------

    def tick(self, workload: float = 1.0) -> float:
        """
        Advance this layer by one simulation tick.

        Parameters
        ----------
        workload:
            Fraction of peak load (0.0–1.0).

        Returns
        -------
        float
            Operations completed this tick.
        """
        # Heat generated proportional to workload and resistance
        resistance = 0.0 if self.state == ConductionState.SUPERCONDUCTING else 1.0
        heat = workload * resistance * 0.01          # minimal heat when superconducting
        self.temperature_k = min(
            self.temperature_k + heat,
            _ROOM_TEMP_K,
        )
        # Passive cooling
        cooling = (self.temperature_k - (_CRITICAL_TEMP_K * 0.8)) * 0.005
        self.temperature_k = max(
            self.temperature_k - max(0.0, cooling),
            _CRITICAL_TEMP_K * 0.7,
        )

        self._update_state()

        # Throughput is maximal when superconducting, degraded otherwise
        multiplier = {
            ConductionState.SUPERCONDUCTING: 1.0,
            ConductionState.TRANSITIONING: 0.6,
            ConductionState.NORMAL: 0.2,
        }[self.state]
        ops = workload * 1_000_000 * multiplier      # 1 M ops/tick base
        self.throughput = ops

        # Update metrics
        self.metrics.total_ops += int(ops)
        self.metrics.total_heat_joules += heat
        self.metrics.peak_throughput = max(self.metrics.peak_throughput, ops)
        self.metrics.uptime_seconds = time.time() - self._created_at

        return ops

    def _update_state(self) -> None:
        """Transition between conduction states based on temperature."""
        tc = _CRITICAL_TEMP_K
        if self.temperature_k < tc * 0.95:
            self.state = ConductionState.SUPERCONDUCTING
        elif self.temperature_k < tc * 1.05:
            self.state = ConductionState.TRANSITIONING
        else:
            self.state = ConductionState.NORMAL

    def cool_down(self, delta_k: float = 10.0) -> None:
        """Actively reduce the layer temperature by *delta_k* Kelvin."""
        self.temperature_k = max(
            self.temperature_k - delta_k,
            _CRITICAL_TEMP_K * 0.7,
        )
        self._update_state()

    def status(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "logical_index": self.logical_index,
            "state": self.state.name,
            "temperature_k": round(self.temperature_k, 3),
            "throughput": self.throughput,
            "total_ops": self.metrics.total_ops,
        }


# ---------------------------------------------------------------------------
# SuperconductorArray
# ---------------------------------------------------------------------------

@dataclass
class ArrayConfig:
    """Configuration for the billion-layer superconductor array."""
    logical_layer_count: int = BILLION
    max_active_layers: int = 2_048
    default_workload: float = 0.85


class SuperconductorArray:
    """
    Manages the full 1-billion-layer superconducting array.

    Uses the same sliding-window eviction strategy as ``Tesseract``
    so only ``config.max_active_layers`` live in memory at once.
    """

    def __init__(self, config: Optional[ArrayConfig] = None) -> None:
        self.config = config or ArrayConfig()
        self.array_id = str(uuid.uuid4())
        self._active: Dict[int, SuperconductorLayer] = {}
        self._eviction_log: Dict[int, Dict[str, Any]] = {}
        self._cursor = 0

    # ------------------------------------------------------------------
    # Layer lifecycle
    # ------------------------------------------------------------------

    def _evict_oldest(self) -> None:
        if not self._active:
            return
        idx = min(self._active)
        layer = self._active.pop(idx)
        self._eviction_log[idx] = layer.status()

    def allocate_layer(self, logical_index: Optional[int] = None) -> SuperconductorLayer:
        """
        Allocate and return a ``SuperconductorLayer`` at *logical_index*.
        """
        if logical_index is None:
            logical_index = self._cursor
            self._cursor += 1

        if logical_index in self._active:
            return self._active[logical_index]

        if len(self._active) >= self.config.max_active_layers:
            self._evict_oldest()

        layer = SuperconductorLayer(logical_index=logical_index)
        self._active[logical_index] = layer
        return layer

    def get_layer(self, logical_index: int) -> Optional[SuperconductorLayer]:
        return self._active.get(logical_index)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def run_tick(self, workload: Optional[float] = None) -> float:
        """
        Execute one simulation tick across all active layers.

        Returns
        -------
        float
            Total operations completed across all active layers.
        """
        w = workload if workload is not None else self.config.default_workload
        total_ops = sum(layer.tick(w) for layer in self._active.values())
        return total_ops

    def cool_all(self, delta_k: float = 5.0) -> None:
        """Apply active cooling to every live layer."""
        for layer in self._active.values():
            layer.cool_down(delta_k)

    def bootstrap(self, count: int = 32) -> List[SuperconductorLayer]:
        """Quickly allocate *count* layers from index 0."""
        count = min(count, self.config.max_active_layers)
        return [self.allocate_layer(i) for i in range(count)]

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    @property
    def active_layer_count(self) -> int:
        return len(self._active)

    @property
    def aggregate_throughput(self) -> float:
        return sum(l.throughput for l in self._active.values())

    @property
    def superconducting_ratio(self) -> float:
        if not self._active:
            return 0.0
        sc = sum(
            1 for l in self._active.values()
            if l.state == ConductionState.SUPERCONDUCTING
        )
        return sc / len(self._active)

    @property
    def mean_temperature_k(self) -> float:
        if not self._active:
            return 0.0
        return sum(l.temperature_k for l in self._active.values()) / len(self._active)

    def status(self) -> Dict[str, Any]:
        return {
            "array_id": self.array_id,
            "logical_layers": self.config.logical_layer_count,
            "active_layers": self.active_layer_count,
            "aggregate_throughput": self.aggregate_throughput,
            "superconducting_ratio": round(self.superconducting_ratio, 4),
            "mean_temperature_k": round(self.mean_temperature_k, 3),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_billion_layer_array(
    max_active_layers: int = 2_048,
    default_workload: float = 0.85,
) -> SuperconductorArray:
    """
    Return a fully configured 1-billion-layer ``SuperconductorArray``.

    Parameters
    ----------
    max_active_layers:
        How many layers are kept live in memory at once.
    default_workload:
        Default fraction of peak load per simulation tick.
    """
    cfg = ArrayConfig(
        logical_layer_count=BILLION,
        max_active_layers=max_active_layers,
        default_workload=default_workload,
    )
    return SuperconductorArray(config=cfg)
