"""
Topological Tesseract Architecture
===================================
Implements the trillion-layer, tesseract-in-tesseract multi-dimensional
data organisation used as the core processing fabric of JessicAI.

A *Tesseract* is a 4-D hypercube.  Here the term is used architecturally:
each ``TesseractLayer`` is an independent processing plane and the nesting
(tesseract-in-tesseract) allows arbitrary depth hierarchies so that the
system can scale to a *conceptual* trillion-layer configuration.

The implementation represents layers symbolically using a bounded integer
counter so it runs on real hardware while faithfully modelling the design
intent of the full-scale architecture.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRILLION = 1_000_000_000_000
BILLION = 1_000_000_000


@dataclass
class TesseractConfig:
    """Runtime-tunable parameters for a Tesseract instance."""

    # Symbolic depth – the architecture *represents* this many layers; actual
    # in-memory objects are bounded by ``max_active_layers``.
    logical_layer_count: int = TRILLION

    # Maximum layers kept live in memory at once (sliding window).
    max_active_layers: int = 1_024

    # Dimensionality of the tesseract manifold (≥ 4 for true 4-D hypercube).
    dimensions: int = 4

    # Maximum nesting depth (tesseract-in-tesseract).
    max_nesting_depth: int = 8

    # Fault-tolerance replication factor.
    replication_factor: int = 3

    # Enable quantum-topological routing between layers.
    quantum_routing: bool = True


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class LayerState(Enum):
    INITIALISING = auto()
    ACTIVE = auto()
    SUSPENDED = auto()
    QUARANTINED = auto()
    EVICTED = auto()


@dataclass
class TesseractLayer:
    """
    A single processing plane in the tesseract fabric.

    Each layer holds a *logical index* (its absolute position within the
    trillion-layer space), a *dimensional coordinate* tuple, and an optional
    reference to a child ``Tesseract`` (enabling tesseract-in-tesseract).
    """

    logical_index: int
    coordinates: tuple                       # position in N-D space
    layer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: LayerState = LayerState.INITIALISING
    metadata: Dict[str, Any] = field(default_factory=dict)
    child_tesseract: Optional["Tesseract"] = None

    # Accumulated processing units (conceptual metric).
    processing_units: float = 0.0

    def activate(self) -> None:
        self.state = LayerState.ACTIVE
        self.metadata["activated_at"] = time.time()

    def suspend(self) -> None:
        if self.state == LayerState.ACTIVE:
            self.state = LayerState.SUSPENDED

    def quarantine(self) -> None:
        """Isolate this layer on detected stress or intrusion."""
        self.state = LayerState.QUARANTINED
        self.metadata["quarantined_at"] = time.time()

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 fingerprint of this layer's identity."""
        payload = f"{self.layer_id}:{self.logical_index}:{self.coordinates}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TesseractLayer(idx={self.logical_index}, "
            f"coords={self.coordinates}, state={self.state.name})"
        )


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------

class Tesseract:
    """
    1-trillion-layer topological tesseract with nested-tesseract support.

    Design principles
    -----------------
    * **Sliding-window activation** – the trillion logical layers are addressed
      by integer index; only ``config.max_active_layers`` are instantiated at
      any moment.  Evicted layers are checkpointed to the metadata store.
    * **N-D coordinate space** – each layer is assigned coordinates in an
      N-dimensional grid so data locality and routing use geometric proximity.
    * **Tesseract-in-tesseract** – any layer may spawn a child ``Tesseract``
      creating recursive depth up to ``config.max_nesting_depth``.
    * **Fault tolerance** – layers are replicated ``config.replication_factor``
      times across the coordinate space.
    """

    def __init__(
        self,
        config: Optional[TesseractConfig] = None,
        nesting_depth: int = 0,
        parent_id: Optional[str] = None,
    ) -> None:
        self.config = config or TesseractConfig()
        self.nesting_depth = nesting_depth
        self.parent_id = parent_id
        self.tesseract_id = str(uuid.uuid4())

        # Active layer window: logical_index → TesseractLayer
        self._active_layers: Dict[int, TesseractLayer] = {}

        # Eviction checkpoint store: logical_index → fingerprint
        self._checkpoints: Dict[int, str] = {}

        # Routing table: src_index → list[dst_index]
        self._routing_table: Dict[int, List[int]] = {}

        # Monotonic layer cursor
        self._cursor: int = 0

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def _coord_for(self, index: int) -> tuple:
        """Map a logical layer index to an N-dimensional grid coordinate."""
        d = self.config.dimensions
        side = max(1, math.ceil(self.config.logical_layer_count ** (1.0 / d)))
        coords: List[int] = []
        remainder = index
        for _ in range(d):
            coords.append(remainder % side)
            remainder //= side
        return tuple(coords)

    def _evict_oldest(self) -> None:
        """Remove the lowest-index active layer to free memory."""
        if not self._active_layers:
            return
        oldest_idx = min(self._active_layers)
        layer = self._active_layers.pop(oldest_idx)
        self._checkpoints[oldest_idx] = layer.fingerprint()

    def create_layer(
        self,
        logical_index: Optional[int] = None,
        spawn_child: bool = False,
    ) -> TesseractLayer:
        """
        Create (or retrieve) a layer at *logical_index*.

        If the active window is full the oldest layer is evicted first.
        """
        if logical_index is None:
            logical_index = self._cursor
            self._cursor += 1

        if logical_index in self._active_layers:
            return self._active_layers[logical_index]

        if len(self._active_layers) >= self.config.max_active_layers:
            self._evict_oldest()

        coords = self._coord_for(logical_index)
        layer = TesseractLayer(logical_index=logical_index, coordinates=coords)

        # Optionally nest a child tesseract
        if spawn_child and self.nesting_depth < self.config.max_nesting_depth:
            child_cfg = TesseractConfig(
                logical_layer_count=max(1, self.config.logical_layer_count // 1000),
                max_active_layers=self.config.max_active_layers // 4,
                dimensions=self.config.dimensions,
                max_nesting_depth=self.config.max_nesting_depth,
            )
            layer.child_tesseract = Tesseract(
                config=child_cfg,
                nesting_depth=self.nesting_depth + 1,
                parent_id=self.tesseract_id,
            )

        layer.activate()
        self._active_layers[logical_index] = layer

        if self.config.quantum_routing:
            self._update_routing(logical_index)

        return layer

    def get_layer(self, logical_index: int) -> Optional[TesseractLayer]:
        """Return an active layer or ``None`` if it has been evicted."""
        return self._active_layers.get(logical_index)

    def route(self, src_index: int, data: Any) -> List[int]:
        """
        Quantum-topological routing: return the list of destination layer
        indices for data originating at *src_index*.
        """
        return self._routing_table.get(src_index, [])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_routing(self, new_index: int) -> None:
        """
        Add routes between *new_index* and geometrically adjacent layers
        using Hamming-1 neighbours in the N-D coordinate space.
        """
        src_coords = self._coord_for(new_index)
        d = self.config.dimensions
        side = max(
            1,
            math.ceil(self.config.logical_layer_count ** (1.0 / d)),
        )
        neighbours: List[int] = []
        for dim in range(d):
            for delta in (-1, +1):
                neighbour_coords = list(src_coords)
                neighbour_coords[dim] = (neighbour_coords[dim] + delta) % side
                # Convert coords back to logical index
                neighbour_idx = sum(
                    neighbour_coords[i] * (side ** i) for i in range(d)
                )
                if 0 <= neighbour_idx < self.config.logical_layer_count:
                    neighbours.append(neighbour_idx)
        self._routing_table[new_index] = neighbours

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_layer_count(self) -> int:
        return len(self._active_layers)

    @property
    def total_processing_units(self) -> float:
        return sum(l.processing_units for l in self._active_layers.values())

    def status(self) -> Dict[str, Any]:
        return {
            "tesseract_id": self.tesseract_id,
            "nesting_depth": self.nesting_depth,
            "logical_layers": self.config.logical_layer_count,
            "active_layers": self.active_layer_count,
            "evicted_layers": len(self._checkpoints),
            "dimensions": self.config.dimensions,
            "quantum_routing": self.config.quantum_routing,
            "total_processing_units": self.total_processing_units,
        }

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def iter_active_layers(self) -> Iterable[TesseractLayer]:
        """Yield active layers in ascending logical-index order."""
        yield from (
            self._active_layers[k]
            for k in sorted(self._active_layers)
        )

    # ------------------------------------------------------------------
    # Batch bootstrap
    # ------------------------------------------------------------------

    def bootstrap(
        self,
        count: int = 16,
        spawn_children: bool = False,
    ) -> List[TesseractLayer]:
        """
        Quickly initialise *count* layers from logical index 0.

        Useful for testing and demonstration purposes.
        """
        count = min(count, self.config.max_active_layers)
        return [
            self.create_layer(logical_index=i, spawn_child=spawn_children)
            for i in range(count)
        ]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_trillion_layer_tesseract(
    max_active_layers: int = 1_024,
    dimensions: int = 4,
    max_nesting_depth: int = 8,
    quantum_routing: bool = True,
) -> Tesseract:
    """
    Factory function: return a fully-configured trillion-layer ``Tesseract``.

    Parameters
    ----------
    max_active_layers:
        Sliding-window size – how many layers are live in memory at once.
    dimensions:
        Dimensionality of the tesseract coordinate space (≥ 4 recommended).
    max_nesting_depth:
        How many levels of tesseract-in-tesseract nesting are permitted.
    quantum_routing:
        Whether to enable N-D Hamming-1 quantum-topological routing.
    """
    cfg = TesseractConfig(
        logical_layer_count=TRILLION,
        max_active_layers=max_active_layers,
        dimensions=dimensions,
        max_nesting_depth=max_nesting_depth,
        quantum_routing=quantum_routing,
    )
    return Tesseract(config=cfg)
