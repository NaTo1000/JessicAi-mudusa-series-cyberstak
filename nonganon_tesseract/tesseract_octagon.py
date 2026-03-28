"""
Tesseract-Octagon Framework
============================

Implements a multi-layer, topologically optimised structure within
a tesseract (4-D hypercube) framework.  Eight octagonal layers are
arranged so that data can flow non-linearly between any pair of
layers, enabling triple-clustering capabilities without the
constraint of a strictly sequential pipeline.

Architecture overview
---------------------
A tesseract has 16 vertices.  We map those vertices onto 8 pairs
(one pair per octagonal layer).  Each layer owns:

* an ``index`` (0-7, corresponding to one octagon vertex)
* a ``state`` dictionary for holding layer-local data
* a bidirectional adjacency list that encodes the topological
  connections to the other seven layers

Non-linear flow is achieved by the :meth:`route` method, which
consults the topology map and forwards a payload to *all* connected
layers rather than the single "next" layer in a chain.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCTAGON_LAYERS: int = 8

# Tesseract edge connectivity expressed as an adjacency list.
# Each of the 8 layer indices is adjacent to exactly 4 others, mirroring
# the structure of a tesseract projected to 8 nodes.
_TESSERACT_TOPOLOGY: Dict[int, List[int]] = {
    0: [1, 2, 4, 6],
    1: [0, 3, 5, 7],
    2: [0, 3, 4, 6],
    3: [1, 2, 5, 7],
    4: [0, 2, 5, 6],
    5: [1, 3, 4, 7],
    6: [0, 2, 4, 7],
    7: [1, 3, 5, 6],
}


# ---------------------------------------------------------------------------
# OctagonLayer
# ---------------------------------------------------------------------------


class OctagonLayer:
    """A single layer within the TesseractOctagon framework.

    Parameters
    ----------
    index:
        Zero-based layer index (0–7).
    label:
        Human-readable label for this layer (e.g. ``"input"``,
        ``"cluster-A"``).
    """

    def __init__(self, index: int, label: str = "") -> None:
        if not (0 <= index < OCTAGON_LAYERS):
            raise ValueError(
                f"Layer index must be in [0, {OCTAGON_LAYERS - 1}], got {index!r}."
            )
        self.index = index
        self.label = label or f"layer-{index}"
        self.state: Dict[str, Any] = {}
        self._adjacent: List[int] = list(_TESSERACT_TOPOLOGY[index])

    # ------------------------------------------------------------------

    @property
    def adjacent_indices(self) -> List[int]:
        """Layer indices that are topologically adjacent to this one."""
        return list(self._adjacent)

    # ------------------------------------------------------------------

    def update_state(self, key: str, value: Any) -> None:
        """Store a value in the layer's local state dictionary."""
        self.state[key] = value

    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """Return a SHA-256 fingerprint of the layer's current state."""
        payload = str(sorted(self.state.items())).encode()
        return hashlib.sha256(payload).hexdigest()

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"OctagonLayer(index={self.index}, label={self.label!r}, "
            f"adjacent={self._adjacent})"
        )


# ---------------------------------------------------------------------------
# TesseractOctagon
# ---------------------------------------------------------------------------


class TesseractOctagon:
    """Multi-layer, topologically optimised tesseract-octagon structure.

    The framework maintains exactly :data:`OCTAGON_LAYERS` (8) layers
    arranged according to the :data:`_TESSERACT_TOPOLOGY` adjacency map.
    Data can be pushed to any single layer and will automatically
    propagate non-linearly to all adjacent layers.

    Parameters
    ----------
    labels:
        Optional sequence of eight label strings, one per layer.
        If omitted, labels default to ``"layer-0"`` … ``"layer-7"``.

    Examples
    --------
    >>> to = TesseractOctagon()
    >>> to.push(layer_index=0, key="signal", value=42)
    >>> adjacents = to.route(layer_index=0, key="signal")
    """

    def __init__(self, labels: Optional[List[str]] = None) -> None:
        if labels is not None and len(labels) != OCTAGON_LAYERS:
            raise ValueError(
                f"Expected exactly {OCTAGON_LAYERS} labels, got {len(labels)}."
            )
        _labels = labels or [
            "input",
            "cluster-alpha",
            "cluster-beta",
            "cluster-gamma",
            "inference",
            "decision",
            "output",
            "vault",
        ]
        self.layers: List[OctagonLayer] = [
            OctagonLayer(index=i, label=_labels[i])
            for i in range(OCTAGON_LAYERS)
        ]
        logger.debug("TesseractOctagon initialised with %d layers.", OCTAGON_LAYERS)

    # ------------------------------------------------------------------

    def layer(self, index: int) -> OctagonLayer:
        """Return the :class:`OctagonLayer` at *index*."""
        if not (0 <= index < OCTAGON_LAYERS):
            raise IndexError(
                f"Layer index must be in [0, {OCTAGON_LAYERS - 1}], got {index!r}."
            )
        return self.layers[index]

    # ------------------------------------------------------------------

    def push(self, layer_index: int, key: str, value: Any) -> None:
        """Write *value* under *key* into the layer at *layer_index*."""
        self.layer(layer_index).update_state(key, value)
        logger.debug("Pushed %r=%r to layer %d.", key, value, layer_index)

    # ------------------------------------------------------------------

    def route(self, layer_index: int, key: str) -> Dict[int, Any]:
        """Propagate the value stored under *key* in layer *layer_index*
        to every adjacent layer and return a mapping of
        ``{target_layer_index: value}``.

        This is the core non-linear data-flow mechanism.
        """
        src = self.layer(layer_index)
        value = src.state.get(key)
        results: Dict[int, Any] = {}
        for adj_idx in src.adjacent_indices:
            self.layers[adj_idx].update_state(key, value)
            results[adj_idx] = value
        logger.debug(
            "Routed %r from layer %d to layers %s.",
            key,
            layer_index,
            list(results.keys()),
        )
        return results

    # ------------------------------------------------------------------

    def broadcast(self, key: str, value: Any) -> None:
        """Write *value* under *key* into **all** layers simultaneously."""
        for lyr in self.layers:
            lyr.update_state(key, value)
        logger.debug("Broadcast %r=%r to all %d layers.", key, value, OCTAGON_LAYERS)

    # ------------------------------------------------------------------

    def collect_states(self) -> Dict[int, Dict[str, Any]]:
        """Return a snapshot of every layer's state dictionary."""
        return {lyr.index: dict(lyr.state) for lyr in self.layers}

    # ------------------------------------------------------------------

    def topology_summary(self) -> str:
        """Return a human-readable summary of the tesseract topology."""
        lines = ["TesseractOctagon topology:"]
        for lyr in self.layers:
            adj_labels = [self.layers[i].label for i in lyr.adjacent_indices]
            lines.append(
                f"  [{lyr.index}] {lyr.label!r:20s} ↔ {adj_labels}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def all_pairs(self) -> List[tuple]:
        """Return all unique layer-pair combinations (for diagnostics)."""
        return list(itertools.combinations(range(OCTAGON_LAYERS), 2))

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"TesseractOctagon(layers={OCTAGON_LAYERS})"
