"""
topological_vertex.py
---------------------
4D vertex mechanics for higher-dimensional data flows.

Each vertex lives in a 4-dimensional space and carries a complex quantum amplitude.
Edges connect vertices and carry entangled correlation weights.  The resulting
graph forms a topological manifold that can be dynamically scaled (infinite
scalability) by adding or removing vertices/edges at runtime.
"""

from __future__ import annotations

import math
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np


class Vertex4D:
    """A single vertex in 4D topological space."""

    def __init__(self, vid: int, coords: Optional[np.ndarray] = None) -> None:
        self.vid = vid
        # 4D coordinates (w, x, y, z)
        if coords is not None:
            assert coords.shape == (4,), "Vertex4D requires 4D coordinate array"
            self.coords: np.ndarray = coords.astype(float)
        else:
            self.coords = np.random.default_rng(vid).standard_normal(4)
        # Complex quantum amplitude associated with this vertex
        rng = np.random.default_rng(vid + 100)
        self.amplitude: complex = complex(rng.standard_normal(), rng.standard_normal())
        self._normalise_amplitude()

    def _normalise_amplitude(self) -> None:
        mag = abs(self.amplitude)
        if mag > 0:
            self.amplitude /= mag

    def rotate_4d(self, plane: Tuple[int, int], angle: float) -> None:
        """
        Apply a 4D rotation in the specified plane (e.g. (0,1) for the wx-plane).
        """
        i, j = plane
        assert 0 <= i < 4 and 0 <= j < 4 and i != j
        c, s = math.cos(angle), math.sin(angle)
        new_coords = self.coords.copy()
        new_coords[i] = c * self.coords[i] - s * self.coords[j]
        new_coords[j] = s * self.coords[i] + c * self.coords[j]
        self.coords = new_coords
        # also rotate amplitude phase
        self.amplitude *= np.exp(1j * angle)

    def distance_to(self, other: "Vertex4D") -> float:
        """Euclidean distance in 4D space."""
        return float(np.linalg.norm(self.coords - other.coords))

    def __repr__(self) -> str:
        return (
            f"Vertex4D(id={self.vid}, "
            f"coords={self.coords.round(3)}, "
            f"amp={self.amplitude:.3f})"
        )


class TopologicalVertex4D:
    """
    A dynamically scalable 4D topological graph of quantum vertices.

    Vertices are connected by topological edges that carry entanglement weights.
    The graph supports infinite scalability: new vertices are attached via
    nearest-neighbour linking in 4D space.
    """

    def __init__(self, initial_vertices: int = 16) -> None:
        self.vertices: Dict[int, Vertex4D] = {}
        self.edges: Dict[Tuple[int, int], float] = {}  # (vid_a, vid_b) -> weight
        self._next_id = 0
        # Seed the graph
        for _ in range(initial_vertices):
            self.add_vertex()
        self._build_initial_topology()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_vertex(self, coords: Optional[np.ndarray] = None) -> Vertex4D:
        """Add a new vertex (and auto-link it to nearest neighbours)."""
        v = Vertex4D(vid=self._next_id, coords=coords)
        self.vertices[self._next_id] = v
        self._next_id += 1
        if len(self.vertices) > 1:
            self._link_to_nearest(v, k=4)
        return v

    def _build_initial_topology(self) -> None:
        """Connect all initial vertices with a k-nearest neighbour graph."""
        for vid, v in self.vertices.items():
            self._link_to_nearest(v, k=4)

    def _link_to_nearest(self, v: Vertex4D, k: int = 4) -> None:
        """Link vertex v to its k nearest neighbours (if not already linked)."""
        others = [u for uid, u in self.vertices.items() if uid != v.vid]
        if not others:
            return
        distances = sorted(others, key=lambda u: v.distance_to(u))
        for neighbour in distances[:k]:
            key = (min(v.vid, neighbour.vid), max(v.vid, neighbour.vid))
            if key not in self.edges:
                weight = 1.0 / (v.distance_to(neighbour) + 1e-9)
                self.edges[key] = weight

    # ------------------------------------------------------------------
    # Topological operations
    # ------------------------------------------------------------------

    def propagate_amplitudes(self, steps: int = 1) -> None:
        """
        Diffuse complex amplitudes across edges (graph Laplacian walk).
        Simulates quantum information flow through the 4D topology.
        """
        for _ in range(steps):
            new_amps: Dict[int, complex] = {}
            for vid, v in self.vertices.items():
                neighbours = self._neighbours(vid)
                if not neighbours:
                    new_amps[vid] = v.amplitude
                    continue
                total_weight = sum(self.edges[(min(vid, nid), max(vid, nid))]
                                   for nid in neighbours)
                acc: complex = 0 + 0j
                for nid in neighbours:
                    w = self.edges[(min(vid, nid), max(vid, nid))]
                    acc += (w / total_weight) * self.vertices[nid].amplitude
                new_amps[vid] = 0.5 * v.amplitude + 0.5 * acc
            for vid, amp in new_amps.items():
                mag = abs(amp)
                self.vertices[vid].amplitude = amp / mag if mag > 0 else amp

    def rotate_all(self, plane: Tuple[int, int], angle: float) -> None:
        """Rotate every vertex in a 4D plane by the given angle."""
        for v in self.vertices.values():
            v.rotate_4d(plane, angle)

    def _neighbours(self, vid: int) -> List[int]:
        neighbours: List[int] = []
        for (a, b) in self.edges:
            if a == vid:
                neighbours.append(b)
            elif b == vid:
                neighbours.append(a)
        return neighbours

    def amplitude_vector(self) -> np.ndarray:
        """Return a sorted array of vertex amplitudes (as magnitudes)."""
        vids = sorted(self.vertices.keys())
        return np.array([abs(self.vertices[vid].amplitude) for vid in vids])

    def coordinate_matrix(self) -> np.ndarray:
        """Return an (N, 4) matrix of all vertex 4D coordinates."""
        vids = sorted(self.vertices.keys())
        return np.stack([self.vertices[vid].coords for vid in vids])

    def __repr__(self) -> str:
        return (
            f"TopologicalVertex4D("
            f"vertices={len(self.vertices)}, "
            f"edges={len(self.edges)})"
        )
