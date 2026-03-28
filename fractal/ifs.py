"""
IFSRenderer – Iterated Function System fractal generator.

An IFS is defined by a finite set of contractive affine transformations
T_1, …, T_n with associated probabilities p_1, …, p_n.  Applying the
chaos game (stochastic iteration) produces the attractor of the IFS in
O(n_iterations) time.

Built-in systems
----------------
* Barnsley Fern (iconic, life-like leaf)
* Dragon Curve (Heighway dragon)
* Koch Snowflake approximation
* Sierpiński Triangle
* Custom random IFS (seeded from frame_index for infinite variation)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class AffineTransform:
    """2-D affine transform: p' = A·p + b."""
    a: float; b: float; c: float; d: float  # matrix coefficients
    e: float; f: float                       # translation
    prob: float = 0.25                       # selection probability


@dataclass
class IFSSystem:
    name: str
    transforms: List[AffineTransform]
    color_map: str = "viridis"


# ---------------------------------------------------------------------------
# Built-in IFS libraries
# ---------------------------------------------------------------------------

BARNSLEY_FERN = IFSSystem(
    name="Barnsley Fern",
    transforms=[
        AffineTransform(0.00,  0.00,  0.00,  0.16, 0.00, 0.00, prob=0.01),
        AffineTransform(0.85,  0.04, -0.04,  0.85, 0.00, 1.60, prob=0.85),
        AffineTransform(0.20, -0.26,  0.23,  0.22, 0.00, 1.60, prob=0.07),
        AffineTransform(-0.15, 0.28,  0.26,  0.24, 0.00, 0.44, prob=0.07),
    ],
    color_map="Greens",
)

SIERPINSKI = IFSSystem(
    name="Sierpiński Triangle",
    transforms=[
        AffineTransform(0.5, 0.0, 0.0, 0.5, 0.0,  0.0,  prob=1/3),
        AffineTransform(0.5, 0.0, 0.0, 0.5, 0.5,  0.0,  prob=1/3),
        AffineTransform(0.5, 0.0, 0.0, 0.5, 0.25, 0.5,  prob=1/3),
    ],
    color_map="plasma",
)

DRAGON_CURVE = IFSSystem(
    name="Dragon Curve",
    transforms=[
        AffineTransform(0.5, -0.5, 0.5,  0.5, 0.0, 0.0, prob=0.5),
        AffineTransform(-0.5, -0.5, 0.5, -0.5, 1.0, 0.0, prob=0.5),
    ],
    color_map="inferno",
)

PREDEFINED_SYSTEMS = [BARNSLEY_FERN, SIERPINSKI, DRAGON_CURVE]


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class IFSRenderer:
    """
    Renders an IFS attractor using the chaos game algorithm.

    Parameters
    ----------
    width, height  : Output image dimensions.
    n_iterations   : Number of chaos game steps.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        n_iterations: int = 200_000,
    ) -> None:
        self.width = width
        self.height = height
        self._n_iters = n_iterations

    def render(
        self,
        system: Optional[IFSSystem] = None,
        frame_index: int = 0,
    ) -> np.ndarray:
        """
        Render the given IFS system (or a random one seeded from *frame_index*).

        Returns (H, W, 3) uint8 array.
        """
        if system is None:
            # Cycle through predefined systems, injecting small random perturbations
            base = PREDEFINED_SYSTEMS[frame_index % len(PREDEFINED_SYSTEMS)]
            system = self._perturb(base, seed=frame_index)

        # Chaos game
        transforms = system.transforms
        cumulative = []
        acc = 0.0
        for t in transforms:
            acc += t.prob
            cumulative.append(acc)

        rng = random.Random(frame_index)
        x, y = 0.0, 0.0
        points = np.empty((self._n_iters, 2), dtype=np.float64)

        for i in range(self._n_iters):
            r = rng.random()
            for j, threshold in enumerate(cumulative):
                if r <= threshold:
                    t = transforms[j]
                    break
            x, y = t.a * x + t.b * y + t.e, t.c * x + t.d * y + t.f
            points[i] = (x, y)

        return self._points_to_image(points, frame_index)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _perturb(self, base: IFSSystem, seed: int) -> IFSSystem:
        """Add small time-varying perturbations to animate the IFS."""
        rng = random.Random(seed)
        noise = 0.04
        new_transforms = []
        for t in base.transforms:
            new_transforms.append(AffineTransform(
                a=t.a + rng.uniform(-noise, noise),
                b=t.b + rng.uniform(-noise, noise),
                c=t.c + rng.uniform(-noise, noise),
                d=t.d + rng.uniform(-noise, noise),
                e=t.e + rng.uniform(-noise * 0.1, noise * 0.1),
                f=t.f + rng.uniform(-noise * 0.1, noise * 0.1),
                prob=t.prob,
            ))
        return IFSSystem(name=base.name, transforms=new_transforms)

    def _points_to_image(self, points: np.ndarray, frame_index: int) -> np.ndarray:
        """Histogram the chaos game points onto a pixel grid."""
        xs = points[:, 0]
        ys = points[:, 1]
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        xrange = (xmax - xmin) or 1.0
        yrange = (ymax - ymin) or 1.0

        # Map to pixel coordinates
        pxs = ((xs - xmin) / xrange * (self.width - 1)).astype(np.int32)
        pys = ((ys - ymin) / yrange * (self.height - 1)).astype(np.int32)
        pxs = np.clip(pxs, 0, self.width - 1)
        pys = np.clip(pys, 0, self.height - 1)

        # Density histogram
        density = np.zeros((self.height, self.width), dtype=np.float32)
        np.add.at(density, (pys, pxs), 1)

        # Log-normalise
        density = np.log1p(density)
        dmax = density.max()
        if dmax > 0:
            density /= dmax

        # Colourise with a time-shifted hue ramp
        hue_shift = (frame_index * 2) % 360
        img = _colourise(density, hue_shift)
        return img


def _colourise(density: np.ndarray, hue_shift: float) -> np.ndarray:
    """Map a (H, W) [0,1] density map to an (H, W, 3) uint8 image."""
    h, w = density.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    hue = ((density * 240 + hue_shift) % 360).astype(np.float32)
    sat = np.clip(density * 1.2, 0, 1).astype(np.float32)
    val = np.clip(density * 1.5, 0, 1).astype(np.float32)

    # Vectorised HSV → RGB
    c_arr = val * sat
    x_arr = c_arr * (1 - np.abs((hue / 60) % 2 - 1))
    m_arr = val - c_arr

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for mask, rv, gv, bv in [
        ((hue < 60), c_arr, x_arr, np.zeros_like(c_arr)),
        ((hue < 120), x_arr, c_arr, np.zeros_like(c_arr)),
        ((hue < 180), np.zeros_like(c_arr), c_arr, x_arr),
        ((hue < 240), np.zeros_like(c_arr), x_arr, c_arr),
        ((hue < 300), x_arr, np.zeros_like(c_arr), c_arr),
        ((hue >= 300), c_arr, np.zeros_like(c_arr), x_arr),
    ]:
        rgb[mask, 0] = rv[mask]
        rgb[mask, 1] = gv[mask]
        rgb[mask, 2] = bv[mask]

    img = ((rgb + m_arr[:, :, np.newaxis]) * 255).clip(0, 255).astype(np.uint8)
    return img
