"""
JuliaRenderer – animated 4-D Julia set sliced through 3-D.

The quaternion Julia set is defined by the iteration

    q_{n+1} = q_n² + c,   q_0 = p

where *p* is the quaternion corresponding to the 3-D point (x, y, z, 0)
and *c* = (c_r, c_i, c_j, c_k) is the Julia parameter.

A 2-D slice can be rendered at very high speed; evolving *c* over time
generates an organic, never-repeating animation.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two quaternions a and b (shape (4,))."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_sq(q: np.ndarray) -> np.ndarray:
    """q² = q · q."""
    return quat_mul(q, q)


# ---------------------------------------------------------------------------
# Julia DE  (quaternion distance estimator)
# ---------------------------------------------------------------------------

def julia_de(
    pos: np.ndarray,  # (3,) – 3-D position, extended to quaternion with w=0
    c: np.ndarray,    # (4,) – Julia parameter quaternion
    max_iter: int = 64,
    bailout: float = 4.0,
) -> float:
    q = np.array([0.0, float(pos[0]), float(pos[1]), float(pos[2])])
    dq = np.array([1.0, 0.0, 0.0, 0.0])  # derivative start
    for _ in range(max_iter):
        dq = 2.0 * quat_mul(q, dq)
        q = quat_sq(q) + c
        r2 = float(np.dot(q, q))
        if r2 > bailout * bailout:
            r = math.sqrt(r2)
            dr = math.sqrt(float(np.dot(dq, dq)))
            return 0.5 * math.log(r) * r / dr if dr > 1e-10 else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# 2-D slice renderer (fast)
# ---------------------------------------------------------------------------

class JuliaRenderer:
    """
    Renders a 2-D slice of the quaternion Julia set.

    The slice plane and Julia parameter *c* are animated over time,
    producing a continuously-evolving image without repetition.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        max_iter: int = 256,
        zoom: float = 1.5,
    ) -> None:
        self.width = width
        self.height = height
        self._max_iter = max_iter
        self._zoom = zoom

    def render(self, frame_index: int = 0) -> np.ndarray:
        """
        Return an (H, W, 3) uint8 image for the given *frame_index*.

        *c* spirals through quaternion space so consecutive frames look
        related yet are never identical.
        """
        t = frame_index * 0.02
        # Animate c along a Lissajous path in quaternion space
        c = np.array([
            -0.4 + 0.3 * math.sin(t * 0.5),
             0.6 * math.cos(t * 0.7),
             0.2 * math.sin(t * 0.3),
             0.1 * math.cos(t * 1.1),
        ])

        xs = np.linspace(-self._zoom, self._zoom, self.width)
        ys = np.linspace(-self._zoom, self._zoom, self.height)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for py, y in enumerate(ys):
            for px, x in enumerate(xs):
                it = self._iterate(complex(x, y), c)
                img[py, px] = self._colour(it, frame_index)

        return img

    def _iterate(self, z0: complex, c: np.ndarray, bailout: float = 4.0) -> int:
        """
        Standard complex Julia iteration (2-D slice for speed).
        Returns the escape iteration count.
        """
        cx = c[0] + 1j * c[1]
        z = z0
        for i in range(self._max_iter):
            if abs(z) > bailout:
                return i
            z = z * z + cx
        return self._max_iter

    def _colour(self, iteration: int, frame_index: int) -> Tuple[int, int, int]:
        if iteration == self._max_iter:
            return 0, 0, 0
        t = frame_index * 0.5
        # Smooth colouring with palette shift
        smooth = iteration + 1 - math.log(math.log(2.0)) / math.log(2.0)
        hue = (smooth * 10 + t * 30) % 360
        r, g, b = _hsv_to_rgb(hue, 0.9, 0.9)
        return int(r * 255), int(g * 255), int(b * 255)


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0.0
    elif h < 120:
        r, g, b = x, c, 0.0
    elif h < 180:
        r, g, b = 0.0, c, x
    elif h < 240:
        r, g, b = 0.0, x, c
    elif h < 300:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return r + m, g + m, b + m
