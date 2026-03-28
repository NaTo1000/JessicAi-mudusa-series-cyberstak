"""
MandelbulbRenderer – 3D Mandelbulb fractal via sphere-tracing / ray-marching.

The Mandelbulb is a three-dimensional analogue of the Mandelbrot set,
first described by Daniel White and Paul Nylander.  It is defined by the
iteration:

    z_{n+1} = z_n^p + c,   z_0 = c

where the exponentiation is performed in spherical coordinates:

    r     = |z|
    theta = atan2(sqrt(x² + y²), z)   (polar angle)
    phi   = atan2(y, x)                (azimuthal angle)

    z^p = r^p * (sin(p·theta)·cos(p·phi),
                 sin(p·theta)·sin(p·phi),
                 cos(p·theta))

Power p=8 produces the canonical "spiky ball"; varying p over time gives
the continuously-evolving quality needed by the honeypot.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Vec3 = np.ndarray   # shape (3,)
Float = float


# ---------------------------------------------------------------------------
# Distance estimator
# ---------------------------------------------------------------------------

def mandelbulb_de(
    pos: np.ndarray,
    power: float = 8.0,
    max_iter: int = 64,
    bailout: float = 2.0,
) -> float:
    """
    Signed distance estimate to the Mandelbulb surface at *pos*.

    Returns a float suitable for sphere-tracing: positive outside the
    fractal, negative inside.
    """
    z = pos.copy()
    dr = 1.0
    r = 0.0
    for _ in range(max_iter):
        r = math.sqrt(float(z[0] ** 2 + z[1] ** 2 + z[2] ** 2))
        if r > bailout:
            break
        # Convert to polar
        # Clamp argument to [-1, 1] to guard against floating-point error
        # when r is very small but nonzero (avoids math domain error in acos).
        theta = math.acos(max(-1.0, min(1.0, float(z[2]) / r))) if r > 1e-10 else 0.0
        phi = math.atan2(float(z[1]), float(z[0]))
        dr = r ** (power - 1.0) * power * dr + 1.0
        # Scale and rotate
        zr = r ** power
        theta *= power
        phi *= power
        # Back to cartesian
        z = zr * np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ]) + pos
    return 0.5 * math.log(r) * r / dr if r and dr else 0.0


# ---------------------------------------------------------------------------
# Vectorised batch DE for NumPy arrays
# ---------------------------------------------------------------------------

def mandelbulb_de_batch(
    points: np.ndarray,   # (N, 3)
    power: float = 8.0,
    max_iter: int = 64,
    bailout: float = 2.0,
) -> np.ndarray:          # (N,)
    """
    Vectorised Mandelbulb DE over a batch of points.

    Uses Python loops internally (no GPU required) but is NumPy-friendly
    so that callers can operate on image-sized batches without per-pixel
    Python overhead via np.apply_along_axis.
    """
    return np.array([
        mandelbulb_de(p, power=power, max_iter=max_iter, bailout=bailout)
        for p in points
    ])


# ---------------------------------------------------------------------------
# Sphere tracer
# ---------------------------------------------------------------------------

def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


class MandelbulbRenderer:
    """
    Software ray-marcher for the 3-D Mandelbulb.

    Produces an (H, W, 3) uint8 numpy array representing one frame.

    Parameters
    ----------
    width, height  : Image dimensions in pixels.
    power          : Mandelbulb exponent (8 = canonical).  Animate over time.
    max_march_steps: Ray-marching iteration limit per pixel.
    min_dist       : Ray-march hit threshold.
    max_dist       : Ray-march miss threshold (far clip).
    fov_deg        : Horizontal field-of-view in degrees.
    """

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        power: float = 8.0,
        max_march_steps: int = 128,
        min_dist: float = 1e-4,
        max_dist: float = 10.0,
        fov_deg: float = 60.0,
    ) -> None:
        self.width = width
        self.height = height
        self.power = power
        self._max_steps = max_march_steps
        self._min_dist = min_dist
        self._max_dist = max_dist
        self._fov_rad = math.radians(fov_deg)

    def render(
        self,
        camera_pos: Optional[np.ndarray] = None,
        look_at: Optional[np.ndarray] = None,
        frame_index: int = 0,
    ) -> np.ndarray:
        """
        Render the Mandelbulb and return an (H, W, 3) uint8 array.

        *frame_index* is used to evolve the power parameter slightly on
        each call to produce a continuously changing animation.
        """
        # Animate power
        t = frame_index * 0.01
        power = self.power + 2.0 * math.sin(t * 0.3)

        if camera_pos is None:
            angle = frame_index * 0.02
            camera_pos = np.array([
                3.0 * math.sin(angle),
                1.5 * math.cos(angle * 0.7),
                3.0 * math.cos(angle),
            ])
        if look_at is None:
            look_at = np.zeros(3)

        # Camera basis
        forward = _normalise(look_at - camera_pos)
        right = _normalise(np.cross(forward, np.array([0.0, 1.0, 0.0])))
        up = np.cross(right, forward)

        aspect = self.width / self.height
        half_fov = math.tan(self._fov_rad / 2.0)

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for py in range(self.height):
            vy = (1.0 - 2.0 * (py + 0.5) / self.height) * half_fov
            for px in range(self.width):
                vx = (2.0 * (px + 0.5) / self.width - 1.0) * half_fov * aspect
                ray_dir = _normalise(forward + vx * right + vy * up)
                colour = self._march(camera_pos, ray_dir, power, frame_index)
                img[py, px] = colour

        return img

    # ------------------------------------------------------------------
    # Sphere-march + lighting
    # ------------------------------------------------------------------

    def _march(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        power: float,
        frame_index: int,
    ) -> Tuple[int, int, int]:
        t = 0.0
        for _ in range(self._max_steps):
            pos = origin + t * direction
            d = mandelbulb_de(pos, power=power)
            if abs(d) < self._min_dist:
                return self._shade(pos, direction, power, frame_index)
            if t > self._max_dist:
                break
            t += d * 0.5  # conservative step for accuracy near surface
        return self._sky_colour(direction, frame_index)

    def _shade(
        self,
        hit: np.ndarray,
        direction: np.ndarray,
        power: float,
        frame_index: int,
    ) -> Tuple[int, int, int]:
        """Phong-like shading with animated light position."""
        eps = 1e-4
        # Numerical gradient → surface normal
        normal = _normalise(np.array([
            mandelbulb_de(hit + np.array([eps, 0, 0]), power) -
            mandelbulb_de(hit - np.array([eps, 0, 0]), power),
            mandelbulb_de(hit + np.array([0, eps, 0]), power) -
            mandelbulb_de(hit - np.array([0, eps, 0]), power),
            mandelbulb_de(hit + np.array([0, 0, eps]), power) -
            mandelbulb_de(hit - np.array([0, 0, eps]), power),
        ]))

        t = frame_index * 0.02
        light_pos = np.array([
            3.0 * math.cos(t),
            2.0 + math.sin(t * 0.5),
            3.0 * math.sin(t),
        ])
        light_dir = _normalise(light_pos - hit)

        diffuse = max(0.0, float(np.dot(normal, light_dir)))
        reflect = _normalise(2 * float(np.dot(normal, light_dir)) * normal - light_dir)
        specular = max(0.0, float(np.dot(reflect, -direction))) ** 16

        hue_shift = (frame_index * 0.5) % 360
        r, g, b = _hsv_to_rgb((hue_shift + 200) % 360, 0.8, diffuse + 0.3)
        r = min(255, int(r * 255 + specular * 200))
        g = min(255, int(g * 255 + specular * 100))
        b = min(255, int(b * 255 + specular * 50))
        return r, g, b

    def _sky_colour(
        self, direction: np.ndarray, frame_index: int
    ) -> Tuple[int, int, int]:
        """Background gradient that also evolves with time."""
        t = frame_index * 0.3
        y = float(direction[1]) * 0.5 + 0.5
        r = int((0.05 + 0.1 * math.sin(t * 0.1)) * 255)
        g = int((0.02 + 0.05 * y) * 255)
        b = int((0.1 + 0.2 * y + 0.1 * math.sin(t * 0.07)) * 255)
        return (
            min(255, max(0, r)),
            min(255, max(0, g)),
            min(255, max(0, b)),
        )


# ---------------------------------------------------------------------------
# HSV → RGB helper
# ---------------------------------------------------------------------------

def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Return (r, g, b) in [0, 1] from HSV."""
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
