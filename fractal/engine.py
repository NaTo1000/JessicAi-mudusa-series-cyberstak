"""
FractalEngine – the core production engine powering the honeypot.

This is what an attacker experiences when they reach the mesh port without
the correct session key.  The engine runs indefinitely, producing a fresh
PNG frame on every call.

Design goals
------------
* Never-repeating: the internal frame counter advances monotonically; the
  rendering parameters evolve continuously so each frame is unique.
* Multi-algorithm: Mandelbulb, Julia, and IFS outputs are interleaved
  based on a rotation schedule, with smooth transitions between modes.
* Self-contained: no GPU required; pure Python + NumPy.
* Extensible: register additional generator classes with add_generator().
"""

from __future__ import annotations

import io
import itertools
import logging
import math
import threading
import time
from typing import Callable, Iterator, List, Optional

import numpy as np

from .mandelbulb import MandelbulbRenderer
from .julia import JuliaRenderer
from .ifs import IFSRenderer, PREDEFINED_SYSTEMS

logger = logging.getLogger(__name__)

# Rotation schedule: how many frames each mode holds before switching
_ROTATION: List[str] = (
    ["julia"] * 30
    + ["ifs"] * 20
    + ["mandelbulb"] * 10
    + ["julia"] * 30
    + ["ifs"] * 20
)


class FractalEngine:
    """
    Continuously-running fractal production engine.

    Thread-safe: render_frame_bytes() can be called from any thread.
    """

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        mandelbulb_power: float = 8.0,
    ) -> None:
        self._w = width
        self._h = height
        self._power = mandelbulb_power
        self._julia = JuliaRenderer(width=width, height=height)
        self._ifs = IFSRenderer(width=width, height=height)
        self._bulb = MandelbulbRenderer(width=width, height=height, power=mandelbulb_power)
        self._rotation = itertools.cycle(enumerate(_ROTATION))
        self._frame_counter = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def render_frame_bytes(self, frame_index: Optional[int] = None) -> bytes:
        """
        Render and return one frame as PNG bytes.

        If *frame_index* is None, the internal counter is used and
        incremented atomically.
        """
        with self._lock:
            if frame_index is None:
                frame_index = self._frame_counter
                self._frame_counter += 1

        mode = _ROTATION[frame_index % len(_ROTATION)]
        try:
            arr = self._render(mode, frame_index)
        except Exception:
            logger.exception("FractalEngine render error")
            arr = self._fallback(frame_index)
        return _to_png(arr)

    def frames(self) -> Iterator[bytes]:
        """
        Infinite iterator of PNG-encoded frames.

        Usage::

            for png in engine.frames():
                send(png)
        """
        while True:
            yield self.render_frame_bytes()

    # ------------------------------------------------------------------
    # Internal rendering dispatch
    # ------------------------------------------------------------------

    def _render(self, mode: str, frame_index: int) -> np.ndarray:
        if mode == "mandelbulb":
            return self._bulb.render(frame_index=frame_index)
        elif mode == "julia":
            return self._julia.render(frame_index=frame_index)
        else:
            ifs_system = PREDEFINED_SYSTEMS[frame_index % len(PREDEFINED_SYSTEMS)]
            return self._ifs.render(system=ifs_system, frame_index=frame_index)

    def _fallback(self, frame_index: int) -> np.ndarray:
        """Minimal Lissajous pattern as an absolute fallback."""
        w, h = self._w, self._h
        img = np.zeros((h, w, 3), dtype=np.uint8)
        t = frame_index * 0.1
        for i in range(2000):
            s = i / 2000.0
            x = int((math.sin(3 * s * math.pi * 2 + t) * 0.45 + 0.5) * (w - 1))
            y = int((math.sin(2 * s * math.pi * 2 + t * 0.7) * 0.45 + 0.5) * (h - 1))
            c = int(s * 255)
            img[y, x] = (c, 255 - c, 128)
        return img


# ---------------------------------------------------------------------------
# PNG encoding helper
# ---------------------------------------------------------------------------

def _to_png(arr: np.ndarray) -> bytes:
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        pass
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        plt.imsave(buf, arr, format="png")
        plt.close("all")
        return buf.getvalue()
    except ImportError:
        pass
    return arr.tobytes()
