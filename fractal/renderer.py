"""
FractalRenderer – thin wrapper that picks the optimal fractal backend
based on the requested resolution and the available hardware.

Priority
--------
1. Mandelbulb (3-D, most visually impressive) when resolution < 512.
2. Julia set (2-D quaternion slice, fast) for streaming thumbnails.
3. IFS Chaos Game (ultra-fast, infinite variation) for low-power devices.
"""

from __future__ import annotations

import io
import logging
from enum import Enum, auto
from typing import Optional

import numpy as np

from .mandelbulb import MandelbulbRenderer
from .julia import JuliaRenderer
from .ifs import IFSRenderer

logger = logging.getLogger(__name__)


class FractalMode(Enum):
    MANDELBULB = auto()
    JULIA = auto()
    IFS = auto()
    AUTO = auto()


class FractalRenderer:
    """
    Unified fractal renderer facade.

    >>> renderer = FractalRenderer(width=256, height=256, mode=FractalMode.AUTO)
    >>> frame = renderer.render(frame_index=0)   # (256, 256, 3) uint8 array
    >>> png_bytes = renderer.render_png(frame_index=0)
    """

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        mode: FractalMode = FractalMode.AUTO,
    ) -> None:
        self.width = width
        self.height = height
        self._mode = mode
        self._mandelbulb = MandelbulbRenderer(width=width, height=height)
        self._julia = JuliaRenderer(width=width, height=height)
        self._ifs = IFSRenderer(width=width, height=height)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def render(self, frame_index: int = 0) -> np.ndarray:
        """Return (H, W, 3) uint8 array."""
        mode = self._resolve_mode()
        try:
            if mode == FractalMode.MANDELBULB:
                return self._mandelbulb.render(frame_index=frame_index)
            elif mode == FractalMode.JULIA:
                return self._julia.render(frame_index=frame_index)
            else:
                return self._ifs.render(frame_index=frame_index)
        except Exception:
            logger.exception("FractalRenderer.render error – falling back to IFS")
            return self._ifs.render(frame_index=frame_index)

    def render_png(self, frame_index: int = 0) -> bytes:
        """Render and return PNG-encoded bytes."""
        array = self.render(frame_index=frame_index)
        return _array_to_png(array)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_mode(self) -> FractalMode:
        if self._mode != FractalMode.AUTO:
            return self._mode
        total_pixels = self.width * self.height
        if total_pixels <= 128 * 128:
            return FractalMode.MANDELBULB
        elif total_pixels <= 512 * 512:
            return FractalMode.JULIA
        return FractalMode.IFS


def _array_to_png(arr: np.ndarray) -> bytes:
    """Convert (H, W, 3) uint8 array to PNG bytes without heavy dependencies."""
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        pass

    try:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        plt.imsave(buf, arr, format="png")
        return buf.getvalue()
    except ImportError:
        pass

    # Last resort: raw bytes (caller must handle)
    return arr.tobytes()
