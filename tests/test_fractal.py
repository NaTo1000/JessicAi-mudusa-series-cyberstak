"""
Tests for the fractal generation engine.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fractal.mandelbulb import (
    MandelbulbRenderer,
    mandelbulb_de,
    mandelbulb_de_batch,
    _hsv_to_rgb,
)
from fractal.julia import JuliaRenderer, quat_mul, quat_sq, julia_de
from fractal.ifs import IFSRenderer, BARNSLEY_FERN, SIERPINSKI
from fractal.engine import FractalEngine


# ---------------------------------------------------------------------------
# Mandelbulb
# ---------------------------------------------------------------------------

class TestMandelbulbDE:
    def test_origin_inside(self):
        """The origin (0,0,0) is inside the Mandelbulb – DE should be ≤ 0."""
        d = mandelbulb_de(np.zeros(3))
        assert d <= 0.0

    def test_far_point_outside(self):
        """A point far from the origin should be outside (DE > 0)."""
        d = mandelbulb_de(np.array([10.0, 10.0, 10.0]))
        assert d > 0.0

    def test_batch_matches_single(self):
        pts = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        batch = mandelbulb_de_batch(pts)
        single = [mandelbulb_de(p) for p in pts]
        for b, s in zip(batch, single):
            assert abs(b - s) < 1e-10


class TestMandelbulbRenderer:
    def test_render_shape(self):
        r = MandelbulbRenderer(width=32, height=32)
        img = r.render(frame_index=0)
        assert img.shape == (32, 32, 3)
        assert img.dtype == np.uint8

    def test_render_different_frames_differ(self):
        r = MandelbulbRenderer(width=32, height=32)
        f0 = r.render(frame_index=0)
        f100 = r.render(frame_index=100)
        # Frames should not be identical (the power parameter evolves)
        assert not np.array_equal(f0, f100)

    def test_hsv_to_rgb_red(self):
        r, g, b = _hsv_to_rgb(0, 1.0, 1.0)
        assert abs(r - 1.0) < 0.01
        assert abs(g) < 0.01
        assert abs(b) < 0.01


# ---------------------------------------------------------------------------
# Julia
# ---------------------------------------------------------------------------

class TestQuaternionOps:
    def test_quat_mul_identity(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([1.0, 2.0, 3.0, 4.0])
        result = quat_mul(identity, q)
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_quat_sq(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(quat_sq(q), np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-10)


class TestJuliaDE:
    def test_returns_float(self):
        pos = np.array([0.0, 0.0, 0.0])
        c = np.array([-0.4, 0.6, 0.0, 0.0])
        d = julia_de(pos, c)
        assert isinstance(d, float)


class TestJuliaRenderer:
    def test_render_shape(self):
        r = JuliaRenderer(width=64, height=64)
        img = r.render(frame_index=0)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8

    def test_frames_differ(self):
        r = JuliaRenderer(width=32, height=32)
        f0 = r.render(frame_index=0)
        f50 = r.render(frame_index=50)
        assert not np.array_equal(f0, f50)


# ---------------------------------------------------------------------------
# IFS
# ---------------------------------------------------------------------------

class TestIFSRenderer:
    def test_render_shape(self):
        r = IFSRenderer(width=64, height=64, n_iterations=5000)
        img = r.render(frame_index=0)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8

    def test_barnsley_fern_non_black(self):
        r = IFSRenderer(width=64, height=64, n_iterations=50000)
        img = r.render(system=BARNSLEY_FERN, frame_index=0)
        # Should have some non-black pixels
        assert img.sum() > 0

    def test_sierpinski_non_black(self):
        r = IFSRenderer(width=64, height=64, n_iterations=50000)
        img = r.render(system=SIERPINSKI, frame_index=0)
        assert img.sum() > 0


# ---------------------------------------------------------------------------
# FractalEngine (honeypot core)
# ---------------------------------------------------------------------------

class TestFractalEngine:
    def test_render_frame_bytes_is_bytes(self):
        eng = FractalEngine(width=32, height=32)
        result = eng.render_frame_bytes(frame_index=0)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_frame_counter_increments(self):
        eng = FractalEngine(width=32, height=32)
        eng.render_frame_bytes()
        eng.render_frame_bytes()
        assert eng._frame_counter == 2

    def test_explicit_frame_index_does_not_advance_counter(self):
        eng = FractalEngine(width=32, height=32)
        eng.render_frame_bytes(frame_index=10)
        assert eng._frame_counter == 0  # counter untouched

    def test_consecutive_frames_differ(self):
        eng = FractalEngine(width=32, height=32)
        f0 = eng.render_frame_bytes(frame_index=0)
        f1 = eng.render_frame_bytes(frame_index=1)
        # Different frame indices must produce different output
        assert f0 != f1

    def test_frames_iterator(self):
        eng = FractalEngine(width=32, height=32)
        gen = eng.frames()
        frames = [next(gen) for _ in range(5)]
        assert all(isinstance(f, bytes) for f in frames)
        # At least some frames should differ
        assert len(set(frames)) > 1
