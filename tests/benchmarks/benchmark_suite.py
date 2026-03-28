"""
Benchmark suite – real-world performance metrics.

Run with: pytest tests/benchmarks/benchmark_suite.py -v --benchmark-sort=mean

Benchmarks cover:
  * Fractal rendering throughput (fps)
  * Encryption throughput (MB/s)
  * SHA-256 hashing throughput (MB/s)
  * Resource monitor sample latency
  * Task scheduler dispatch latency
"""
from __future__ import annotations

import asyncio
import os
import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fractal benchmarks
# ---------------------------------------------------------------------------

from fractal.engine import FractalEngine
from fractal.julia import JuliaRenderer
from fractal.ifs import IFSRenderer, BARNSLEY_FERN
from fractal.mandelbulb import MandelbulbRenderer


def test_julia_128_fps(benchmark):
    """Benchmark Julia set rendering at 128×128."""
    renderer = JuliaRenderer(width=128, height=128)
    result = benchmark(renderer.render, frame_index=0)
    assert result.shape == (128, 128, 3)


def test_ifs_512_fps(benchmark):
    """Benchmark IFS chaos-game rendering at 512×512 (200k iterations)."""
    renderer = IFSRenderer(width=512, height=512, n_iterations=200_000)
    result = benchmark(renderer.render, system=BARNSLEY_FERN, frame_index=0)
    assert result.shape == (512, 512, 3)


def test_fractal_engine_frame_rate(benchmark):
    """End-to-end frame production rate including PNG encoding."""
    engine = FractalEngine(width=128, height=128)
    result = benchmark(engine.render_frame_bytes, frame_index=0)
    assert isinstance(result, bytes)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Encryption benchmarks
# ---------------------------------------------------------------------------

from security.encryption import MeshEncryption


def test_encrypt_1mb_throughput(benchmark):
    """AES-256-GCM / NaCl secretbox: encrypt 1 MB."""
    key = os.urandom(32)
    enc = MeshEncryption(key)
    data = os.urandom(1024 * 1024)
    result = benchmark(enc.encrypt, data)
    assert len(result) > len(data)  # ciphertext is larger (nonce + tag)


def test_decrypt_1mb_throughput(benchmark):
    """AES-256-GCM / NaCl secretbox: decrypt 1 MB."""
    key = os.urandom(32)
    enc = MeshEncryption(key)
    data = os.urandom(1024 * 1024)
    ct = enc.encrypt(data)
    result = benchmark(enc.decrypt, ct)
    assert result == data


# ---------------------------------------------------------------------------
# SHA-256 benchmarks
# ---------------------------------------------------------------------------

import hashlib


def test_sha256_10mb_throughput(benchmark):
    """SHA-256 hashing of 10 MB."""
    data = os.urandom(10 * 1024 * 1024)
    result = benchmark(hashlib.sha256, data)
    assert len(result.hexdigest()) == 64


# ---------------------------------------------------------------------------
# Resource monitor benchmarks
# ---------------------------------------------------------------------------

from hive.resource_monitor import ResourceMonitor


def test_resource_sample_latency(benchmark):
    """Measure single resource sample latency."""
    monitor = ResourceMonitor()
    result = benchmark(monitor._sample)
    assert result.cpu_count >= 1


# ---------------------------------------------------------------------------
# Hive node task dispatch
# ---------------------------------------------------------------------------

from hive.node import HiveNode, TaskPacket
import json


@pytest.mark.asyncio
async def test_local_task_dispatch_latency(benchmark):
    """Measure round-trip latency for a local no-op task dispatch."""
    node = HiveNode(host="127.0.0.1", port=17100)
    await node.start()

    payload = json.dumps({"fn": "ping", "args": []}).encode()
    packet = TaskPacket(payload=payload)

    async def _dispatch():
        return await node.submit_task(packet)

    # Use a simple timer since pytest-benchmark doesn't play well with async
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        result = await node.submit_task(TaskPacket(
            payload=json.dumps({"fn": "ping", "args": []}).encode()
        ))
        times.append(time.perf_counter() - t0)
        assert result and result.result == "pong"

    await node.stop()

    avg_ms = (sum(times) / len(times)) * 1000
    p99_ms = sorted(times)[int(0.99 * len(times))] * 1000
    print(f"\nTask dispatch avg={avg_ms:.2f}ms  p99={p99_ms:.2f}ms")
    assert avg_ms < 500, f"Average dispatch latency too high: {avg_ms:.2f}ms"
