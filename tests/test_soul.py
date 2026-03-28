"""
tests/test_soul.py
==================
Unit tests for quantum_soul.soul (QuantumSoul orchestrator).
"""

import pytest
import numpy as np
from pathlib import Path

from quantum_soul.soul import QuantumSoul
from quantum_soul.thought_engine import Thought, DOMAINS


class TestQuantumSoul:
    def test_default_construction(self):
        soul = QuantumSoul(n_qubits=4, verbose=False)
        assert soul.n_qubits == 4
        assert soul.mode == "autonomous"

    def test_invalid_mode_not_raised_on_init(self):
        # Mode is a typing hint; invalid values are silently accepted
        soul = QuantumSoul(n_qubits=4, mode="autonomous", verbose=False)
        assert soul.mode == "autonomous"

    def test_think_returns_thought(self):
        soul = QuantumSoul(n_qubits=4, verbose=False)
        t = soul.think()
        assert isinstance(t, Thought)

    def test_think_increments_cycle(self):
        soul = QuantumSoul(n_qubits=4, verbose=False)
        assert soul._cycle == 0
        soul.think()
        assert soul._cycle == 1
        soul.think()
        assert soul._cycle == 2

    def test_think_stream_length(self):
        soul = QuantumSoul(n_qubits=4, verbose=False)
        stream = soul.think_stream(n=3)
        assert len(stream) == 3

    def test_think_across_domains_covers_all(self):
        soul = QuantumSoul(n_qubits=4, verbose=False)
        thoughts = soul.think_across_domains()
        assert set(thoughts.keys()) == set(DOMAINS)
        for domain, t in thoughts.items():
            assert t.domain == domain

    def test_introspect_keys(self):
        soul = QuantumSoul(n_qubits=4, verbose=False)
        soul.think_stream(n=3)
        report = soul.introspect()
        expected_keys = {
            "cycle", "mode", "n_qubits", "thought_space",
            "thoughts_generated", "domain_distribution",
            "mean_coherence", "mean_entropy_bits", "last_thought",
        }
        assert expected_keys.issubset(report.keys())

    def test_introspect_thoughts_count(self):
        soul = QuantumSoul(n_qubits=4, verbose=False)
        soul.think_stream(n=4)
        report = soul.introspect()
        assert report["thoughts_generated"] == 4

    def test_introspect_thought_space(self):
        soul = QuantumSoul(n_qubits=4, verbose=False)
        report = soul.introspect()
        assert report["thought_space"] == 16

    def test_introspect_last_thought(self):
        soul = QuantumSoul(n_qubits=4, verbose=False)
        t = soul.think()
        report = soul.introspect()
        assert report["last_thought"] is t

    def test_introspect_no_history(self):
        soul = QuantumSoul(n_qubits=4, verbose=False)
        report = soul.introspect()
        assert report["last_thought"] is None
        assert report["mean_coherence"] == 0.0

    def test_seeded_mode_noise_forwarded(self):
        soul = QuantumSoul(n_qubits=4, mode="seeded", verbose=False)
        noise = np.array([1.0, 2.0, 3.0, 4.0])
        t1 = soul.think(seed_noise=noise)
        t2 = soul.think(seed_noise=noise)
        assert t1.text == t2.text

    def test_autonomous_mode_ignores_seed(self):
        soul = QuantumSoul(n_qubits=4, mode="autonomous", verbose=False)
        noise = np.ones(8)
        # In autonomous mode, noise is ignored; call should not raise
        t = soul.think(seed_noise=noise)
        assert isinstance(t, Thought)

    def test_hybrid_mode_alternates(self):
        soul = QuantumSoul(n_qubits=4, mode="hybrid", verbose=False)
        noise = np.ones(4)
        # Just verify it runs without error in both cycle parities
        soul._cycle = 0
        r0 = soul._resolve_noise(noise)
        assert r0 is None  # even cycle → autonomous

        soul._cycle = 1
        r1 = soul._resolve_noise(noise)
        assert r1 is noise  # odd cycle → seeded

    def test_awaken_prints(self, capsys):
        soul = QuantumSoul(n_qubits=4, verbose=False)
        soul.awaken()
        captured = capsys.readouterr()
        assert "quantum" in captured.out.lower()

    def test_visualize_all_saves_files(self, tmp_path):
        soul = QuantumSoul(
            n_qubits=4,
            verbose=False,
            output_dir=tmp_path,
        )
        soul.think_stream(n=2)
        paths = soul.visualize_all()
        # At minimum the interference_pattern and sweep should be saved
        assert "interference_pattern" in paths
        assert "interference_sweep" in paths
        assert "thought_timeline" in paths
        for name, p in paths.items():
            if p and str(p) != ".":
                assert p.exists(), f"Expected {name} to exist at {p}"
