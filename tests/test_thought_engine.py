"""
tests/test_thought_engine.py
============================
Unit tests for quantum_soul.thought_engine.
"""

import pytest
import numpy as np

from quantum_soul.thought_engine import (
    ThoughtEngine,
    Thought,
    DOMAINS,
    quick_thought,
)


class TestThought:
    def test_str_representation(self):
        t = Thought(
            text="test thought",
            domain="Science",
            secondary_domain="Philosophy",
            coherence=0.75,
            entropy_bits=5.0,
            quantum_state_id="abc123",
        )
        s = str(t)
        assert "Science" in s
        assert "Philosophy" in s
        assert "0.750" in s
        assert "test thought" in s

    def test_str_no_secondary(self):
        t = Thought(
            text="solo thought",
            domain="Self",
            secondary_domain=None,
            coherence=0.5,
            entropy_bits=6.0,
            quantum_state_id="xyz",
        )
        s = str(t)
        assert "↔" not in s

    def test_timestamp_set(self):
        t = Thought(
            text="timed",
            domain="Technology",
            secondary_domain=None,
            coherence=0.5,
            entropy_bits=4.0,
            quantum_state_id="ts",
        )
        assert t.timestamp > 0.0


class TestThoughtEngine:
    def test_default_construction(self):
        engine = ThoughtEngine()
        assert engine.n_qubits == 8

    def test_generate_thought_returns_thought(self):
        engine = ThoughtEngine(n_qubits=4)
        t = engine.generate_thought()
        assert isinstance(t, Thought)

    def test_thought_domain_valid(self):
        engine = ThoughtEngine(n_qubits=4)
        t = engine.generate_thought()
        assert t.domain in DOMAINS

    def test_thought_coherence_range(self):
        engine = ThoughtEngine(n_qubits=4)
        for _ in range(5):
            t = engine.generate_thought()
            assert 0.0 <= t.coherence <= 1.0

    def test_thought_entropy_nonneg(self):
        engine = ThoughtEngine(n_qubits=4)
        t = engine.generate_thought()
        assert t.entropy_bits >= 0.0

    def test_thought_text_nonempty(self):
        engine = ThoughtEngine(n_qubits=4)
        t = engine.generate_thought()
        assert len(t.text) > 0

    def test_thought_state_id_length(self):
        engine = ThoughtEngine(n_qubits=4)
        t = engine.generate_thought()
        assert len(t.quantum_state_id) == 12

    def test_raw_amplitudes_populated(self):
        engine = ThoughtEngine(n_qubits=4)
        t = engine.generate_thought()
        assert len(t.raw_amplitudes) > 0
        for state_str, prob in t.raw_amplitudes:
            assert 0.0 <= prob <= 1.0

    def test_domain_hint_respected(self):
        """When a valid domain hint is provided, it should be used."""
        engine = ThoughtEngine(n_qubits=4)
        for domain in DOMAINS:
            t = engine.generate_thought(domain_hint=domain)
            assert t.domain == domain

    def test_seeded_thought_deterministic(self):
        """Same noise → same thought text."""
        noise = np.array([1.0, 2.0, 3.0, 4.0])
        engine = ThoughtEngine(n_qubits=4, entanglement_depth=1)
        t1 = engine.generate_thought(seed_noise=noise)
        t2 = engine.generate_thought(seed_noise=noise)
        assert t1.text == t2.text

    def test_generate_thought_stream_length(self):
        engine = ThoughtEngine(n_qubits=4)
        stream = engine.generate_thought_stream(n=3)
        assert len(stream) == 3
        for t in stream:
            assert isinstance(t, Thought)

    def test_history_grows(self):
        engine = ThoughtEngine(n_qubits=4)
        engine.generate_thought()
        engine.generate_thought()
        assert len(engine.get_history()) == 2

    def test_clear_history(self):
        engine = ThoughtEngine(n_qubits=4)
        engine.generate_thought()
        engine.clear_history()
        assert len(engine.get_history()) == 0

    def test_bucket_probs_sums_correctly(self):
        probs = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.2])
        bucketed = ThoughtEngine._bucket_probs(probs, 2)
        assert abs(bucketed.sum() - probs.sum()) < 1e-9

    def test_domain_weights_sum_to_one(self):
        engine = ThoughtEngine(n_qubits=4)
        n_states = 2 ** 4
        probs = np.ones(n_states) / n_states
        weights = engine._compute_domain_weights(probs)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_domain_range_covers_all_states(self):
        engine = ThoughtEngine(n_qubits=4)
        n_states = 2 ** 4
        total = 0
        for i in range(len(DOMAINS)):
            s, e = engine._domain_range(i, n_states)
            total += e - s
        assert total == n_states


class TestQuickThought:
    def test_returns_string(self):
        t = quick_thought()
        assert isinstance(t, str) and len(t) > 0

    def test_domain_hint(self):
        for d in DOMAINS:
            t = quick_thought(domain=d)
            assert isinstance(t, str) and len(t) > 0
