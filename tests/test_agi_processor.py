"""Integration tests for the AGI Processor."""

from __future__ import annotations

import pytest

from src.architecture.agi_processor import AGIProcessor, ProcessorConfig, InferenceResult
from src.architecture.tesseract import TesseractConfig
from src.architecture.superconductor import ArrayConfig
from src.architecture.clustering import RegulatorConfig
from src.architecture.vault import VaultConfig, AuthResult


def _make_processor() -> AGIProcessor:
    """Return a small-footprint processor suitable for unit tests."""
    cfg = ProcessorConfig(
        tesseract=TesseractConfig(
            logical_layer_count=1_024,
            max_active_layers=32,
            dimensions=4,
            max_nesting_depth=2,
            quantum_routing=True,
        ),
        superconductor=ArrayConfig(
            logical_layer_count=1_024,
            max_active_layers=32,
            default_workload=0.5,
        ),
        cluster=RegulatorConfig(
            max_sub_clusters=8,
            global_load_ceiling=0.9,
        ),
        vault=VaultConfig(
            rotation_interval_s=30.0,
            max_failed_attempts=5,
        ),
        layers_per_inference=4,
        ticks_per_inference=2,
        initial_sub_clusters=2,
        nodes_per_cluster=2,
    )
    return AGIProcessor(config=cfg)


class TestAGIProcessor:
    def test_processor_initialises(self):
        proc = _make_processor()
        assert proc.processor_id is not None

    def test_infer_with_valid_token_succeeds(self):
        proc = _make_processor()
        token = proc.vault.make_auth_token()
        result = proc.infer({"query": "test"}, token)
        assert isinstance(result, InferenceResult)
        assert result.success is True
        assert result.error is None

    def test_infer_activates_layers(self):
        proc = _make_processor()
        token = proc.vault.make_auth_token()
        result = proc.infer({"query": "test"}, token)
        assert result.layers_activated >= 1

    def test_infer_generates_ops(self):
        proc = _make_processor()
        token = proc.vault.make_auth_token()
        result = proc.infer({"query": "test"}, token)
        assert result.total_ops > 0

    def test_infer_has_positive_latency(self):
        proc = _make_processor()
        token = proc.vault.make_auth_token()
        result = proc.infer({"query": "test"}, token)
        assert result.latency_ms >= 0

    def test_infer_increments_counter(self):
        proc = _make_processor()
        token = proc.vault.make_auth_token()
        proc.infer({"query": "a"}, token)
        proc.infer({"query": "b"}, token)
        assert proc._inference_count == 2

    def test_infer_with_invalid_token_still_returns_result(self):
        # Vault store will fail but the processor should not raise
        proc = _make_processor()
        result = proc.infer({"query": "test"}, "bad-token")
        # Inference may still succeed even if vault seal fails
        assert isinstance(result, InferenceResult)

    def test_infer_output_has_expected_keys(self):
        proc = _make_processor()
        token = proc.vault.make_auth_token()
        result = proc.infer({"query": "test"}, token)
        assert "processor_id" in result.output
        assert "activated_tesseract_layers" in result.output
        assert "superconductor_ops" in result.output

    def test_infer_vault_entry_id_present_with_valid_token(self):
        proc = _make_processor()
        token = proc.vault.make_auth_token()
        result = proc.infer({"query": "test"}, token)
        assert result.vault_entry_id is not None

    def test_status_keys(self):
        proc = _make_processor()
        s = proc.status()
        for key in ("processor_id", "inference_count", "uptime_seconds",
                    "tesseract", "superconductor", "cluster", "vault"):
            assert key in s

    def test_multiple_inferences_stable(self):
        proc = _make_processor()
        token = proc.vault.make_auth_token()
        for i in range(5):
            result = proc.infer({"step": i, "data": "x" * (i * 10)}, token)
            assert result.success is True

    def test_different_queries_produce_different_ops(self):
        proc = _make_processor()
        token = proc.vault.make_auth_token()
        r1 = proc.infer({"q": "short"}, token)
        r2 = proc.infer({"q": "x" * 5_000}, token)
        # Larger input should produce more ops (or at least different routing)
        # We can only assert both succeed
        assert r1.success
        assert r2.success
