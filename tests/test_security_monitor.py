"""
Tests for SecurityMonitor.
"""

from __future__ import annotations

import datetime
import json
import tempfile
from pathlib import Path

import pytest

from huggingface_fork.config import AppConfig, SecurityConfig
from huggingface_fork.security.monitor import SecurityMonitor


def _make_monitor(
    max_failed: int = 3,
    lockout_sec: int = 60,
    verbose: bool = False,
    tmp_dir: Path = None,
) -> SecurityMonitor:
    cfg = AppConfig()
    cfg.security.max_failed_attempts = max_failed
    cfg.security.lockout_seconds = lockout_sec
    cfg.security.verbose_logging = verbose
    if tmp_dir:
        cfg.security.logs_dir = tmp_dir
    return SecurityMonitor(cfg)


class TestSecurityMonitor:
    def setup_method(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_audit_log_created(self):
        monitor = _make_monitor(tmp_dir=self.tmp_path)
        log_path = self.tmp_path / "access_audit.jsonl"
        assert log_path.exists()

    def test_auth_attempt_logged(self):
        monitor = _make_monitor(tmp_dir=self.tmp_path)
        monitor.record_auth_attempt(
            source_id="sess1",
            auth_type="gpg",
            success=True,
        )
        log_path = self.tmp_path / "access_audit.jsonl"
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event_type"] == "auth_attempt"
        assert record["auth_type"] == "gpg"
        assert record["success"] is True
        assert "timestamp" in record

    def test_source_id_hashed(self):
        monitor = _make_monitor(tmp_dir=self.tmp_path)
        monitor.record_auth_attempt(
            source_id="my-secret-session-id",
            auth_type="voice",
            success=False,
        )
        log_path = self.tmp_path / "access_audit.jsonl"
        content = log_path.read_text()
        assert "my-secret-session-id" not in content

    def test_no_lockout_initially(self):
        monitor = _make_monitor(tmp_dir=self.tmp_path)
        assert monitor.is_locked_out("sess1") is False
        assert monitor.lockout_remaining_seconds("sess1") == 0.0

    def test_lockout_after_max_failures(self):
        monitor = _make_monitor(max_failed=3, lockout_sec=300, tmp_dir=self.tmp_path)
        for _ in range(3):
            monitor.record_auth_attempt(
                source_id="attacker", auth_type="gpg", success=False
            )
        assert monitor.is_locked_out("attacker") is True
        assert monitor.lockout_remaining_seconds("attacker") > 0

    def test_success_clears_failure_count(self):
        monitor = _make_monitor(max_failed=3, lockout_sec=300, tmp_dir=self.tmp_path)
        monitor.record_auth_attempt(source_id="sess1", auth_type="gpg", success=False)
        monitor.record_auth_attempt(source_id="sess1", auth_type="gpg", success=False)
        monitor.record_auth_attempt(source_id="sess1", auth_type="gpg", success=True)
        assert monitor.is_locked_out("sess1") is False

    def test_model_load_logged(self):
        monitor = _make_monitor(tmp_dir=self.tmp_path)
        monitor.record_model_load(
            model_id="bert-base-uncased", source_id="sess1", restricted=False
        )
        log_path = self.tmp_path / "access_audit.jsonl"
        content = log_path.read_text()
        records = [json.loads(l) for l in content.strip().splitlines()]
        model_records = [r for r in records if r["event_type"] == "model_load"]
        assert len(model_records) == 1
        assert model_records[0]["model_id"] == "bert-base-uncased"

    def test_dataset_load_logged(self):
        monitor = _make_monitor(tmp_dir=self.tmp_path)
        monitor.record_dataset_load(
            dataset_id="squad", source_id="sess1", restricted=True
        )
        log_path = self.tmp_path / "access_audit.jsonl"
        records = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
        ds_records = [r for r in records if r["event_type"] == "dataset_load"]
        assert len(ds_records) == 1
        assert ds_records[0]["restricted"] is True

    def test_capability_unlock_logged(self):
        monitor = _make_monitor(tmp_dir=self.tmp_path)
        monitor.record_capability_unlock(source_id="sess1", success=True)
        log_path = self.tmp_path / "access_audit.jsonl"
        records = [json.loads(l) for l in log_path.read_text().strip().splitlines()]
        unlock_records = [r for r in records if r["event_type"] == "capability_unlock"]
        assert unlock_records[0]["success"] is True

    def test_hash_id_is_deterministic(self):
        h1 = SecurityMonitor._hash_id("same-id")
        h2 = SecurityMonitor._hash_id("same-id")
        assert h1 == h2

    def test_hash_id_different_inputs(self):
        h1 = SecurityMonitor._hash_id("id-one")
        h2 = SecurityMonitor._hash_id("id-two")
        assert h1 != h2
