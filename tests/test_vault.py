"""Tests for the non-ganon triple-coded vault."""

from __future__ import annotations

import time

import pytest

from src.architecture.vault import (
    TripleCodedVault,
    VaultConfig,
    VaultEntry,
    VaultState,
    AuthResult,
)


class TestVaultEntry:
    def test_touch_increments_access(self):
        entry = VaultEntry()
        assert entry.access_count == 0
        entry.touch()
        assert entry.access_count == 1
        entry.touch()
        assert entry.access_count == 2

    def test_touch_sets_last_accessed(self):
        entry = VaultEntry()
        assert entry.last_accessed is None
        entry.touch()
        assert entry.last_accessed is not None


class TestTripleCodedVault:
    def _make(self, **kwargs) -> TripleCodedVault:
        cfg = VaultConfig(**kwargs)
        return TripleCodedVault(config=cfg)

    def test_initial_state_locked(self):
        v = self._make()
        assert v.state == VaultState.LOCKED

    def test_valid_token_grants_access(self):
        v = self._make()
        token = v.make_auth_token()
        result = v.unlock(token)
        assert result == AuthResult.GRANTED

    def test_valid_token_transitions_to_unlocked(self):
        v = self._make()
        token = v.make_auth_token()
        v.unlock(token)
        assert v.state == VaultState.UNLOCKED

    def test_invalid_token_denies_access(self):
        v = self._make()
        result = v.unlock("completely-wrong-token")
        assert result == AuthResult.DENIED

    def test_failed_attempts_tracked(self):
        v = self._make()
        v.unlock("bad-token")
        assert v._failed_attempts == 1

    def test_quarantine_after_max_failed_attempts(self):
        v = self._make(max_failed_attempts=3)
        for _ in range(3):
            v.unlock("bad-token")
        assert v.state == VaultState.QUARANTINED

    def test_honeypot_returned_when_quarantined(self):
        v = self._make(max_failed_attempts=1)
        v.unlock("bad")
        assert v.state == VaultState.QUARANTINED
        # Even with valid token, quarantined vault returns honeypot
        token = v.make_auth_token()
        result = v.unlock(token)
        assert result == AuthResult.HONEYPOT

    def test_store_and_retrieve(self):
        v = self._make()
        token = v.make_auth_token()
        entry = VaultEntry(payload={"data": "hello"})
        store_result = v.store(entry, token)
        assert store_result == AuthResult.GRANTED

        retrieve_result, retrieved = v.retrieve(entry.entry_id, token)
        assert retrieve_result == AuthResult.GRANTED
        assert retrieved is not None
        assert retrieved.payload == {"data": "hello"}

    def test_retrieve_increments_access_count(self):
        v = self._make()
        token = v.make_auth_token()
        entry = VaultEntry(payload={})
        v.store(entry, token)
        v.retrieve(entry.entry_id, token)
        assert entry.access_count == 1

    def test_retrieve_missing_entry_returns_none(self):
        v = self._make()
        token = v.make_auth_token()
        v.unlock(token)   # unlock first
        result, retrieved = v.retrieve("nonexistent-id", token)
        assert result == AuthResult.GRANTED
        assert retrieved is None

    def test_notify_stress_quarantines(self):
        v = self._make()
        v.notify_stress("test_stress")
        assert v.state == VaultState.QUARANTINED

    def test_notify_stress_disabled(self):
        v = self._make(auto_quarantine_on_stress=False)
        v.notify_stress("should be ignored")
        assert v.state == VaultState.LOCKED

    def test_containment_transfer_manifest(self):
        v = self._make()
        token = v.make_auth_token()
        entry = VaultEntry(payload={"x": 1})
        v.store(entry, token)
        v.notify_stress("initiate transfer")
        manifest = v.initiate_containment_transfer()
        assert manifest["vault_id"] == v.vault_id
        assert manifest["entry_count"] == 1
        assert v.state == VaultState.TRANSFERRING

    def test_rotate_secret_changes_token(self):
        v = self._make()
        token1 = v.make_auth_token()
        v.rotate_secret()
        token2 = v.make_auth_token()
        assert token1 != token2

    def test_status_keys(self):
        v = self._make()
        s = v.status()
        for key in ("vault_id", "state", "entry_count", "failed_attempts"):
            assert key in s

    def test_session_secret_is_hex(self):
        v = self._make()
        secret = v.session_secret
        # Should be 64-char hex string (32 bytes)
        assert len(secret) == 64
        int(secret, 16)   # raises if not valid hex
