"""
Tests for GPG authentication.

These tests do not require a real GPG key — they validate the parsing and
validation logic using well-formed and malformed fixture data.
"""

from __future__ import annotations

import datetime
import json
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from huggingface_fork.auth.gpg_auth import GPGAuthenticator
from huggingface_fork.config import AuthConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(fingerprint: str = "") -> AuthConfig:
    cfg = AuthConfig()
    cfg.gpg_fingerprint = fingerprint
    cfg.authorized_user = "NaTo1000"
    return cfg


def _make_token(user: str = "NaTo1000", nonce: str = "abc123") -> str:
    """Return a minimal clearsign-like token (not GPG-signed, for payload tests)."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = json.dumps({"user": user, "timestamp": ts, "nonce": nonce})
    return textwrap.dedent(f"""\
        -----BEGIN PGP SIGNED MESSAGE-----
        Hash: SHA256

        {payload}
        -----BEGIN PGP SIGNATURE-----

        iDummySignatureData==
        -----END PGP SIGNATURE-----
    """)


# ---------------------------------------------------------------------------
# _extract_payload tests
# ---------------------------------------------------------------------------

class TestExtractPayload:
    def test_valid_json_payload(self):
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        token = _make_token("NaTo1000")
        payload = GPGAuthenticator._extract_payload(token)
        assert payload["user"] == "NaTo1000"
        assert "timestamp" in payload
        assert payload["nonce"] == "abc123"

    def test_invalid_json_returns_empty_dict(self):
        token = textwrap.dedent("""\
            -----BEGIN PGP SIGNED MESSAGE-----
            Hash: SHA256

            not-valid-json
            -----BEGIN PGP SIGNATURE-----
            -----END PGP SIGNATURE-----
        """)
        assert GPGAuthenticator._extract_payload(token) == {}

    def test_empty_body_returns_empty_dict(self):
        token = textwrap.dedent("""\
            -----BEGIN PGP SIGNED MESSAGE-----

            -----BEGIN PGP SIGNATURE-----
            -----END PGP SIGNATURE-----
        """)
        assert GPGAuthenticator._extract_payload(token) == {}


# ---------------------------------------------------------------------------
# _validate_payload tests
# ---------------------------------------------------------------------------

class TestValidatePayload:
    def setup_method(self):
        self.auth = GPGAuthenticator(_make_config())

    def _fresh_payload(self, **overrides):
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        p = {"user": "NaTo1000", "timestamp": ts, "nonce": "deadbeef"}
        p.update(overrides)
        return p

    def test_valid_payload_passes(self):
        assert self.auth._validate_payload(self._fresh_payload()) is True

    def test_wrong_user_fails(self):
        assert self.auth._validate_payload(self._fresh_payload(user="attacker")) is False

    def test_missing_nonce_fails(self):
        p = self._fresh_payload()
        del p["nonce"]
        assert self.auth._validate_payload(p) is False

    def test_stale_timestamp_fails(self):
        old_ts = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(seconds=400)
        ).isoformat()
        assert self.auth._validate_payload(self._fresh_payload(timestamp=old_ts)) is False

    def test_invalid_timestamp_fails(self):
        assert self.auth._validate_payload(self._fresh_payload(timestamp="not-a-date")) is False

    def test_empty_payload_fails(self):
        assert self.auth._validate_payload({}) is False


# ---------------------------------------------------------------------------
# _fingerprint_allowed tests
# ---------------------------------------------------------------------------

class TestFingerprintAllowed:
    def test_no_configured_fingerprint_allows_any_valid(self):
        auth = GPGAuthenticator(_make_config(fingerprint=""))
        assert auth._fingerprint_allowed("AABBCCDD") is True

    def test_no_configured_fingerprint_rejects_empty(self):
        auth = GPGAuthenticator(_make_config(fingerprint=""))
        assert auth._fingerprint_allowed("") is False

    def test_matching_fingerprint_allowed(self):
        fp = "AABBCCDDEEFF00112233445566778899AABBCCDD"
        auth = GPGAuthenticator(_make_config(fingerprint=fp))
        assert auth._fingerprint_allowed(fp) is True

    def test_non_matching_fingerprint_rejected(self):
        auth = GPGAuthenticator(_make_config(fingerprint="AABBCCDDEEFF00112233445566778899AABBCCDD"))
        assert auth._fingerprint_allowed("DEADBEEF") is False

    def test_case_insensitive(self):
        fp = "aabbccddeeff00112233445566778899aabbccdd"
        auth = GPGAuthenticator(_make_config(fingerprint=fp.upper()))
        assert auth._fingerprint_allowed(fp.lower()) is True


# ---------------------------------------------------------------------------
# verify_signed_token integration (mocked GPG)
# ---------------------------------------------------------------------------

class TestVerifySignedToken:
    def _auth_with_mock_gpg(self, fingerprint: str = "") -> GPGAuthenticator:
        auth = GPGAuthenticator(_make_config(fingerprint=fingerprint))
        auth._gpg_bin = "/usr/bin/gpg"
        return auth

    def test_gpg_failure_returns_false(self):
        auth = self._auth_with_mock_gpg()
        mock_result = auth._VerifyResult(success=False, stderr="BADSIG")
        with patch.object(auth, "_run_gpg_verify", return_value=mock_result):
            assert auth.verify_signed_token(_make_token()) is False

    def test_gpg_success_but_wrong_fingerprint_returns_false(self):
        auth = self._auth_with_mock_gpg(fingerprint="EXPECTED_FP")
        mock_result = auth._VerifyResult(
            success=True,
            stderr="[GNUPG:] VALIDSIG WRONG_FINGERPRINT 2024-01-01 ...",
        )
        with patch.object(auth, "_run_gpg_verify", return_value=mock_result):
            assert auth.verify_signed_token(_make_token()) is False

    def test_gpg_success_correct_fingerprint_valid_payload(self):
        fp = "AABBCCDDEEFF00112233445566778899AABBCCDD"
        auth = self._auth_with_mock_gpg(fingerprint=fp)
        mock_result = auth._VerifyResult(
            success=True,
            stderr=f"[GNUPG:] VALIDSIG {fp} 2024-01-01 ...",
        )
        with patch.object(auth, "_run_gpg_verify", return_value=mock_result):
            assert auth.verify_signed_token(_make_token()) is True

    def test_no_gpg_binary_raises(self):
        auth = GPGAuthenticator(_make_config())
        auth._gpg_bin = None
        with pytest.raises(Exception):
            auth.verify_signed_token(_make_token())
