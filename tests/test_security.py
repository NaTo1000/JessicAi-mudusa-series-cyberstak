"""
Tests for the encryption, tunnel, and honeypot security layer.
"""
from __future__ import annotations

import hashlib
import os

import pytest

from security.encryption import (
    MeshEncryption,
    KeyPair,
    generate_keypair,
    sha256_of,
    sha256_file,
    derive_session_key,
)


class TestSha256:
    def test_known_hash(self):
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        assert sha256_of(data) == expected

    def test_empty(self):
        assert sha256_of(b"") == hashlib.sha256(b"").hexdigest()

    def test_sha256_file(self, tmp_path):
        p = tmp_path / "test.bin"
        p.write_bytes(b"abc")
        assert sha256_file(p) == hashlib.sha256(b"abc").hexdigest()


class TestKeyPair:
    def test_generate_keypair_returns_32_bytes(self):
        kp = generate_keypair()
        assert isinstance(kp, KeyPair)
        assert len(kp.private_key_bytes) == 32
        assert len(kp.public_key_bytes) == 32

    def test_fingerprint_is_16_chars(self):
        kp = generate_keypair()
        assert len(kp.fingerprint) == 16

    def test_two_keypairs_differ(self):
        kp1 = generate_keypair()
        kp2 = generate_keypair()
        assert kp1.public_key_bytes != kp2.public_key_bytes


class TestDeriveSessionKey:
    def test_produces_32_bytes(self):
        kp1 = generate_keypair()
        kp2 = generate_keypair()
        sk = derive_session_key(kp1.private_key_bytes, kp2.public_key_bytes)
        assert isinstance(sk, bytes)
        assert len(sk) == 32

    def test_deterministic(self):
        kp1 = generate_keypair()
        kp2 = generate_keypair()
        sk1 = derive_session_key(kp1.private_key_bytes, kp2.public_key_bytes)
        sk2 = derive_session_key(kp1.private_key_bytes, kp2.public_key_bytes)
        assert sk1 == sk2

    def test_different_keys_produce_different_session_key(self):
        kp1 = generate_keypair()
        kp2 = generate_keypair()
        kp3 = generate_keypair()
        sk1 = derive_session_key(kp1.private_key_bytes, kp2.public_key_bytes)
        sk2 = derive_session_key(kp1.private_key_bytes, kp3.public_key_bytes)
        assert sk1 != sk2


class TestMeshEncryption:
    def setup_method(self):
        self.key = os.urandom(32)
        self.enc = MeshEncryption(self.key)

    def test_encrypt_decrypt_roundtrip(self):
        plaintext = b"secret message"
        ct = self.enc.encrypt(plaintext)
        assert ct != plaintext
        pt = self.enc.decrypt(ct)
        assert pt == plaintext

    def test_wrong_key_cannot_decrypt(self):
        ct = self.enc.encrypt(b"data")
        other = MeshEncryption(os.urandom(32))
        with pytest.raises(Exception):
            other.decrypt(ct)

    def test_invalid_key_length(self):
        with pytest.raises(ValueError):
            MeshEncryption(b"short")

    def test_sign_manifest(self):
        data = b"manifest content"
        sig = self.enc.sign_manifest(data)
        assert isinstance(sig, str)
        assert len(sig) == 64  # hex SHA-256

    def test_verify_manifest_valid(self):
        data = b"manifest content"
        sig = self.enc.sign_manifest(data)
        assert self.enc.verify_manifest(data, sig)

    def test_verify_manifest_tampered(self):
        data = b"manifest content"
        sig = self.enc.sign_manifest(data)
        assert not self.enc.verify_manifest(b"different", sig)

    def test_encrypt_large_payload(self):
        plaintext = os.urandom(65536)
        ct = self.enc.encrypt(plaintext)
        assert self.enc.decrypt(ct) == plaintext
