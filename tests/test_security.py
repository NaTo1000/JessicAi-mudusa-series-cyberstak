"""Tests for SecurityVault."""

import pytest
from nonganon_tesseract.security import SecurityVault, VaultedContainer


class TestVaultedContainer:
    def test_seal_returns_hex_digest(self):
        vc = VaultedContainer("test", {"key": "value"})
        digest = vc.seal()
        assert len(digest) == 64
        int(digest, 16)

    def test_verify_sealed(self):
        vc = VaultedContainer("test", {"key": "value"})
        vc.seal()
        assert vc.verify() is True

    def test_verify_unsealed_returns_false(self):
        vc = VaultedContainer("test", {"key": "value"})
        assert vc.verify() is False

    def test_verify_after_mutation_returns_false(self):
        vc = VaultedContainer("test", {"key": "value"})
        vc.seal()
        vc._payload["key"] = "tampered"
        assert vc.verify() is False

    def test_release_without_passphrase(self):
        vc = VaultedContainer("test", [1, 2, 3])
        assert vc.release("any") == [1, 2, 3]

    def test_release_correct_passphrase(self):
        vc = VaultedContainer("test", {"secret": True})
        vc.set_passphrase("correcthorsebatterystaple")
        payload = vc.release("correcthorsebatterystaple")
        assert payload["secret"] is True

    def test_release_wrong_passphrase_raises(self):
        vc = VaultedContainer("test", {"data": 1})
        vc.set_passphrase("correct")
        with pytest.raises(PermissionError):
            vc.release("wrong")

    def test_seal_digest_property(self):
        vc = VaultedContainer("test", {})
        assert vc.seal_digest is None
        vc.seal()
        assert vc.seal_digest is not None

    def test_repr_contains_name(self):
        vc = VaultedContainer("my-container", {})
        assert "my-container" in repr(vc)


class TestSecurityVault:
    def test_store_and_retrieve(self):
        sv = SecurityVault("v1")
        sv.store("item", {"val": 1})
        container = sv.get("item")
        assert container.name == "item"

    def test_get_unknown_raises(self):
        sv = SecurityVault("v1")
        with pytest.raises(KeyError):
            sv.get("nonexistent")

    def test_seal_all_returns_digests(self):
        sv = SecurityVault("v1")
        sv.store("a", {"x": 1})
        sv.store("b", {"y": 2})
        digests = sv.seal_all()
        assert "a" in digests
        assert "b" in digests

    def test_verify_all_all_true(self):
        sv = SecurityVault("v1")
        sv.store("a", {"x": 1})
        sv.store("b", {"y": 2})
        sv.seal_all()
        results = sv.verify_all()
        assert all(results.values())

    def test_manifest_structure(self):
        sv = SecurityVault("v1")
        sv.store("a", {"x": 1})
        sv.seal_all()
        manifest = sv.manifest()
        assert manifest["a"]["sealed"] is True
        assert len(manifest["a"]["digest"]) == 64

    def test_passphrase_protected_container(self):
        sv = SecurityVault("v2")
        sv.store("secret-item", {"confidential": True}, passphrase="pass123")
        container = sv.get("secret-item")
        with pytest.raises(PermissionError):
            container.release("wrong")
        payload = container.release("pass123")
        assert payload["confidential"] is True

    def test_repr_contains_vault_id(self):
        sv = SecurityVault("my-vault")
        assert "my-vault" in repr(sv)
