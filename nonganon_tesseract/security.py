"""
Security Vault
===============

Provides secured, self-contained containers for clusters and chains.
Each :class:`VaultedContainer` wraps an arbitrary Python object and
protects it with:

* **SHA-256 integrity sealing** – a digest is computed over the
  serialised representation of the payload at seal time.  Any
  mutation after sealing is detectable via :meth:`verify`.
* **GPG-armored passphrase guard** – when a passphrase is set the
  container refuses to release its contents until the correct
  passphrase is supplied.  The passphrase itself is never stored in
  plaintext; only its SHA-256 digest is retained.

The :class:`SecurityVault` acts as a registry of named
:class:`VaultedContainer` objects and provides bulk seal / verify
helpers.

Note on GPG
-----------
Full asymmetric GPG encryption of container *contents* would require
a GPG key-pair and the ``gnupg`` library (or ``subprocess`` calls to
``gpg``).  This implementation instead uses a *SHA-256–based
passphrase guard* (HMAC-style) so the module remains dependency-free
and portable.  The design leaves a clear extension point
(:meth:`VaultedContainer.set_passphrase`) for callers who wish to
bolt on full GPG encryption.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VaultedContainer
# ---------------------------------------------------------------------------


class VaultedContainer:
    """A sealed, optionally passphrase-protected container.

    Parameters
    ----------
    name:
        Human-readable identifier for this container.
    payload:
        The object to protect.  Must be JSON-serialisable for the
        integrity seal to work.

    Examples
    --------
    >>> vc = VaultedContainer("my-cluster", {"centroid": [0.5, 0.5]})
    >>> vc.seal()
    >>> vc.verify()
    True
    >>> vc.set_passphrase("s3cr3t")
    >>> vc.release("s3cr3t")
    {'centroid': [0.5, 0.5]}
    """

    def __init__(self, name: str, payload: Any) -> None:
        self.name = name
        self._payload = payload
        self._seal: Optional[str] = None
        self._passphrase_hash: Optional[str] = None

    # ------------------------------------------------------------------

    def seal(self) -> str:
        """Compute and store a SHA-256 integrity seal for the payload.

        Returns
        -------
        str
            The hex-encoded SHA-256 digest.
        """
        raw = json.dumps(self._payload, sort_keys=True, default=str).encode()
        self._seal = hashlib.sha256(raw).hexdigest()
        logger.debug("VaultedContainer %r sealed: %s", self.name, self._seal)
        return self._seal

    # ------------------------------------------------------------------

    def verify(self) -> bool:
        """Return ``True`` if the payload matches the stored seal.

        Returns ``False`` if the container has not been sealed yet or
        if the payload has been altered since sealing.
        """
        if self._seal is None:
            return False
        raw = json.dumps(self._payload, sort_keys=True, default=str).encode()
        current = hashlib.sha256(raw).hexdigest()
        result = hmac.compare_digest(self._seal, current)
        logger.debug(
            "VaultedContainer %r verify=%s.", self.name, result
        )
        return result

    # ------------------------------------------------------------------

    def set_passphrase(self, passphrase: str) -> None:
        """Set a passphrase guard on this container.

        The passphrase is stored as its SHA-256 digest — never in
        plaintext.

        Parameters
        ----------
        passphrase:
            Arbitrary string passphrase.
        """
        self._passphrase_hash = hashlib.sha256(passphrase.encode()).hexdigest()
        logger.debug("VaultedContainer %r passphrase guard set.", self.name)

    # ------------------------------------------------------------------

    def release(self, passphrase: str) -> Any:
        """Return the payload if *passphrase* is correct (or not set).

        Parameters
        ----------
        passphrase:
            The passphrase to verify.

        Returns
        -------
        Any
            The protected payload.

        Raises
        ------
        PermissionError
            If the passphrase is incorrect.
        """
        if self._passphrase_hash is not None:
            candidate = hashlib.sha256(passphrase.encode()).hexdigest()
            if not hmac.compare_digest(self._passphrase_hash, candidate):
                raise PermissionError(
                    f"VaultedContainer {self.name!r}: incorrect passphrase."
                )
        return self._payload

    # ------------------------------------------------------------------

    @property
    def seal_digest(self) -> Optional[str]:
        """The SHA-256 seal digest, or ``None`` if not yet sealed."""
        return self._seal

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VaultedContainer(name={self.name!r}, sealed={self._seal is not None}, "
            f"passphrase_protected={self._passphrase_hash is not None})"
        )


# ---------------------------------------------------------------------------
# SecurityVault
# ---------------------------------------------------------------------------


class SecurityVault:
    """Registry of named :class:`VaultedContainer` objects.

    Parameters
    ----------
    vault_id:
        Human-readable identifier for this vault.

    Examples
    --------
    >>> sv = SecurityVault("main-vault")
    >>> sv.store("cluster-alpha", {"centroid": [1.0, 2.0]})
    >>> sv.seal_all()
    >>> sv.verify_all()
    {'cluster-alpha': True}
    """

    def __init__(self, vault_id: str = "nonganon-vault") -> None:
        self.vault_id = vault_id
        self._containers: Dict[str, VaultedContainer] = {}

    # ------------------------------------------------------------------

    def store(
        self,
        name: str,
        payload: Any,
        passphrase: Optional[str] = None,
    ) -> VaultedContainer:
        """Create and register a :class:`VaultedContainer`.

        Parameters
        ----------
        name:
            Unique name for the container within this vault.
        payload:
            Object to store.
        passphrase:
            If provided, the container will be passphrase-protected.

        Returns
        -------
        VaultedContainer
            The newly created container.
        """
        container = VaultedContainer(name, payload)
        if passphrase is not None:
            container.set_passphrase(passphrase)
        self._containers[name] = container
        logger.debug("SecurityVault %r: stored %r.", self.vault_id, name)
        return container

    # ------------------------------------------------------------------

    def seal_all(self) -> Dict[str, str]:
        """Seal every container and return a map of ``{name: digest}``."""
        return {name: c.seal() for name, c in self._containers.items()}

    # ------------------------------------------------------------------

    def verify_all(self) -> Dict[str, bool]:
        """Verify integrity of every container and return a status map."""
        return {name: c.verify() for name, c in self._containers.items()}

    # ------------------------------------------------------------------

    def get(self, name: str) -> VaultedContainer:
        """Retrieve the :class:`VaultedContainer` with *name*.

        Raises
        ------
        KeyError
            If no container with *name* is found.
        """
        if name not in self._containers:
            raise KeyError(
                f"SecurityVault {self.vault_id!r}: unknown container {name!r}."
            )
        return self._containers[name]

    # ------------------------------------------------------------------

    def manifest(self) -> Dict[str, Dict[str, Any]]:
        """Return a JSON-serialisable manifest of all containers."""
        return {
            name: {
                "sealed": c.seal_digest is not None,
                "digest": c.seal_digest,
                "passphrase_protected": c._passphrase_hash is not None,
            }
            for name, c in self._containers.items()
        }

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SecurityVault(id={self.vault_id!r}, "
            f"containers={list(self._containers.keys())})"
        )
