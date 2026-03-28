"""
GPG-based authentication for the JessicAi HuggingFace Fork.

Access flow
-----------
1. The caller supplies a *signed token* — a small JSON payload that has been
   clearsigned (``gpg --clearsign``) with the authorised user's private key.
2. :class:`GPGAuthenticator` verifies the signature against the expected
   fingerprint.  If the fingerprint has not been configured via the
   ``JESSICAI_GPG_FINGERPRINT`` environment variable, verification falls back
   to checking that *any* trusted key in the local keyring signed the token.
3. On success the method returns ``True`` and the SecurityMonitor records a
   successful event.  On failure, the attempt is logged and the caller is
   denied full access.

Signed-token format (JSON inside the GPG clearsign envelope)::

    {
        "user": "NaTo1000",
        "timestamp": "<ISO-8601 UTC>",
        "nonce": "<random hex>"
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from huggingface_fork.config import AuthConfig, load_config

logger = logging.getLogger(__name__)

# Maximum age of a signed token in seconds before it is considered stale.
_TOKEN_MAX_AGE_SECONDS = 300


class GPGAuthError(Exception):
    """Raised when GPG authentication fails."""


class GPGAuthenticator:
    """Verify GPG-signed authentication tokens.

    Parameters
    ----------
    config:
        Authentication configuration.  If *None*, the global config is loaded
        from the environment.
    """

    def __init__(self, config: Optional[AuthConfig] = None) -> None:
        self._cfg = config or load_config().auth
        self._gpg_bin = self._find_gpg()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_signed_token(self, signed_token: str) -> bool:
        """Verify a GPG-clearsigned token.

        Parameters
        ----------
        signed_token:
            The full ASCII-armored clearsigned message (as produced by
            ``gpg --clearsign``).

        Returns
        -------
        bool
            ``True`` if the token is validly signed by the authorised key,
            ``False`` otherwise.

        Raises
        ------
        GPGAuthError
            If GPG itself cannot be invoked or returns an unexpected error.
        """
        if not self._gpg_bin:
            raise GPGAuthError(
                "gpg binary not found. Install GnuPG to use GPG authentication."
            )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".asc", delete=False
        ) as tmp:
            tmp.write(signed_token)
            tmp_path = Path(tmp.name)

        try:
            result = self._run_gpg_verify(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        if not result.success:
            logger.warning("GPG signature verification failed: %s", result.stderr)
            return False

        # Extract the signing key's fingerprint from GPG output.
        signing_fp = self._extract_fingerprint(result.stderr)
        if not self._fingerprint_allowed(signing_fp):
            logger.warning(
                "Signing key fingerprint %s is not the authorised key.", signing_fp
            )
            return False

        # Validate the JSON payload inside the token.
        payload = self._extract_payload(signed_token)
        if not self._validate_payload(payload):
            return False

        logger.info("GPG authentication successful for user '%s'.", payload.get("user"))
        return True

    def fingerprint_sha256(self, fingerprint: str) -> str:
        """Return the SHA-256 hash of a key fingerprint for safe logging."""
        return hashlib.sha256(fingerprint.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_gpg(self) -> Optional[str]:
        for candidate in ("gpg2", "gpg"):
            result = subprocess.run(
                ["which", candidate], capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        return None

    def _build_gpg_cmd(self, file_path: Path) -> list[str]:
        cmd = [self._gpg_bin, "--batch", "--no-tty", "--status-fd", "2"]
        if self._cfg.gpg_homedir.exists():
            cmd += ["--homedir", str(self._cfg.gpg_homedir)]
        cmd += ["--verify", str(file_path)]
        return cmd

    class _VerifyResult:
        def __init__(self, success: bool, stderr: str) -> None:
            self.success = success
            self.stderr = stderr

    def _run_gpg_verify(self, file_path: Path) -> "_VerifyResult":
        cmd = self._build_gpg_cmd(file_path)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise GPGAuthError(f"gpg binary not executable: {exc}") from exc
        return self._VerifyResult(
            success=proc.returncode == 0,
            stderr=proc.stderr,
        )

    @staticmethod
    def _extract_fingerprint(gpg_output: str) -> str:
        """Parse the signing key fingerprint from GPG's status output."""
        for line in gpg_output.splitlines():
            # GPG status lines look like: [GNUPG:] VALIDSIG <FP> ...
            if "[GNUPG:]" in line and "VALIDSIG" in line:
                parts = line.split()
                if len(parts) >= 3:
                    return parts[2]
        return ""

    def _fingerprint_allowed(self, fingerprint: str) -> bool:
        """Check whether the signing fingerprint is the authorised one."""
        expected = self._cfg.gpg_fingerprint.upper().replace(" ", "")
        if not expected:
            # No fingerprint configured — any valid signature is accepted.
            return bool(fingerprint)
        return fingerprint.upper().replace(" ", "") == expected

    @staticmethod
    def _extract_payload(signed_token: str) -> dict:
        """Extract the JSON object from inside a GPG clearsign block."""
        lines = signed_token.splitlines()
        in_body = False
        body_lines: list[str] = []
        for line in lines:
            if line.startswith("-----BEGIN PGP SIGNED MESSAGE-----"):
                in_body = False
                continue
            if line == "" and not in_body:
                in_body = True
                continue
            if line.startswith("-----BEGIN PGP SIGNATURE-----"):
                break
            if in_body:
                body_lines.append(line)

        body = "\n".join(body_lines).strip()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            logger.debug("Token body is not valid JSON: %r", body)
            return {}

    def _validate_payload(self, payload: dict) -> bool:
        """Validate the JSON payload fields."""
        import datetime

        if payload.get("user") != self._cfg.authorized_user:
            logger.warning(
                "Token user '%s' does not match authorised user '%s'.",
                payload.get("user"),
                self._cfg.authorized_user,
            )
            return False

        ts_str = payload.get("timestamp", "")
        try:
            ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            now = datetime.datetime.now(datetime.timezone.utc)
            age = abs((now - ts).total_seconds())
            if age > _TOKEN_MAX_AGE_SECONDS:
                logger.warning(
                    "Token is %d seconds old (max %d). Rejecting.",
                    age,
                    _TOKEN_MAX_AGE_SECONDS,
                )
                return False
        except (ValueError, AttributeError):
            logger.warning("Token has invalid or missing timestamp: %r", ts_str)
            return False

        if not payload.get("nonce"):
            logger.warning("Token is missing a nonce.")
            return False

        return True
