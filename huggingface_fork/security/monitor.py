"""
Security monitoring and access-event logging.

Every authentication attempt, model load, dataset fetch, and capability
unlock is recorded in a structured JSON-lines log file stored under
``<project_root>/logs/``.  Log entries are never written to stdout so that
they cannot be intercepted by shell pipe sniffers.

The :class:`SecurityMonitor` also enforces a *failed-attempt lockout*: after
``config.security.max_failed_attempts`` consecutive failures from the same
source identifier, the source is locked out for
``config.security.lockout_seconds`` seconds.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional

from huggingface_fork.config import AppConfig, SecurityConfig, load_config

logger = logging.getLogger(__name__)


class SecurityMonitor:
    """Thread-safe security event monitor and logger.

    Parameters
    ----------
    config:
        Application configuration.  If *None*, the global config is loaded
        from the environment.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        cfg = config or load_config()
        self._sec_cfg: SecurityConfig = cfg.security
        self._log_path = self._sec_cfg.logs_dir / "access_audit.jsonl"
        self._lock = threading.Lock()
        # Map source_id -> (failure_count, lockout_until)
        self._failure_tracker: defaultdict[
            str, tuple[int, Optional[datetime.datetime]]
        ] = defaultdict(lambda: (0, None))
        self._ensure_log_file()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_auth_attempt(
        self,
        *,
        source_id: str,
        auth_type: str,
        success: bool,
        detail: str = "",
    ) -> None:
        """Record an authentication attempt.

        Parameters
        ----------
        source_id:
            An opaque identifier for the caller (e.g., hashed IP address or
            session token).  Never log raw PII.
        auth_type:
            One of ``"gpg"``, ``"voice"``, or ``"combined"``.
        success:
            Whether the attempt succeeded.
        detail:
            Optional human-readable context (sanitised — no secrets).
        """
        self._write_event(
            event_type="auth_attempt",
            source_id=self._hash_id(source_id),
            auth_type=auth_type,
            success=success,
            detail=detail,
        )
        if not success:
            self._record_failure(source_id)
        else:
            self._clear_failures(source_id)

    def record_model_load(
        self,
        *,
        model_id: str,
        source_id: str,
        restricted: bool,
    ) -> None:
        """Record a model-load event."""
        self._write_event(
            event_type="model_load",
            model_id=model_id,
            source_id=self._hash_id(source_id),
            restricted=restricted,
        )

    def record_dataset_load(
        self,
        *,
        dataset_id: str,
        source_id: str,
        restricted: bool,
    ) -> None:
        """Record a dataset-load event."""
        self._write_event(
            event_type="dataset_load",
            dataset_id=dataset_id,
            source_id=self._hash_id(source_id),
            restricted=restricted,
        )

    def record_capability_unlock(
        self,
        *,
        source_id: str,
        success: bool,
        detail: str = "",
    ) -> None:
        """Record a full-capability unlock attempt."""
        self._write_event(
            event_type="capability_unlock",
            source_id=self._hash_id(source_id),
            success=success,
            detail=detail,
        )

    def is_locked_out(self, source_id: str) -> bool:
        """Return ``True`` if *source_id* is currently in a lockout period."""
        with self._lock:
            count, until = self._failure_tracker[source_id]
            if until is None:
                return False
            if datetime.datetime.now(datetime.timezone.utc) < until:
                return True
            # Lockout has expired — reset.
            self._failure_tracker[source_id] = (0, None)
            return False

    def lockout_remaining_seconds(self, source_id: str) -> float:
        """Return seconds remaining in the lockout period (0 if not locked)."""
        with self._lock:
            _, until = self._failure_tracker[source_id]
            if until is None:
                return 0.0
            remaining = (until - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            return max(0.0, remaining)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_failure(self, source_id: str) -> None:
        with self._lock:
            count, _ = self._failure_tracker[source_id]
            count += 1
            if count >= self._sec_cfg.max_failed_attempts:
                until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
                    seconds=self._sec_cfg.lockout_seconds
                )
                self._failure_tracker[source_id] = (count, until)
                logger.warning(
                    "Source %s locked out until %s after %d failed attempts.",
                    self._hash_id(source_id),
                    until.isoformat(),
                    count,
                )
            else:
                self._failure_tracker[source_id] = (count, None)

    def _clear_failures(self, source_id: str) -> None:
        with self._lock:
            self._failure_tracker[source_id] = (0, None)

    def _write_event(self, **fields) -> None:
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **fields,
        }
        if self._sec_cfg.verbose_logging:
            logger.debug("SecurityEvent: %s", json.dumps(record))
        with self._lock:
            with self._log_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")

    def _ensure_log_file(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_path.exists():
            self._log_path.touch(mode=0o600)

    @staticmethod
    def _hash_id(source_id: str) -> str:
        """One-way hash of a source identifier to avoid storing raw PII."""
        return hashlib.sha256(source_id.encode()).hexdigest()[:16]
