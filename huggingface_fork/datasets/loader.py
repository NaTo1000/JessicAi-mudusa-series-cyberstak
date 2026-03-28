"""
HuggingFace dataset loader with access-control integration.

The :class:`DatasetLoader` wraps the ``datasets`` library and applies the
same authentication-gating and audit-logging as :class:`ModelLoader`.

Unauthenticated callers can only access locally cached (offline) datasets.
Authenticated (unlocked) callers can stream and download any publicly
available dataset from the HuggingFace Hub, using the configured API token
for private or gated datasets.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from huggingface_fork.auth.gpg_auth import GPGAuthenticator
from huggingface_fork.auth.voice_auth import VoiceAuthenticator
from huggingface_fork.config import AppConfig, load_config
from huggingface_fork.security.monitor import SecurityMonitor

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load HuggingFace datasets with tiered access control.

    Parameters
    ----------
    config:
        Application configuration.  If *None*, the global config is loaded.
    session_id:
        Opaque string used as the *source_id* for audit log entries.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        session_id: str = "anonymous",
    ) -> None:
        self._cfg = config or load_config()
        self._session_id = session_id
        self._unlocked = False
        self._monitor = SecurityMonitor(self._cfg)
        self._gpg_auth = GPGAuthenticator(self._cfg.auth)
        self._voice_auth = VoiceAuthenticator(self._cfg.auth)

    # ------------------------------------------------------------------
    # Authentication / unlock
    # ------------------------------------------------------------------

    def unlock(
        self,
        signed_token: str,
        audio_file: Optional[Path] = None,
    ) -> bool:
        """Unlock full dataset capabilities via GPG + voice authentication.

        Parameters
        ----------
        signed_token:
            GPG-clearsigned JSON token.
        audio_file:
            Path to a WAV voice sample or *None* to use the microphone.

        Returns
        -------
        bool
            ``True`` if both factors pass.
        """
        if self._monitor.is_locked_out(self._session_id):
            remaining = self._monitor.lockout_remaining_seconds(self._session_id)
            logger.warning(
                "Session locked out.  Try again in %.0f seconds.", remaining
            )
            return False

        gpg_ok = self._gpg_auth.verify_signed_token(signed_token)
        self._monitor.record_auth_attempt(
            source_id=self._session_id,
            auth_type="gpg",
            success=gpg_ok,
        )
        if not gpg_ok:
            return False

        voice_ok = self._voice_auth.verify(audio_file)
        self._monitor.record_auth_attempt(
            source_id=self._session_id,
            auth_type="voice",
            success=voice_ok,
        )
        if not voice_ok:
            return False

        self._unlocked = True
        self._monitor.record_capability_unlock(
            source_id=self._session_id,
            success=True,
            detail="Full dataset-loading capabilities unlocked.",
        )
        logger.info("Full dataset-loading capabilities UNLOCKED.")
        return True

    @property
    def is_unlocked(self) -> bool:
        """Whether this loader instance has passed combined authentication."""
        return self._unlocked

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load(
        self,
        dataset_id: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        streaming: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Load a dataset from the HuggingFace Hub.

        Parameters
        ----------
        dataset_id:
            HuggingFace dataset identifier (e.g. ``"squad"``).
        subset:
            Dataset configuration / subset name (e.g. ``"plain_text"``).
        split:
            Dataset split to load (e.g. ``"train"``, ``"validation"``).
        streaming:
            If ``True``, return an iterable streaming dataset without
            downloading the full corpus.
        **kwargs:
            Additional keyword arguments forwarded to
            ``datasets.load_dataset``.

        Returns
        -------
        Any
            A ``datasets.DatasetDict``, ``datasets.Dataset``, or
            ``datasets.IterableDatasetDict`` depending on parameters.

        Raises
        ------
        ImportError
            If the ``datasets`` package is not installed.
        PermissionError
            If the dataset requires authentication and the loader is not
            in an unlocked state.
        """
        try:
            import datasets as hf_datasets  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "datasets is not installed. "
                "Install it with: pip install datasets"
            ) from exc

        restricted = not self._unlocked
        self._monitor.record_dataset_load(
            dataset_id=dataset_id,
            source_id=self._session_id,
            restricted=restricted,
        )

        token: Optional[str] = None
        if self._unlocked and self._cfg.huggingface.hub_token:
            token = self._cfg.huggingface.hub_token
            os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
        else:
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        cache_dir = str(self._cfg.huggingface.cache_dir)

        load_kwargs: dict[str, Any] = dict(
            path=dataset_id,
            cache_dir=cache_dir,
            streaming=streaming,
            token=token,
            **kwargs,
        )
        if subset is not None:
            load_kwargs["name"] = subset
        if split is not None:
            load_kwargs["split"] = split

        logger.info("Loading dataset '%s' (subset=%s, split=%s).", dataset_id, subset, split)
        return hf_datasets.load_dataset(**load_kwargs)

    def list_datasets(
        self,
        search: Optional[str] = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[Any]:
        """List datasets available on the HuggingFace Hub.

        Requires an unlocked (authenticated) session for Hub API access.

        Parameters
        ----------
        search:
            Optional search query string.
        limit:
            Maximum number of results to return.
        **kwargs:
            Additional keyword arguments forwarded to
            ``huggingface_hub.list_datasets``.

        Returns
        -------
        list
            List of dataset metadata objects.
        """
        if not self._unlocked:
            logger.warning("Hub API listing requires unlocked access.")
            return []

        try:
            from huggingface_hub import list_datasets  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is not installed. "
                "Install it with: pip install huggingface_hub"
            ) from exc

        token = self._cfg.huggingface.hub_token or None
        results = list(
            list_datasets(search=search, token=token, limit=limit, **kwargs)
        )
        logger.info("Found %d datasets matching query '%s'.", len(results), search)
        return results
