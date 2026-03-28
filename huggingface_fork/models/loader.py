"""
HuggingFace model loader with access-control integration.

The :class:`ModelLoader` is the primary entry-point for loading any model
from the HuggingFace Hub.  It wraps ``transformers.AutoModel`` (and its
task-specific siblings) and adds:

* **Authentication gating** — full unrestricted loading (including private
  models and gated repositories) is only available after :meth:`unlock`
  has been called with a valid combined GPG + voice authentication.
* **Audit logging** — every load attempt is recorded by the
  :class:`~huggingface_fork.security.monitor.SecurityMonitor`.
* **Offline / restricted mode** — unauthenticated callers can still load
  freely downloadable public models but cannot perform Hub API calls that
  require a token.
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


class ModelLoader:
    """Load HuggingFace models with tiered access control.

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
        """Unlock full capabilities via GPG + voice authentication.

        Parameters
        ----------
        signed_token:
            GPG-clearsigned JSON token (see :mod:`huggingface_fork.auth.gpg_auth`).
        audio_file:
            Path to a WAV voice sample.  If *None*, the microphone is used.

        Returns
        -------
        bool
            ``True`` if both factors pass and the caller is now in *unlocked* mode.
        """
        if self._monitor.is_locked_out(self._session_id):
            remaining = self._monitor.lockout_remaining_seconds(self._session_id)
            logger.warning(
                "Session locked out.  Try again in %.0f seconds.", remaining
            )
            return False

        # Factor 1: GPG signature.
        gpg_ok = self._gpg_auth.verify_signed_token(signed_token)
        self._monitor.record_auth_attempt(
            source_id=self._session_id,
            auth_type="gpg",
            success=gpg_ok,
        )
        if not gpg_ok:
            logger.warning("GPG authentication failed.")
            return False

        # Factor 2: Voice validation.
        voice_ok = self._voice_auth.verify(audio_file)
        self._monitor.record_auth_attempt(
            source_id=self._session_id,
            auth_type="voice",
            success=voice_ok,
        )
        if not voice_ok:
            logger.warning("Voice authentication failed.")
            return False

        self._unlocked = True
        self._monitor.record_capability_unlock(
            source_id=self._session_id,
            success=True,
            detail="Full model-loading capabilities unlocked.",
        )
        logger.info("Full model-loading capabilities UNLOCKED.")
        return True

    @property
    def is_unlocked(self) -> bool:
        """Whether this loader instance has passed combined authentication."""
        return self._unlocked

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(
        self,
        model_id: str,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Load a model from the HuggingFace Hub.

        Parameters
        ----------
        model_id:
            HuggingFace model identifier (e.g. ``"bert-base-uncased"``).
        task:
            Optional task string (e.g. ``"text-classification"``).  When
            provided, ``transformers.pipeline`` is used instead of a bare
            ``AutoModel`` load.
        **kwargs:
            Additional keyword arguments forwarded to the underlying
            ``transformers`` loading call.

        Returns
        -------
        Any
            The loaded model or pipeline object.

        Raises
        ------
        ImportError
            If the ``transformers`` package is not installed.
        PermissionError
            If the model requires authentication and the loader is not
            in an unlocked state.
        """
        try:
            import transformers  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. "
                "Install it with: pip install transformers"
            ) from exc

        restricted = not self._unlocked
        self._monitor.record_model_load(
            model_id=model_id,
            source_id=self._session_id,
            restricted=restricted,
        )

        token: Optional[str] = None
        if self._unlocked and self._cfg.huggingface.hub_token:
            token = self._cfg.huggingface.hub_token
            # Enable internet access for authenticated users.
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
        else:
            # Unauthenticated: use locally cached files only.
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        cache_dir = str(self._cfg.huggingface.cache_dir)

        if task:
            logger.info("Loading pipeline for task '%s', model '%s'.", task, model_id)
            return transformers.pipeline(
                task,
                model=model_id,
                token=token,
                cache_dir=cache_dir,
                **kwargs,
            )

        logger.info("Loading model '%s'.", model_id)
        return transformers.AutoModel.from_pretrained(
            model_id,
            token=token,
            cache_dir=cache_dir,
            **kwargs,
        )

    def load_tokenizer(self, model_id: str, **kwargs: Any) -> Any:
        """Load the tokenizer associated with *model_id*.

        Parameters
        ----------
        model_id:
            HuggingFace model identifier.
        **kwargs:
            Additional keyword arguments forwarded to
            ``transformers.AutoTokenizer.from_pretrained``.

        Returns
        -------
        Any
            The loaded tokenizer.
        """
        try:
            from transformers import AutoTokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. "
                "Install it with: pip install transformers"
            ) from exc

        token: Optional[str] = None
        if self._unlocked and self._cfg.huggingface.hub_token:
            token = self._cfg.huggingface.hub_token

        cache_dir = str(self._cfg.huggingface.cache_dir)
        logger.info("Loading tokenizer for '%s'.", model_id)
        return AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            cache_dir=cache_dir,
            **kwargs,
        )
