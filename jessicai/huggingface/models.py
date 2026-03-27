"""
Model management: fetch metadata, download, cache, and sync models.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from huggingface_hub import ModelInfo

from jessicai.config import get_settings
from jessicai.huggingface.client import HuggingFaceClient

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "jessicai_manifest.json"


class ModelManager:
    """High-level model lifecycle manager.

    Handles downloading, caching, and keeping models up-to-date.
    """

    def __init__(self, client: HuggingFaceClient | None = None) -> None:
        self._client = client or HuggingFaceClient()
        settings = get_settings()
        self._cache_dir = settings.model_cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._cache_dir / _MANIFEST_FILENAME
        self._manifest: dict[str, Any] = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest_path.exists():
            with self._manifest_path.open() as fh:
                return json.load(fh)
        return {}

    def _save_manifest(self) -> None:
        with self._manifest_path.open("w") as fh:
            json.dump(self._manifest, fh, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_cached(self) -> list[str]:
        """Return repo_ids of all locally cached models."""
        return list(self._manifest.keys())

    def is_cached(self, repo_id: str) -> bool:
        return repo_id in self._manifest

    def get_model_path(self, repo_id: str) -> Path | None:
        """Return the local path for *repo_id* if it is cached."""
        entry = self._manifest.get(repo_id)
        if entry:
            return Path(entry["path"])
        return None

    def fetch_model_info(self, repo_id: str) -> ModelInfo:
        """Retrieve Hub metadata for a model without downloading it."""
        info = self._client._api.model_info(repo_id)
        logger.debug("Fetched info for %s: sha=%s", repo_id, info.sha)
        return info

    def download(
        self,
        repo_id: str,
        force: bool = False,
        ignore_patterns: list[str] | None = None,
    ) -> Path:
        """Download *repo_id* and record it in the local manifest.

        Args:
            repo_id: Hugging Face model repository id (e.g. ``"gpt2"``).
            force: Re-download even if already cached.
            ignore_patterns: Glob patterns for files to skip.

        Returns:
            Local directory containing the downloaded model.
        """
        if self.is_cached(repo_id) and not force:
            path = self.get_model_path(repo_id)
            logger.info("Model %s already cached at %s", repo_id, path)
            return path  # type: ignore[return-value]

        logger.info("Downloading model %s …", repo_id)
        local_dir = self._client.download_model(
            repo_id=repo_id,
            ignore_patterns=ignore_patterns,
        )
        self._manifest[repo_id] = {
            "path": local_dir,
            "downloaded_at": time.time(),
        }
        self._save_manifest()
        return Path(local_dir)

    def sync(self, repo_id: str) -> bool:
        """Check for upstream updates and re-download if needed.

        Returns:
            ``True`` if the model was updated, ``False`` if already current.
        """
        info = self.fetch_model_info(repo_id)
        entry = self._manifest.get(repo_id, {})
        cached_sha = entry.get("sha")
        if cached_sha == info.sha:
            logger.info("Model %s is up-to-date (sha=%s)", repo_id, info.sha)
            return False

        logger.info(
            "Model %s has new sha %s (cached: %s). Re-downloading …",
            repo_id,
            info.sha,
            cached_sha,
        )
        self.download(repo_id, force=True)
        self._manifest[repo_id]["sha"] = info.sha
        self._save_manifest()
        return True

    def remove(self, repo_id: str) -> None:
        """Remove a model from the local manifest (does not delete files)."""
        if repo_id in self._manifest:
            del self._manifest[repo_id]
            self._save_manifest()
            logger.info("Removed %s from manifest", repo_id)
