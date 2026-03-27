"""
Low-level Hugging Face Hub API client.

Wraps ``huggingface_hub.HfApi`` and exposes authenticated helpers used
by the rest of the JessicAI stack.
"""

from __future__ import annotations

import logging
from typing import Iterator

from huggingface_hub import HfApi, ModelInfo, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from jessicai.config import get_settings

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Authenticated wrapper around ``huggingface_hub.HfApi``."""

    def __init__(self, token: str | None = None) -> None:
        settings = get_settings()
        self._token = token or settings.hf_token()
        self._api = HfApi(token=self._token, endpoint=settings.huggingface_endpoint)
        logger.info("HuggingFaceClient initialised (endpoint=%s)", settings.huggingface_endpoint)

    # ------------------------------------------------------------------
    # Identity helpers
    # ------------------------------------------------------------------

    def whoami(self) -> dict:
        """Return information about the authenticated user/org."""
        return self._api.whoami()

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def search_models(
        self,
        query: str = "",
        task: str | None = None,
        limit: int = 20,
    ) -> list[ModelInfo]:
        """Search the Hugging Face Hub for models.

        Args:
            query: Free-text search query.
            task: Filter by pipeline tag (e.g. ``"text-generation"``).
            limit: Maximum number of results to return.

        Returns:
            List of :class:`~huggingface_hub.ModelInfo` objects.
        """
        results: list[ModelInfo] = list(
            self._api.list_models(
                search=query or None,
                filter=task,
                limit=limit,
                sort="downloads",
                direction=-1,
            )
        )
        logger.debug("search_models query=%r task=%r -> %d results", query, task, len(results))
        return results

    # ------------------------------------------------------------------
    # File / snapshot downloads
    # ------------------------------------------------------------------

    def download_file(
        self,
        repo_id: str,
        filename: str,
        cache_dir: str | None = None,
    ) -> str:
        """Download a single file from a Hub repository.

        Returns:
            Local path to the cached file.
        """
        settings = get_settings()
        cache = cache_dir or str(settings.model_cache_dir)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=self._token,
            cache_dir=cache,
        )
        logger.info("Downloaded %s/%s -> %s", repo_id, filename, path)
        return path

    def download_model(
        self,
        repo_id: str,
        cache_dir: str | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> str:
        """Download the full model snapshot.

        Returns:
            Local directory containing the model files.
        """
        settings = get_settings()
        cache = cache_dir or str(settings.model_cache_dir)
        path = snapshot_download(
            repo_id=repo_id,
            token=self._token,
            cache_dir=cache,
            ignore_patterns=ignore_patterns or ["*.msgpack", "flax_model*", "tf_model*"],
        )
        logger.info("Snapshot downloaded %s -> %s", repo_id, path)
        return path

    # ------------------------------------------------------------------
    # Upload helpers
    # ------------------------------------------------------------------

    def upload_model(
        self,
        local_dir: str,
        repo_id: str,
        commit_message: str = "Upload model via JessicAI",
        private: bool = True,
    ) -> str:
        """Upload a local model directory to the Hub.

        Returns:
            URL of the repository on the Hub.
        """
        try:
            self._api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        except HfHubHTTPError as exc:
            logger.warning("create_repo: %s", exc)

        url = self._api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info("Uploaded model to %s", url)
        return str(url)
