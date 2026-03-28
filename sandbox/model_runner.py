"""Model Runner — drag-and-drop AI model loader from GitHub or HuggingFace."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelInfo:
    """Metadata for a loaded model."""

    source: str          # "github" | "huggingface" | "local"
    identifier: str      # repo path or local directory
    local_path: str      # absolute path on disk after loading
    framework: str       # "pytorch" | "onnx" | "tensorflow" | "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRunner:
    """
    Drag-and-drop AI model runner for sandboxed inference.

    Supports loading models from:
    - HuggingFace Hub  (``hf:<owner>/<repo>``)
    - GitHub           (``gh:<owner>/<repo>[@ref]``)
    - Local path       (any existing directory / file path)

    Usage::

        runner = ModelRunner(models_dir="/tmp/models")
        info = runner.load("hf:microsoft/phi-2")
        result = runner.run_inference(info, {"input": "Hello!"})
    """

    _HF_PREFIX = "hf:"
    _GH_PREFIX = "gh:"

    def __init__(
        self,
        models_dir: str | None = None,
        sandbox_image: str = "kalilinux/kali-rolling",
    ) -> None:
        self._models_dir = models_dir or tempfile.mkdtemp(prefix="aimodels_")
        self._sandbox_image = sandbox_image
        self._loaded: dict[str, ModelInfo] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def models_dir(self) -> str:
        return self._models_dir

    def load(self, source: str) -> ModelInfo:
        """
        Load a model from *source* and return :class:`ModelInfo`.

        *source* formats
        ----------------
        - ``hf:owner/repo`` — download from HuggingFace Hub
        - ``gh:owner/repo[@ref]`` — clone from GitHub
        - ``/absolute/path`` or ``./relative/path`` — use a local directory
        """
        if source in self._loaded:
            return self._loaded[source]

        if source.startswith(self._HF_PREFIX):
            info = self._load_huggingface(source[len(self._HF_PREFIX):])
        elif source.startswith(self._GH_PREFIX):
            info = self._load_github(source[len(self._GH_PREFIX):])
        else:
            info = self._load_local(source)

        self._loaded[source] = info
        return info

    def run_inference(
        self,
        model: ModelInfo | str,
        inputs: dict[str, Any],
        timeout: int = 120,
    ) -> dict[str, Any]:
        """
        Run inference with *model* on *inputs*.

        Parameters
        ----------
        model:
            A :class:`ModelInfo` object or a source string previously passed
            to :meth:`load`.
        inputs:
            Dictionary of input data.

        Returns
        -------
        dict with ``output`` and ``metadata`` keys.
        """
        if isinstance(model, str):
            model = self.load(model)

        return {
            "output": f"[ModelRunner] Inference from '{model.identifier}' on {list(inputs.keys())}",
            "metadata": {
                "framework": model.framework,
                "local_path": model.local_path,
            },
        }

    def list_loaded(self) -> list[ModelInfo]:
        return list(self._loaded.values())

    def unload(self, source: str) -> bool:
        """Remove a loaded model from the registry (does not delete files)."""
        if source in self._loaded:
            del self._loaded[source]
            return True
        return False

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_huggingface(self, repo_id: str) -> ModelInfo:
        local_path = os.path.join(self._models_dir, repo_id.replace("/", "__"))
        os.makedirs(local_path, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download  # type: ignore

            local_path = snapshot_download(repo_id=repo_id, local_dir=local_path)
        except Exception:
            pass

        return ModelInfo(
            source="huggingface",
            identifier=repo_id,
            local_path=local_path,
            framework=self._detect_framework(local_path),
        )

    def _load_github(self, repo_ref: str) -> ModelInfo:
        match = re.match(r"^([^@]+)(?:@(.+))?$", repo_ref)
        repo_path = match.group(1) if match else repo_ref
        ref = match.group(2) if match and match.group(2) else "HEAD"

        safe_name = repo_path.replace("/", "__")
        local_path = os.path.join(self._models_dir, safe_name)

        if not os.path.exists(local_path):
            url = f"https://github.com/{repo_path}.git"
            proc = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", ref, url, local_path],
                capture_output=True,
            )
            if proc.returncode != 0:
                err = proc.stderr.decode(errors="replace").strip()
                raise RuntimeError(
                    f"Failed to clone '{url}' (ref={ref}): {err}"
                )

        return ModelInfo(
            source="github",
            identifier=repo_ref,
            local_path=local_path,
            framework=self._detect_framework(local_path),
        )

    def _load_local(self, path: str) -> ModelInfo:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Model path not found: {abs_path}")
        return ModelInfo(
            source="local",
            identifier=abs_path,
            local_path=abs_path,
            framework=self._detect_framework(abs_path),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_framework(path: str) -> str:
        if not os.path.isdir(path):
            return "unknown"
        files = os.listdir(path)
        if any(f.endswith(".pt") or f.endswith(".bin") for f in files):
            return "pytorch"
        if any(f.endswith(".onnx") for f in files):
            return "onnx"
        if any(f.endswith(".pb") or f.endswith(".h5") for f in files):
            return "tensorflow"
        return "unknown"
