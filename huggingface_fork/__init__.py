"""
JessicAi HuggingFace Fork — Autonomous AI Integration
======================================================
Fully autonomous HuggingFace integration with GPG + voice authentication,
secure internet access, and comprehensive logging.

Only user 'NaTo1000' (validated by GPG key and voice token) gains full access.
All other callers receive a read-only, restricted view of the system.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jessicai-huggingface-fork")
except PackageNotFoundError:
    __version__ = "0.1.0"

from huggingface_fork.auth.gpg_auth import GPGAuthenticator
from huggingface_fork.auth.voice_auth import VoiceAuthenticator
from huggingface_fork.models.loader import ModelLoader
from huggingface_fork.datasets.loader import DatasetLoader
from huggingface_fork.security.monitor import SecurityMonitor

__all__ = [
    "GPGAuthenticator",
    "VoiceAuthenticator",
    "ModelLoader",
    "DatasetLoader",
    "SecurityMonitor",
]
