"""
Configuration for the JessicAi HuggingFace Fork.

Values can be overridden via environment variables or a `.env` file at the
project root.  All sensitive values (tokens, key fingerprints) must be
supplied through the environment — never hard-coded in source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
GPG_KEYRING_DIR = PROJECT_ROOT / ".gnupg"
VOICE_SAMPLES_DIR = PROJECT_ROOT / "voice_samples"

for _d in (LOGS_DIR, GPG_KEYRING_DIR, VOICE_SAMPLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataclass-based configuration
# ---------------------------------------------------------------------------
@dataclass
class AuthConfig:
    """Authentication configuration."""

    # The HuggingFace username that is granted full (unlocked) access.
    authorized_user: str = field(
        default_factory=lambda: os.environ.get("JESSICAI_AUTHORIZED_USER", "NaTo1000")
    )

    # GPG fingerprint (40-char hex) of the key that must sign auth tokens.
    # Set via the environment variable JESSICAI_GPG_FINGERPRINT.
    gpg_fingerprint: str = field(
        default_factory=lambda: os.environ.get("JESSICAI_GPG_FINGERPRINT", "")
    )

    # Path to the GPG homedir (defaults to project-local .gnupg directory).
    gpg_homedir: Path = field(default_factory=lambda: GPG_KEYRING_DIR)

    # Minimum voice-match confidence score (0.0 – 1.0) to pass voice auth.
    voice_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("JESSICAI_VOICE_THRESHOLD", "0.85")
        )
    )

    # Directory containing enrolled voice-print files (.npy).
    voice_samples_dir: Path = field(default_factory=lambda: VOICE_SAMPLES_DIR)


@dataclass
class HuggingFaceConfig:
    """HuggingFace API configuration."""

    # HuggingFace Hub API token (read-only public token is fine for public
    # models; supply a write token via the environment for private resources).
    hub_token: str = field(
        default_factory=lambda: os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
    )

    # Local cache directory for downloaded models / datasets.
    cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "HF_HOME",
                str(PROJECT_ROOT / ".hf_cache"),
            )
        )
    )

    # Whether to allow internet fetches (overridden to False when auth fails).
    allow_internet: bool = True

    # HuggingFace Hub endpoint (default: public hub).
    endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "HF_ENDPOINT", "https://huggingface.co"
        )
    )


@dataclass
class SecurityConfig:
    """Security and monitoring configuration."""

    # Directory for audit / access logs.
    logs_dir: Path = field(default_factory=lambda: LOGS_DIR)

    # Maximum number of failed auth attempts before temporary lockout.
    max_failed_attempts: int = field(
        default_factory=lambda: int(
            os.environ.get("JESSICAI_MAX_FAILED_ATTEMPTS", "3")
        )
    )

    # Lockout duration in seconds after max_failed_attempts is reached.
    lockout_seconds: int = field(
        default_factory=lambda: int(
            os.environ.get("JESSICAI_LOCKOUT_SECONDS", "300")
        )
    )

    # Enable verbose security event logging.
    verbose_logging: bool = field(
        default_factory=lambda: os.environ.get(
            "JESSICAI_VERBOSE_LOGGING", "false"
        ).lower()
        == "true"
    )


@dataclass
class AppConfig:
    """Root application configuration."""

    auth: AuthConfig = field(default_factory=AuthConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


def load_config() -> AppConfig:
    """Load application configuration from environment / defaults."""
    return AppConfig()
