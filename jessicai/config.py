"""
Configuration management.

Loads settings from environment variables (optionally from a .env file)
and exposes them as a typed Pydantic Settings object.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from the project root when this module is first imported
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Hugging Face
    # ------------------------------------------------------------------
    huggingface_api_key: SecretStr = Field(
        ...,
        description="Hugging Face access token (HUGGINGFACE_API_KEY)",
    )
    huggingface_namespace: str = Field(
        default="",
        description="HF user/org namespace for uploads and fine-tunes",
    )
    huggingface_endpoint: str = Field(
        default="https://huggingface.co",
        description="Hugging Face Hub endpoint URL",
    )

    # ------------------------------------------------------------------
    # Model cache
    # ------------------------------------------------------------------
    model_cache_dir: Path = Field(
        default=Path(".cache/models"),
        description="Local directory for cached models",
    )

    # ------------------------------------------------------------------
    # Device hosts
    # ------------------------------------------------------------------
    flipper_zero_host: str = Field(default="", description="Flipper Zero host/IP")
    pineapple_pager_host: str = Field(
        default="", description="Wi-Fi Pineapple Pager host/IP"
    )
    raspberry_pi_host: str = Field(
        default="", description="Raspberry Pi 5 host/IP"
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Log level")
    log_file: Path = Field(
        default=Path("logs/jessicai.log"), description="Log file path"
    )

    @field_validator("model_cache_dir", "log_file", mode="before")
    @classmethod
    def _make_path(cls, v: str | Path) -> Path:
        return Path(v)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_level(cls, v: str) -> str:
        return v.upper()

    def hf_token(self) -> str:
        """Return the raw Hugging Face API token string."""
        return self.huggingface_api_key.get_secret_value()


def get_settings() -> Settings:
    """Return a cached Settings instance.

    Raises ``ValueError`` if required environment variables are missing.
    """
    return _settings_cache()


_cached: Settings | None = None


def _settings_cache() -> Settings:
    global _cached
    if _cached is None:
        _cached = Settings()  # type: ignore[call-arg]
    return _cached


def reset_settings_cache() -> None:
    """Clear the cached settings (useful in tests)."""
    global _cached
    _cached = None
