"""
Tests for config loading.
"""

from __future__ import annotations

import os

import pytest

from huggingface_fork.config import AuthConfig, HuggingFaceConfig, SecurityConfig, load_config


class TestAuthConfig:
    def test_default_authorized_user(self):
        cfg = AuthConfig()
        assert cfg.authorized_user == "NaTo1000"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("JESSICAI_AUTHORIZED_USER", "OtherUser")
        cfg = AuthConfig()
        assert cfg.authorized_user == "OtherUser"

    def test_voice_threshold_default(self):
        cfg = AuthConfig()
        assert cfg.voice_threshold == pytest.approx(0.85)

    def test_voice_threshold_env(self, monkeypatch):
        monkeypatch.setenv("JESSICAI_VOICE_THRESHOLD", "0.9")
        cfg = AuthConfig()
        assert cfg.voice_threshold == pytest.approx(0.9)


class TestHuggingFaceConfig:
    def test_allow_internet_default(self):
        cfg = HuggingFaceConfig()
        assert cfg.allow_internet is True

    def test_hub_token_env(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_testtoken")
        cfg = HuggingFaceConfig()
        assert cfg.hub_token == "hf_testtoken"

    def test_endpoint_default(self):
        cfg = HuggingFaceConfig()
        assert cfg.endpoint == "https://huggingface.co"

    def test_endpoint_env(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "https://custom.hf.internal")
        cfg = HuggingFaceConfig()
        assert cfg.endpoint == "https://custom.hf.internal"


class TestSecurityConfig:
    def test_max_failed_attempts_default(self):
        cfg = SecurityConfig()
        assert cfg.max_failed_attempts == 3

    def test_lockout_seconds_default(self):
        cfg = SecurityConfig()
        assert cfg.lockout_seconds == 300

    def test_verbose_logging_default(self):
        cfg = SecurityConfig()
        assert cfg.verbose_logging is False

    def test_verbose_logging_env(self, monkeypatch):
        monkeypatch.setenv("JESSICAI_VERBOSE_LOGGING", "true")
        cfg = SecurityConfig()
        assert cfg.verbose_logging is True


class TestLoadConfig:
    def test_returns_app_config(self):
        from huggingface_fork.config import AppConfig
        cfg = load_config()
        assert isinstance(cfg, AppConfig)
        assert hasattr(cfg, "auth")
        assert hasattr(cfg, "huggingface")
        assert hasattr(cfg, "security")
