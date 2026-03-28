"""
Tests for voice authentication.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from huggingface_fork.auth.voice_auth import VoiceAuthenticator
from huggingface_fork.config import AuthConfig


def _make_config(threshold: float = 0.85, tmp_dir: Path = None) -> AuthConfig:
    cfg = AuthConfig()
    cfg.authorized_user = "NaTo1000"
    cfg.voice_threshold = threshold
    if tmp_dir:
        cfg.voice_samples_dir = tmp_dir
    return cfg


class TestVoiceAuthenticator:
    def setup_method(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.cfg = _make_config(tmp_dir=self.tmp_path)
        self.va = VoiceAuthenticator(self.cfg)

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_not_enrolled_initially(self):
        assert self.va.is_enrolled() is False

    def test_enroll_with_audio_files(self):
        # Create a simple synthetic WAV file
        from scipy.io import wavfile

        sample_rate = 16_000
        duration = 2
        t = np.linspace(0, duration, sample_rate * duration)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        wav_files = []
        for i in range(3):
            p = self.tmp_path / f"sample_{i}.wav"
            wavfile.write(str(p), sample_rate, audio)
            wav_files.append(p)

        self.va.enroll(audio_files=wav_files)
        assert self.va.is_enrolled() is True

    def test_cosine_similarity_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert VoiceAuthenticator._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert VoiceAuthenticator._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert VoiceAuthenticator._cosine_similarity(a, b) == 0.0

    def test_verify_fails_when_not_enrolled(self):
        result = self.va.verify()
        assert result is False

    def test_verify_passes_matching_voiceprint(self):
        from scipy.io import wavfile

        sample_rate = 16_000
        duration = 2
        t = np.linspace(0, duration, sample_rate * duration)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        wav_files = []
        for i in range(3):
            p = self.tmp_path / f"sample_{i}.wav"
            wavfile.write(str(p), sample_rate, audio)
            wav_files.append(p)

        self.va.enroll(audio_files=wav_files)

        # Use the same audio for verification — should exceed threshold.
        verify_wav = self.tmp_path / "verify.wav"
        wavfile.write(str(verify_wav), sample_rate, audio)
        assert self.va.verify(audio_file=verify_wav) is True

    def test_verify_fails_different_audio(self):
        from scipy.io import wavfile

        sample_rate = 16_000
        duration = 2
        t = np.linspace(0, duration, sample_rate * duration)
        audio_enroll = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        audio_verify = (np.sin(2 * np.pi * 880 * t) * 32767 + np.random.normal(0, 5000, len(t)).astype(np.int16)).astype(np.int16)

        wav_files = []
        for i in range(3):
            p = self.tmp_path / f"enroll_{i}.wav"
            wavfile.write(str(p), sample_rate, audio_enroll)
            wav_files.append(p)

        cfg = _make_config(threshold=0.99, tmp_dir=self.tmp_path)
        va = VoiceAuthenticator(cfg)
        va.enroll(audio_files=wav_files)

        verify_wav = self.tmp_path / "different.wav"
        wavfile.write(str(verify_wav), sample_rate, audio_verify)
        # With a very high threshold and different audio, should fail.
        assert va.verify(audio_file=verify_wav) is False

    def test_resample_changes_length(self):
        data = np.ones(16000, dtype=np.int16)
        resampled = VoiceAuthenticator._resample(data, 16000, 8000)
        assert len(resampled) == 8000
