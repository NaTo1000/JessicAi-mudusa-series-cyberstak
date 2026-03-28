"""
Voice-based authentication for the JessicAi HuggingFace Fork.

Overview
--------
Voice authentication is a *second factor* that must pass **in addition** to
the GPG signature check before full capabilities are unlocked.

How it works
~~~~~~~~~~~~
1. During **enrollment** the user records several short speech samples.
   :meth:`VoiceAuthenticator.enroll` computes a speaker embedding (an
   averaged MFCC vector) from those samples and saves it as a ``.npy``
   file in the configured ``voice_samples_dir``.

2. During **verification** a fresh recording (or a supplied audio file) is
   compared against the stored embedding using cosine similarity.  If the
   similarity score exceeds ``voice_threshold`` the check passes.

Dependencies
~~~~~~~~~~~~
The voice module uses ``sounddevice`` for live microphone capture,
``scipy`` for WAV I/O, ``numpy`` for numerical processing, and
``python_speech_features`` for MFCC extraction.  These are all listed in
``requirements.txt``.  If they are not installed the module degrades
gracefully and logs an informative error.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from huggingface_fork.config import AuthConfig, load_config

logger = logging.getLogger(__name__)

# Default recording duration in seconds for live capture.
_DEFAULT_RECORD_SECONDS = 4
_SAMPLE_RATE = 16_000  # 16 kHz, sufficient for MFCC-based speaker ID


class VoiceAuthError(Exception):
    """Raised when voice authentication encounters an unrecoverable error."""


class VoiceAuthenticator:
    """Speaker-verification-based second-factor authentication.

    Parameters
    ----------
    config:
        Authentication configuration.  If *None*, the global config is loaded
        from the environment.
    """

    def __init__(self, config: Optional[AuthConfig] = None) -> None:
        self._cfg = config or load_config().auth
        self._voiceprint_path = (
            self._cfg.voice_samples_dir
            / f"{self._cfg.authorized_user}_voiceprint.npy"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_enrolled(self) -> bool:
        """Return ``True`` if a voice-print has been enrolled for the user."""
        return self._voiceprint_path.exists()

    def enroll(self, audio_files: Optional[list[Path]] = None) -> None:
        """Enroll the authorised user's voice-print.

        Parameters
        ----------
        audio_files:
            List of WAV file paths to use for enrollment.  If *None*, the
            system records ``3`` samples from the default microphone.
        """
        embeddings: list[np.ndarray] = []
        if audio_files:
            for f in audio_files:
                emb = self._embedding_from_file(f)
                if emb is not None:
                    embeddings.append(emb)
        else:
            for i in range(3):
                logger.info(
                    "Enrollment sample %d/3 — please speak for %d seconds…",
                    i + 1,
                    _DEFAULT_RECORD_SECONDS,
                )
                audio = self._record_audio(_DEFAULT_RECORD_SECONDS)
                emb = self._compute_embedding(audio)
                if emb is not None:
                    embeddings.append(emb)
                time.sleep(0.5)

        if not embeddings:
            raise VoiceAuthError("No valid audio samples for enrollment.")

        mean_embedding = np.mean(np.stack(embeddings), axis=0)
        np.save(str(self._voiceprint_path), mean_embedding)
        logger.info("Voice-print enrolled and saved to %s.", self._voiceprint_path)

    def verify(self, audio_file: Optional[Path] = None) -> bool:
        """Verify a voice sample against the stored voice-print.

        Parameters
        ----------
        audio_file:
            Path to a WAV file.  If *None*, audio is captured from the
            default microphone.

        Returns
        -------
        bool
            ``True`` if the speaker matches the enrolled voice-print above
            the configured threshold.
        """
        if not self.is_enrolled():
            logger.error(
                "No voice-print found for user '%s'. Run enroll() first.",
                self._cfg.authorized_user,
            )
            return False

        stored = np.load(str(self._voiceprint_path))

        if audio_file:
            emb = self._embedding_from_file(audio_file)
        else:
            logger.info(
                "Voice verification — please speak for %d seconds…",
                _DEFAULT_RECORD_SECONDS,
            )
            audio = self._record_audio(_DEFAULT_RECORD_SECONDS)
            emb = self._compute_embedding(audio)

        if emb is None:
            logger.warning("Could not compute voice embedding from the provided audio.")
            return False

        score = self._cosine_similarity(stored, emb)
        logger.info("Voice match score: %.4f (threshold: %.4f)", score, self._cfg.voice_threshold)

        passed = score >= self._cfg.voice_threshold
        if passed:
            logger.info("Voice authentication PASSED.")
        else:
            logger.warning("Voice authentication FAILED (score below threshold).")
        return passed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_audio(self, duration: int) -> np.ndarray:
        """Record audio from the default microphone."""
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise VoiceAuthError(
                "sounddevice is not installed. "
                "Install it with: pip install sounddevice"
            ) from exc

        audio = sd.rec(
            int(duration * _SAMPLE_RATE),
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio.flatten()

    def _embedding_from_file(self, path: Path) -> Optional[np.ndarray]:
        """Load a WAV file and compute its speaker embedding."""
        try:
            from scipy.io import wavfile  # type: ignore[import-untyped]
        except ImportError as exc:
            raise VoiceAuthError(
                "scipy is not installed. Install it with: pip install scipy"
            ) from exc

        rate, data = wavfile.read(str(path))
        if data.ndim > 1:
            data = data[:, 0]
        # Resample if necessary.
        if rate != _SAMPLE_RATE:
            data = self._resample(data, rate, _SAMPLE_RATE)
        audio = data.astype(np.float32) / (np.iinfo(np.int16).max + 1)
        return self._compute_embedding(audio)

    @staticmethod
    def _resample(data: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        ratio = target_rate / orig_rate
        new_length = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, new_length)
        return np.interp(indices, np.arange(len(data)), data).astype(data.dtype)

    @staticmethod
    def _compute_embedding(audio: np.ndarray) -> Optional[np.ndarray]:
        """Compute an MFCC-based speaker embedding."""
        try:
            from python_speech_features import mfcc  # type: ignore[import-untyped]
        except ImportError:
            # Fallback: use raw signal statistics as a minimal embedding.
            logger.warning(
                "python_speech_features not installed; using minimal embedding."
            )
            return np.array([audio.mean(), audio.std(), np.percentile(audio, 25),
                             np.percentile(audio, 75)])

        if len(audio) < _SAMPLE_RATE // 4:
            return None

        features = mfcc(audio, samplerate=_SAMPLE_RATE, numcep=20, nfilt=40)
        # Use mean + std of each cepstral coefficient as the embedding.
        return np.concatenate([features.mean(axis=0), features.std(axis=0)])

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
