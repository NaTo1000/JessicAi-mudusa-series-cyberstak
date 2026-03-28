"""
tests/test_noise_filter.py
==========================
Unit tests for quantum_soul.noise_filter.
"""

import pytest
import numpy as np

from quantum_soul.noise_filter import QuantumNoiseFilter


class TestQuantumNoiseFilter:
    def test_default_construction(self):
        f = QuantumNoiseFilter()
        assert f.parity_threshold == 0.01
        assert f.smoothing_sigma == 1.5
        assert f.amplitude_threshold == 0.005

    def test_invalid_parity_threshold(self):
        with pytest.raises(ValueError):
            QuantumNoiseFilter(parity_threshold=-0.1)
        with pytest.raises(ValueError):
            QuantumNoiseFilter(parity_threshold=1.5)

    def test_invalid_smoothing_sigma(self):
        with pytest.raises(ValueError):
            QuantumNoiseFilter(smoothing_sigma=-1)

    def test_invalid_amplitude_threshold(self):
        with pytest.raises(ValueError):
            QuantumNoiseFilter(amplitude_threshold=2.0)

    # filter_counts -----------------------------------------------------------

    def test_filter_counts_removes_low_freq(self):
        counts = {"00": 100, "01": 2, "10": 98, "11": 1}
        f = QuantumNoiseFilter(parity_threshold=0.01)
        filtered = f.filter_counts(counts, shots=201)
        assert "01" not in filtered
        assert "11" not in filtered
        assert "00" in filtered
        assert "10" in filtered

    def test_filter_counts_keeps_all_above_threshold(self):
        counts = {"00": 50, "01": 50, "10": 50, "11": 50}
        f = QuantumNoiseFilter(parity_threshold=0.01)
        filtered = f.filter_counts(counts, shots=200)
        assert len(filtered) == 4

    # smooth_probabilities ----------------------------------------------------

    def test_smooth_output_sums_to_one(self):
        probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        f = QuantumNoiseFilter(smoothing_sigma=1.0)
        smoothed = f.smooth_probabilities(probs)
        assert abs(smoothed.sum() - 1.0) < 1e-6

    def test_smooth_all_nonneg(self):
        probs = np.array([0.0, 0.5, 0.3, 0.2])
        f = QuantumNoiseFilter(smoothing_sigma=0.5)
        smoothed = f.smooth_probabilities(probs)
        assert np.all(smoothed >= 0)

    def test_smooth_zero_sigma_is_passthrough(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        f = QuantumNoiseFilter(smoothing_sigma=0)
        smoothed = f.smooth_probabilities(probs)
        np.testing.assert_allclose(smoothed, probs)

    def test_smooth_zero_array_returns_uniform(self):
        probs = np.zeros(4)
        f = QuantumNoiseFilter(smoothing_sigma=1.0)
        smoothed = f.smooth_probabilities(probs)
        np.testing.assert_allclose(smoothed, np.full(4, 0.25))

    # extract_dominant_states -------------------------------------------------

    def test_extract_dominant_removes_sub_threshold(self):
        probs = np.array([0.001, 0.490, 0.505, 0.004])
        f = QuantumNoiseFilter(amplitude_threshold=0.005)
        extracted = f.extract_dominant_states(probs)
        assert extracted[0] == 0.0
        assert extracted[3] == 0.0
        assert abs(extracted.sum() - 1.0) < 1e-6

    def test_extract_dominant_fallback_to_max(self):
        probs = np.array([0.001, 0.002, 0.003])
        f = QuantumNoiseFilter(amplitude_threshold=0.5)
        extracted = f.extract_dominant_states(probs)
        # Should keep only the maximum (index 2)
        assert extracted[2] == 1.0
        assert extracted[0] == 0.0
        assert extracted[1] == 0.0

    # apply_all ---------------------------------------------------------------

    def test_apply_all_sums_to_one(self):
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(16))
        f = QuantumNoiseFilter()
        cleaned = f.apply_all(probs)
        assert abs(cleaned.sum() - 1.0) < 1e-6

    def test_apply_all_nonneg(self):
        rng = np.random.default_rng(1)
        probs = rng.dirichlet(np.ones(16))
        f = QuantumNoiseFilter()
        cleaned = f.apply_all(probs)
        assert np.all(cleaned >= 0)

    # noise_level & SNR -------------------------------------------------------

    def test_noise_level_pure_signal(self):
        # All mass is above threshold
        probs = np.array([0.5, 0.5])
        f = QuantumNoiseFilter(amplitude_threshold=0.1)
        assert f.noise_level(probs) == 0.0

    def test_noise_level_pure_noise(self):
        probs = np.array([0.002, 0.002, 0.003])
        f = QuantumNoiseFilter(amplitude_threshold=0.5)
        assert f.noise_level(probs) == 1.0

    def test_snr_high_signal(self):
        probs = np.array([0.5, 0.5])
        f = QuantumNoiseFilter(amplitude_threshold=0.1)
        assert f.signal_to_noise_ratio(probs) > 1e6

    def test_snr_zero_probs(self):
        probs = np.zeros(4)
        f = QuantumNoiseFilter()
        snr = f.signal_to_noise_ratio(probs)
        assert snr >= 0.0

    # report ------------------------------------------------------------------

    def test_report_keys(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        f = QuantumNoiseFilter()
        r = f.report(probs)
        assert "noise_level" in r
        assert "snr" in r
        assert "n_dominant_states" in r
        assert "max_probability" in r

    def test_report_max_probability(self):
        probs = np.array([0.1, 0.7, 0.2])
        f = QuantumNoiseFilter()
        r = f.report(probs)
        assert abs(r["max_probability"] - 0.7) < 1e-9
