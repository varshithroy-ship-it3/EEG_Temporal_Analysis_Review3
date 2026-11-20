"""
EEG Feature Extraction Module for CHB-MIT Scalp EEG Database.
Extracts time-domain and frequency-domain physiological features from EEG windows
using efficient vectorized scientific computation.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


class EEGFeatureExtractor:
    """
    Extracts time-domain and spectral features from multi-channel EEG windows.
    """

    def __init__(self, sfreq: float = 256.0):
        self.sfreq = sfreq

        # Standard physiological EEG frequency bands
        self.bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        }

    def extract_features_batch(
        self,
        signals: np.ndarray,
        windows_meta: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Extracts features across a batch of EEG windows.

        Args:
            signals: 3D numpy array [n_windows x n_channels x n_samples]
            windows_meta: List of metadata dicts corresponding to each window

        Returns:
            pd.DataFrame containing metadata columns (window_id, recording_id, start_seconds,
            end_seconds, label) and numeric feature columns.
        """
        n_windows, n_channels, n_samples = signals.shape
        if n_windows == 0:
            return pd.DataFrame()

        # ---------------------------------------------------------------------
        # 1. Time-Domain Features (Vectorized across windows & channels)
        # ---------------------------------------------------------------------
        # Mean across time samples [n_windows x n_channels]
        ch_means = np.mean(signals, axis=2)
        ch_stds = np.std(signals, axis=2)
        ch_vars = np.var(signals, axis=2)
        ch_rms = np.sqrt(np.mean(signals ** 2, axis=2))
        ch_ptp = np.ptp(signals, axis=2)  # Peak-to-peak amplitude
        ch_mins = np.min(signals, axis=2)
        ch_maxs = np.max(signals, axis=2)
        ch_energy = np.sum(signals ** 2, axis=2)

        # Aggregate across channels: global mean, max, std
        feat_mean = np.mean(ch_means, axis=1)
        feat_std = np.mean(ch_stds, axis=1)
        feat_var = np.mean(ch_vars, axis=1)
        feat_rms = np.mean(ch_rms, axis=1)
        feat_max_rms = np.max(ch_rms, axis=1)  # Detects focal channel amplitude surges
        feat_ptp = np.mean(ch_ptp, axis=1)
        feat_max_ptp = np.max(ch_ptp, axis=1)
        feat_min = np.min(ch_mins, axis=1)
        feat_max = np.max(ch_maxs, axis=1)
        feat_energy = np.mean(ch_energy, axis=1)

        # Higher-order statistical moments (flattened over channels per window for speed)
        reshaped = signals.reshape(n_windows, -1)
        feat_skew = skew(reshaped, axis=1, bias=False)
        feat_kurt = kurtosis(reshaped, axis=1, bias=False)

        # ---------------------------------------------------------------------
        # 2. Frequency-Domain Features (FFT Power Spectral Density)
        # ---------------------------------------------------------------------
        # rfft along time axis: output shape [n_windows x n_channels x n_freqs]
        fft_vals = np.abs(np.fft.rfft(signals, axis=2)) ** 2
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.sfreq)

        # Mean PSD across channels [n_windows x n_freqs]
        psd_mean_ch = np.mean(fft_vals, axis=1)
        total_power = np.sum(psd_mean_ch[:, (freqs >= 0.5) & (freqs <= 45.0)], axis=1) + 1e-12

        # Sub-band absolute and relative powers
        band_powers = {}
        for band_name, (fmin, fmax) in self.bands.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            bp = np.sum(psd_mean_ch[:, mask], axis=1)
            band_powers[f"power_{band_name}"] = bp
            band_powers[f"rel_power_{band_name}"] = bp / total_power

        # Spectral Entropy: H = - sum(p * log2(p))
        # Normalize PSD to probability distribution across band 0.5 - 45 Hz
        mask_all = (freqs >= 0.5) & (freqs <= 45.0)
        psd_norm = psd_mean_ch[:, mask_all] / total_power[:, None]
        psd_norm = np.clip(psd_norm, 1e-12, 1.0)
        feat_spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm), axis=1)

        # Dominant Frequency
        dominant_freq_idx = np.argmax(psd_mean_ch[:, mask_all], axis=1)
        band_freqs = freqs[mask_all]
        feat_dominant_freq = band_freqs[dominant_freq_idx]

        # Slow-to-Fast Rhythm Ratio: (Delta + Theta) / (Alpha + Beta + Gamma)
        slow_power = band_powers["power_delta"] + band_powers["power_theta"]
        fast_power = band_powers["power_alpha"] + band_powers["power_beta"] + band_powers["power_gamma"] + 1e-12
        feat_slow_fast_ratio = slow_power / fast_power

        # ---------------------------------------------------------------------
        # Assemble Feature DataFrame
        # ---------------------------------------------------------------------
        data_dict = {
            "window_id": [m["window_id"] for m in windows_meta],
            "recording_id": [m["recording_id"] for m in windows_meta],
            "start_seconds": [m["start_seconds"] for m in windows_meta],
            "end_seconds": [m["end_seconds"] for m in windows_meta],
            "overlap_seconds": [m.get("overlap_seconds", 0.0) for m in windows_meta],
            "overlap_ratio": [m.get("overlap_ratio", 0.0) for m in windows_meta],
            "label": [m["label"] for m in windows_meta],
            # Time-domain features
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "feat_var": feat_var,
            "feat_rms": feat_rms,
            "feat_max_rms": feat_max_rms,
            "feat_ptp": feat_ptp,
            "feat_max_ptp": feat_max_ptp,
            "feat_min": feat_min,
            "feat_max": feat_max,
            "feat_energy": feat_energy,
            "feat_skew": feat_skew,
            "feat_kurtosis": feat_kurt,
            # Frequency-domain features
            "feat_power_delta": band_powers["power_delta"],
            "feat_power_theta": band_powers["power_theta"],
            "feat_power_alpha": band_powers["power_alpha"],
            "feat_power_beta": band_powers["power_beta"],
            "feat_power_gamma": band_powers["power_gamma"],
            "feat_rel_power_delta": band_powers["rel_power_delta"],
            "feat_rel_power_theta": band_powers["rel_power_theta"],
            "feat_rel_power_alpha": band_powers["rel_power_alpha"],
            "feat_rel_power_beta": band_powers["rel_power_beta"],
            "feat_rel_power_gamma": band_powers["rel_power_gamma"],
            "feat_total_power": total_power,
            "feat_spectral_entropy": feat_spectral_entropy,
            "feat_dominant_freq": feat_dominant_freq,
            "feat_slow_fast_ratio": feat_slow_fast_ratio,
        }

        df = pd.DataFrame(data_dict)
        return df

    @staticmethod
    def get_feature_column_names(df: pd.DataFrame) -> List[str]:
        """
        Returns all numeric model input feature column names (excluding identifiers and labels).
        """
        exclude = {
            "window_id", "recording_id", "start_seconds", "end_seconds",
            "overlap_seconds", "overlap_ratio", "label"
        }
        return [c for c in df.columns if c not in exclude and c.startswith("feat_")]
