"""
EEG Preprocessing Pipeline for CHB-MIT Database.
Applies bandpass filtering (0.5-45.0 Hz), notch filtering for powerline noise (60 Hz),
signal quality checks, and NaN/Inf cleaning.
"""

from typing import List, Optional, Tuple, Dict, Any
import mne
import numpy as np


class EEGPreprocessor:
    """
    Standardizes and filters continuous multi-channel EEG signals.
    """

    def __init__(
        self,
        l_freq: float = 0.5,
        h_freq: float = 45.0,
        notch_freq: Optional[float] = 60.0,
        filter_method: str = "fir",
    ):
        """
        Args:
            l_freq: Lower cutoff frequency (Hz) to eliminate DC offset and slow baseline wander.
            h_freq: Upper cutoff frequency (Hz) to suppress muscle EMG and high-frequency noise.
            notch_freq: Powerline notch filter frequency in Hz (60 Hz for USA / CHB-MIT).
            filter_method: Filtering method for MNE ('fir' or 'iir').
        """
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.filter_method = filter_method

    def preprocess_raw(
        self,
        raw: mne.io.Raw,
        channels: Optional[List[str]] = None,
        copy: bool = True,
        verbose: bool = False,
    ) -> mne.io.Raw:
        """
        Preprocesses an MNE Raw EEG object:
        1. Selects specified EEG channels.
        2. Validates finite values (NaN / Inf sanitization).
        3. Applies notch filter at 60 Hz (if applicable and within Nyquist limit).
        4. Applies bandpass filter (0.5 - 45.0 Hz).

        Returns:
            Preprocessed mne.io.Raw object.
        """
        raw_proc = raw.copy() if copy else raw

        # Pick channels if specified
        if channels is not None:
            available = set(raw_proc.ch_names)
            selected = [c for c in channels if c in available]
            if len(selected) < len(channels):
                missing = [c for c in channels if c not in available]
                raise ValueError(f"Channels missing in EEG: {missing}")
            raw_proc.pick_channels(selected)

        # Check for NaN / Inf in signal data
        data = raw_proc.get_data()
        if not np.all(np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            raw_proc._data = data

        sfreq = raw_proc.info["sfreq"]
        nyquist = sfreq / 2.0

        # Notch filter for 60 Hz line noise if within Nyquist
        if self.notch_freq and self.notch_freq < nyquist:
            try:
                raw_proc.notch_filter(
                    freqs=self.notch_freq,
                    method=self.filter_method,
                    verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"Warning: Notch filter failed ({e}), continuing with bandpass.")

        # Bandpass filter (0.5 - 45.0 Hz)
        h_cutoff = min(self.h_freq, nyquist - 1.0)
        raw_proc.filter(
            l_freq=self.l_freq,
            h_freq=h_cutoff,
            method=self.filter_method,
            verbose=verbose,
        )

        return raw_proc

    def check_signal_quality(self, data: np.ndarray, sfreq: float) -> Dict[str, Any]:
        """
        Performs signal quality audit on a 2D numpy array [channels x samples].

        Returns:
            Dictionary containing metrics:
            - is_valid: bool
            - flatline_channels: list of indices with zero variance
            - saturated_samples_ratio: fraction of samples > 500 uV (typical scalp EEG limit)
            - rms_amplitude: mean RMS across channels
        """
        std_per_ch = np.std(data, axis=1)
        flatline_idx = np.where(std_per_ch < 1e-12)[0].tolist()

        # In CHB-MIT, units are Volts (e.g. 50 uV = 50e-6 V)
        # Saturated voltage threshold: 1000 uV = 1e-3 V
        saturated = np.mean(np.abs(data) > 1e-3)
        rms = float(np.sqrt(np.mean(data ** 2)))

        is_valid = len(flatline_idx) == 0 and saturated < 0.10

        return {
            "is_valid": is_valid,
            "flatline_channels": flatline_idx,
            "saturation_ratio": float(saturated),
            "rms_amplitude": rms,
        }
