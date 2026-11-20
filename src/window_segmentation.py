"""
EEG Window Segmentation Module for CHB-MIT Scalp EEG Database.
Segments continuous EEG into overlapping temporal windows (e.g. 2s duration, 1s step),
computing exact intersection with expert-annotated seizure intervals.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


class EEGWindowSegmenter:
    """
    Partitions continuous multi-channel EEG signals into fixed-duration overlapping windows
    and assigns ground-truth labels based on clinical annotations.
    """

    def __init__(
        self,
        window_duration_sec: float = 2.0,
        step_size_sec: float = 1.0,
        seizure_overlap_threshold: float = 0.50,
    ):
        """
        Args:
            window_duration_sec: Duration of each temporal window in seconds (default: 2.0s).
            step_size_sec: Step / stride size between successive windows (default: 1.0s -> 50% overlap).
            seizure_overlap_threshold: Minimum fraction of window duration overlapping an
                                       annotated seizure to receive label=1 (default: 0.50 = 50%).
        """
        self.window_duration_sec = float(window_duration_sec)
        self.step_size_sec = float(step_size_sec)
        self.overlap_sec = self.window_duration_sec - self.step_size_sec
        self.seizure_overlap_threshold = float(seizure_overlap_threshold)

    @property
    def policy_description(self) -> str:
        return (
            f"Window: {self.window_duration_sec:.1f}s | Step: {self.step_size_sec:.1f}s | "
            f"Seizure label if overlap >= {self.seizure_overlap_threshold*100:.0f}% of window duration."
        )

    def segment_recording(
        self,
        raw_data: np.ndarray,
        sfreq: float,
        recording_id: str,
        channel_names: List[str],
        annotations_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Segments a multi-channel recording into windows.

        Args:
            raw_data: 2D numpy array [n_channels x n_samples]
            sfreq: Sampling rate in Hz (read from EDF header)
            recording_id: Unique recording identifier (e.g. 'chb01_03')
            channel_names: Ordered list of channel names
            annotations_df: DataFrame of expert annotations for this recording

        Returns:
            Tuple of:
            - List of window metadata dictionaries
            - 3D numpy array [n_windows x n_channels x samples_per_window]
        """
        n_channels, total_samples = raw_data.shape
        window_samples = int(round(self.window_duration_sec * sfreq))
        step_samples = int(round(self.step_size_sec * sfreq))

        if total_samples < window_samples:
            return [], np.empty((0, n_channels, window_samples))

        # Filter annotations for this recording
        rec_annots = []
        if annotations_df is not None and not annotations_df.empty:
            match = annotations_df[annotations_df["recording_id"] == recording_id]
            for _, row in match.iterrows():
                rec_annots.append((float(row["start_seconds"]), float(row["end_seconds"])))

        windows_meta: List[Dict[str, Any]] = []
        window_signals: List[np.ndarray] = []

        start_sample = 0
        idx = 0

        while start_sample + window_samples <= total_samples:
            end_sample = start_sample + window_samples
            start_sec = start_sample / sfreq
            end_sec = end_sample / sfreq

            # Calculate overlap with any annotated seizure in this recording
            overlap_sec = 0.0
            for sz_start, sz_end in rec_annots:
                intersection = max(0.0, min(end_sec, sz_end) - max(start_sec, sz_start))
                overlap_sec += intersection

            overlap_ratio = overlap_sec / self.window_duration_sec
            label = 1 if overlap_ratio >= self.seizure_overlap_threshold else 0

            window_id = f"WIN_{recording_id}_{idx:05d}"
            windows_meta.append({
                "window_id": window_id,
                "recording_id": recording_id,
                "start_seconds": round(start_sec, 4),
                "end_seconds": round(end_sec, 4),
                "duration_seconds": self.window_duration_sec,
                "overlap_seconds": round(overlap_sec, 4),
                "overlap_ratio": round(overlap_ratio, 4),
                "label": int(label),
                "n_channels": n_channels,
            })

            window_signals.append(raw_data[:, start_sample:end_sample])

            idx += 1
            start_sample += step_samples

        signals_array = np.array(window_signals, dtype=np.float32)
        return windows_meta, signals_array
