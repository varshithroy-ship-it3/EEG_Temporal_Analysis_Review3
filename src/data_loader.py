"""
EDF Data Loader for CHB-MIT Scalp EEG Database.
Loads and manages EDF files using MNE-Python, extracting recording metadata,
validating channel consistency, and managing memory safely.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import mne
import numpy as np


class EDFDataLoader:
    """
    Handles robust loading of EDF EEG recordings with MNE-Python.
    """

    def __init__(self, data_dir: str = "data/chbmit/chb01"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def get_recording_info(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts metadata from an EDF file header without loading full signal into RAM.

        Returns:
            Dictionary with:
            - recording_id
            - file_path
            - sfreq (Hz)
            - n_samples
            - duration_seconds
            - n_channels
            - channel_names
            - start_datetime
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.data_dir / path.name

        if not path.exists():
            raise FileNotFoundError(f"EDF file does not exist: {path}")

        try:
            raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
            sfreq = float(raw.info["sfreq"])
            n_samples = int(raw.n_times)
            duration_sec = float(n_samples / sfreq)
            meas_date = raw.info.get("meas_date")

            info = {
                "recording_id": path.stem,
                "file_path": str(path),
                "file_name": path.name,
                "sfreq": sfreq,
                "n_samples": n_samples,
                "duration_seconds": duration_sec,
                "n_channels": len(raw.ch_names),
                "channel_names": list(raw.ch_names),
                "start_datetime": str(meas_date) if meas_date is not None else None,
            }
            raw.close()
            return info
        except Exception as e:
            raise RuntimeError(f"Error loading EDF header from {path}: {str(e)}")

    def load_raw(
        self,
        file_path: str,
        preload: bool = True,
        channels: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> mne.io.Raw:
        """
        Loads an MNE Raw object from EDF file.

        Args:
            file_path: Path or filename of the EDF.
            preload: Whether to read raw signals into memory.
            channels: Specific channels to pick (loads only these if specified).
            verbose: Verbosity level for MNE.

        Returns:
            mne.io.Raw instance
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.data_dir / path.name

        if not path.exists():
            raise FileNotFoundError(f"EDF file not found: {path}")

        try:
            raw = mne.io.read_raw_edf(str(path), preload=preload, verbose=verbose)
            if channels:
                available = set(raw.ch_names)
                missing = [c for c in channels if c not in available]
                if missing:
                    raise ValueError(
                        f"Requested channels {missing} missing in recording {path.name}."
                    )
                raw.pick_channels(channels)
            return raw
        except Exception as e:
            raise RuntimeError(f"Failed to read raw EDF at {path}: {str(e)}")

    def find_common_channels(self, file_paths: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Identifies channels present across all specified recordings.

        Returns:
            Tuple of (common_channels, dict_of_missing_per_file)
        """
        all_channels: Dict[str, List[str]] = {}
        for p in file_paths:
            info = self.get_recording_info(p)
            all_channels[info["recording_id"]] = info["channel_names"]

        if not all_channels:
            return [], {}

        # Compute intersection preserving order of first recording
        first_key = list(all_channels.keys())[0]
        first_channels = all_channels[first_key]
        common = [
            ch for ch in first_channels
            if all(ch in all_channels[k] for k in all_channels)
        ]

        # Missing report
        missing_report = {}
        for k, chs in all_channels.items():
            missing = [c for c in first_channels if c not in chs]
            if missing:
                missing_report[k] = missing

        return common, missing_report
