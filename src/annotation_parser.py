"""
Annotation Parser for CHB-MIT Scalp EEG Database.
Parses summary text files (e.g., chb01-summary.txt) to extract expert-annotated
seizure intervals and validate them against recording boundaries.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class AnnotationParser:
    """
    Parses and manages expert clinical seizure annotations from CHB-MIT summary files.
    """

    def __init__(self, summary_path: str):
        self.summary_path = Path(summary_path)
        if not self.summary_path.exists():
            raise FileNotFoundError(f"Summary file not found at: {self.summary_path}")

    def parse(self, target_recordings: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Parses the summary file and returns structured annotation intervals.

        Args:
            target_recordings: List of file names to filter by (e.g. ['chb01_03.edf', 'chb01_04.edf']).
                               If None, all files in the summary are parsed.

        Returns:
            pd.DataFrame with columns:
            ['recording_id', 'event_id', 'event_type', 'start_seconds', 'end_seconds', 'source']
        """
        with open(self.summary_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Regular expressions for file blocks
        # Pattern handles varying spaces and lines per recording
        file_blocks = re.split(r"File Name:\s*", content)
        records = []
        global_event_idx = 1

        for block in file_blocks[1:]:
            lines = [line.strip() for line in block.split("\n") if line.strip()]
            if not lines:
                continue

            file_name = lines[0]
            # Normalize recording_id (e.g., 'chb01_03')
            rec_id = Path(file_name).stem

            if target_recordings:
                # Support matching by full filename 'chb01_03.edf' or stem 'chb01_03'
                match_found = any(
                    file_name == t or rec_id == Path(t).stem for t in target_recordings
                )
                if not match_found:
                    continue

            # Parse number of seizures
            num_seizures = 0
            for line in lines:
                m_num = re.match(r"Number of Seizures in File:\s*(\d+)", line, re.IGNORECASE)
                if m_num:
                    num_seizures = int(m_num.group(1))
                    break

            if num_seizures > 0:
                # Find all seizure start/end pairs in this block
                starts = [
                    float(m.group(1))
                    for line in lines
                    for m in [re.search(r"Seizure\s*(?:\d+)?\s*Start Time:\s*(\d+)\s*seconds", line, re.IGNORECASE)]
                    if m
                ]
                ends = [
                    float(m.group(1))
                    for line in lines
                    for m in [re.search(r"Seizure\s*(?:\d+)?\s*End Time:\s*(\d+)\s*seconds", line, re.IGNORECASE)]
                    if m
                ]

                for sz_idx, (start, end) in enumerate(zip(starts, ends), start=1):
                    event_id = f"EXP_{rec_id}_SZ{sz_idx:02d}"
                    records.append({
                        "recording_id": rec_id,
                        "event_id": event_id,
                        "event_type": "expert_seizure",
                        "start_seconds": float(start),
                        "end_seconds": float(end),
                        "source": f"CHB-MIT:{self.summary_path.name}"
                    })
                    global_event_idx += 1

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                "recording_id", "event_id", "event_type", "start_seconds", "end_seconds", "source"
            ])
        else:
            df = df.sort_values(by=["recording_id", "start_seconds"]).reset_index(drop=True)

        return df

    def validate_durations(self, annotations_df: pd.DataFrame, recording_durations: Dict[str, float]) -> pd.DataFrame:
        """
        Validates that annotation intervals strictly reside within the recorded duration.

        Args:
            annotations_df: DataFrame from parse()
            recording_durations: Mapping of recording_id -> duration in seconds

        Returns:
            Validated DataFrame. Raises ValueError if invalid.
        """
        for _, row in annotations_df.iterrows():
            rec_id = row["recording_id"]
            start = row["start_seconds"]
            end = row["end_seconds"]

            if start >= end:
                raise ValueError(
                    f"Invalid annotation in {rec_id}: start_seconds ({start}) >= end_seconds ({end})"
                )

            if rec_id in recording_durations:
                max_dur = recording_durations[rec_id]
                if end > max_dur + 1.0:  # Allow 1 second boundary tolerance
                    raise ValueError(
                        f"Annotation in {rec_id} ends at {end}s, which exceeds file duration {max_dur}s"
                    )

        return annotations_df

    def save_to_csv(self, df: pd.DataFrame, output_path: str = "outputs/expert_annotations.csv") -> str:
        """
        Saves annotations DataFrame to CSV file.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        return str(out_file)
