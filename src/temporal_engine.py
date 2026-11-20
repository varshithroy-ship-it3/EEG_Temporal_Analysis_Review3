"""
Temporal Progression Engine for EEG Seizure Detection.
Reconstructs interval-based temporal events by grouping contiguous positive prediction
windows into coherent seizure episodes and integrating expert clinical annotations.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


class TemporalEngine:
    """
    Groups window-level classifications into temporal episode intervals and compiles
    both model predictions and expert annotations into a unified temporal event schema.
    """

    def __init__(
        self,
        max_gap_tolerance_sec: float = 2.0,
        min_episode_duration_sec: float = 2.0,
    ):
        """
        Args:
            max_gap_tolerance_sec: Maximum time gap (seconds) between positive windows
                                   to still bridge them into a continuous episode.
            min_episode_duration_sec: Minimum duration (seconds) required for a predicted episode.
        """
        self.max_gap_tolerance_sec = float(max_gap_tolerance_sec)
        self.min_episode_duration_sec = float(min_episode_duration_sec)

    def group_prediction_windows(
        self,
        pred_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Groups contiguous or overlapping positive prediction windows into episodic intervals.

        Args:
            pred_df: DataFrame with columns ['window_id', 'recording_id', 'start_seconds',
                     'end_seconds', 'predicted_label', 'predicted_probability']

        Returns:
            List of predicted event dictionaries.
        """
        predicted_events: List[Dict[str, Any]] = []

        # Process per recording independently to maintain strict relative timelines
        for rec_id, group in pred_df.groupby("recording_id"):
            pos_windows = group[group["predicted_label"] == 1].sort_values("start_seconds").to_dict("records")
            if not pos_windows:
                continue

            current_cluster: List[Dict[str, Any]] = [pos_windows[0]]
            episode_counter = 1

            for win in pos_windows[1:]:
                prev_win = current_cluster[-1]
                # Check for overlap or gap within tolerance
                if win["start_seconds"] <= prev_win["end_seconds"] + self.max_gap_tolerance_sec:
                    current_cluster.append(win)
                else:
                    # Finalize current episode
                    ep = self._create_predicted_event(rec_id, episode_counter, current_cluster)
                    if ep["duration_seconds"] >= self.min_episode_duration_sec:
                        predicted_events.append(ep)
                        episode_counter += 1
                    current_cluster = [win]

            if current_cluster:
                ep = self._create_predicted_event(rec_id, episode_counter, current_cluster)
                if ep["duration_seconds"] >= self.min_episode_duration_sec:
                    predicted_events.append(ep)

        return predicted_events

    def _create_predicted_event(
        self,
        recording_id: str,
        counter: int,
        windows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Creates a structured event record from clustered windows.
        """
        start_sec = min(w["start_seconds"] for w in windows)
        end_sec = max(w["end_seconds"] for w in windows)
        probs = [w["predicted_probability"] for w in windows]
        window_ids = [w["window_id"] for w in windows]

        event_id = f"PRED_{recording_id}_EP{counter:02d}"

        return {
            "event_id": event_id,
            "recording_id": recording_id,
            "event_type": "predicted_seizure",
            "start_seconds": round(float(start_sec), 3),
            "end_seconds": round(float(end_sec), 3),
            "duration_seconds": round(float(end_sec - start_sec), 3),
            "source": "model_prediction",
            "source_type": "random_forest_classifier",
            "confidence": round(float(sum(probs) / len(probs)), 4),
            "max_confidence": round(float(max(probs)), 4),
            "window_count": len(windows),
            "constituent_windows": ";".join(window_ids),
            "supporting_annotation_id": None,
        }

    def compile_all_events(
        self,
        pred_df: pd.DataFrame,
        expert_df: pd.DataFrame,
        output_csv_path: str = "outputs/temporal_events.csv",
    ) -> pd.DataFrame:
        """
        Compiles both expert clinical annotations and grouped model predictions
        into a unified, traceable temporal event table.

        Required columns:
        - event_id
        - recording_id
        - event_type
        - start_seconds
        - end_seconds
        - source
        - source_type
        - confidence
        - supporting_annotation_id

        Returns:
            pd.DataFrame and writes to outputs/temporal_events.csv
        """
        all_events: List[Dict[str, Any]] = []

        # 1. Format Expert Annotations
        for _, row in expert_df.iterrows():
            rec_id = str(row["recording_id"])
            evt_id = str(row["event_id"])
            s = float(row["start_seconds"])
            e = float(row["end_seconds"])
            all_events.append({
                "event_id": evt_id,
                "recording_id": rec_id,
                "event_type": "expert_seizure",
                "start_seconds": round(s, 3),
                "end_seconds": round(e, 3),
                "duration_seconds": round(e - s, 3),
                "source": "expert_annotation",
                "source_type": "clinical_ground_truth",
                "confidence": 1.0,
                "max_confidence": 1.0,
                "window_count": None,
                "constituent_windows": None,
                "supporting_annotation_id": evt_id,
            })

        # 2. Format Grouped Model Predictions
        pred_events = self.group_prediction_windows(pred_df)
        all_events.extend(pred_events)

        events_df = pd.DataFrame(all_events)
        if not events_df.empty:
            events_df = events_df.sort_values(by=["recording_id", "start_seconds"]).reset_index(drop=True)

        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        events_df.to_csv(out_path, index=False)

        return events_df
