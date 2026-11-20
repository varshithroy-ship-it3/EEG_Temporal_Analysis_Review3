"""
Evidence Extractor Module for EEG Temporal Progression.
Extracts rigorous quantitative evidence (onset latency, termination offset, overlap ratios,
duration comparisons, and traceability anchors) for each computed temporal relationship.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


class EvidenceExtractor:
    """
    Extracts quantitative, verifiable evidence metrics linking model predictions
    to clinical expert annotations.
    """

    def extract_relationship_evidence(
        self,
        rel_row: Dict[str, Any],
        events_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Computes detailed evidentiary parameters for a relationship record.
        """
        evt_a_id = rel_row["event_a_id"]
        evt_b_id = rel_row["event_b_id"]

        match_a = events_df[events_df["event_id"] == evt_a_id]
        match_b = events_df[events_df["event_id"] == evt_b_id]

        row_a = match_a.iloc[0].to_dict() if not match_a.empty else {}
        row_b = match_b.iloc[0].to_dict() if not match_b.empty else {}

        s_a = float(rel_row["event_a_start"])
        e_a = float(rel_row["event_a_end"])
        s_b = float(rel_row["event_b_start"])
        e_b = float(rel_row["event_b_end"])

        dur_a = max(0.0, e_a - s_a)
        dur_b = max(0.0, e_b - s_b)
        overlap = float(rel_row.get("overlap_seconds", 0.0))
        gap = float(rel_row.get("gap_seconds", 0.0))

        onset_latency = round(s_a - s_b, 3)  # Negative if A preceded B
        offset_latency = round(e_a - e_b, 3)

        overlap_ratio_a = round(overlap / dur_a, 4) if dur_a > 0 else 0.0
        overlap_ratio_b = round(overlap / dur_b, 4) if dur_b > 0 else 0.0

        confidence_a = row_a.get("confidence", 1.0)

        evidence = {
            "relationship_id": rel_row["relationship_id"],
            "recording_id": rel_row["recording_id"],
            "event_a_id": evt_a_id,
            "event_b_id": evt_b_id,
            "relation_type": rel_row["relation_type"],
            "event_a_duration": dur_a,
            "event_b_duration": dur_b,
            "onset_latency_seconds": onset_latency,
            "offset_latency_seconds": offset_latency,
            "overlap_seconds": overlap,
            "gap_seconds": gap,
            "overlap_ratio_relative_to_prediction": overlap_ratio_a,
            "overlap_ratio_relative_to_expert": overlap_ratio_b,
            "prediction_confidence": confidence_a,
            "evidence_reference": rel_row["evidence_reference"],
        }
        return evidence
