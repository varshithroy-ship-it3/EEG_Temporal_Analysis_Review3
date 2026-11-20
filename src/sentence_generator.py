"""
Evidence-Grounded Sentence Generator.
Generates deterministic, auditably grounded clinical descriptions from Allen Interval
relationships and empirical timing evidence without unsubstantiated causal claims.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.evidence_extractor import EvidenceExtractor


class SentenceGenerator:
    """
    Deterministic sentence synthesis engine translating temporal interval relations
    into scientifically precise, verifiable statements grounded in recording-relative seconds.
    """

    def __init__(self):
        self.evidence_extractor = EvidenceExtractor()

    def generate_sentence(self, evidence: Dict[str, Any]) -> str:
        """
        Synthesizes an evidence-grounded statement from a structured evidence dictionary.
        Strictly describes temporal alignment rather than claiming physiological causation.
        """
        rel_type = evidence["relation_type"]
        rec_id = evidence["recording_id"]
        evt_a = evidence["event_a_id"]
        evt_b = evidence["event_b_id"]
        overlap = evidence["overlap_seconds"]
        gap = evidence["gap_seconds"]
        onset_lat = evidence["onset_latency_seconds"]
        offset_lat = evidence["offset_latency_seconds"]
        conf = evidence.get("prediction_confidence", 1.0)
        conf_str = f" (mean confidence: {conf*100:.1f}%)" if conf is not None else ""

        if rel_type == "EQUALS":
            return (
                f"In recording {rec_id}, the model-predicted seizure interval ({evt_a}){conf_str} "
                f"was temporally concurrent with the expert-annotated seizure interval ({evt_b}) "
                f"across its entire {overlap:.1f} second span."
            )

        elif rel_type == "OVERLAPS":
            return (
                f"In recording {rec_id}, the model-predicted seizure interval ({evt_a}){conf_str} "
                f"temporally preceded and overlapped with expert annotation ({evt_b}) by {overlap:.1f} seconds "
                f"(prediction onset was {abs(onset_lat):.1f}s prior to expert onset)."
            )

        elif rel_type == "OVERLAPPED_BY":
            return (
                f"In recording {rec_id}, the model-predicted seizure interval ({evt_a}){conf_str} "
                f"overlapped with the latter portion of expert annotation ({evt_b}) by {overlap:.1f} seconds "
                f"(prediction onset occurred {onset_lat:.1f}s after expert onset)."
            )

        elif rel_type == "DURING":
            return (
                f"In recording {rec_id}, the model-predicted seizure interval ({evt_a}){conf_str} "
                f"occurred entirely during expert-annotated seizure interval ({evt_b}), "
                f"spanning {overlap:.1f} seconds within the annotated window."
            )

        elif rel_type == "CONTAINS":
            return (
                f"In recording {rec_id}, the model-predicted seizure episode ({evt_a}){conf_str} "
                f"temporally contained the entire expert-annotated seizure interval ({evt_b}) "
                f"with {overlap:.1f} seconds of mutual coverage."
            )

        elif rel_type == "MEETS":
            return (
                f"In recording {rec_id}, the model-predicted episode ({evt_a}){conf_str} "
                f"ended exactly at the onset of expert-annotated seizure interval ({evt_b}) "
                f"without an intervening temporal gap."
            )

        elif rel_type == "MET_BY":
            return (
                f"In recording {rec_id}, the model-predicted episode ({evt_a}){conf_str} "
                f"began immediately at the termination of expert-annotated seizure interval ({evt_b})."
            )

        elif rel_type == "BEFORE":
            return (
                f"In recording {rec_id}, the model-predicted event ({evt_a}){conf_str} "
                f"temporally preceded expert-annotated seizure interval ({evt_b}), "
                f"terminating {gap:.1f} seconds prior to the annotated seizure onset."
            )

        elif rel_type == "AFTER":
            return (
                f"In recording {rec_id}, the model-predicted event ({evt_a}){conf_str} "
                f"temporally followed expert-annotated seizure interval ({evt_b}), "
                f"starting {gap:.1f} seconds after the annotated seizure concluded."
            )

        elif rel_type == "STARTS":
            return (
                f"In recording {rec_id}, the model-predicted event ({evt_a}){conf_str} "
                f"shared a simultaneous onset with expert-annotated seizure ({evt_b}) "
                f"and terminated earlier, yielding {overlap:.1f} seconds of co-occurrence."
            )

        elif rel_type == "STARTED_BY":
            return (
                f"In recording {rec_id}, the model-predicted event ({evt_a}){conf_str} "
                f"began concurrently with expert-annotated seizure ({evt_b}) "
                f"and persisted beyond it for a total overlap of {overlap:.1f} seconds."
            )

        elif rel_type == "FINISHES":
            return (
                f"In recording {rec_id}, the model-predicted event ({evt_a}){conf_str} "
                f"commenced after the expert-annotated seizure ({evt_b}) and terminated "
                f"simultaneously with it ({overlap:.1f} seconds shared overlap)."
            )

        elif rel_type == "FINISHED_BY":
            return (
                f"In recording {rec_id}, the model-predicted event ({evt_a}){conf_str} "
                f"commenced before the expert-annotated seizure ({evt_b}) and terminated "
                f"simultaneously with it ({overlap:.1f} seconds shared overlap)."
            )

        else:
            return (
                f"In recording {rec_id}, model-predicted interval ({evt_a}){conf_str} "
                f"was associated in time with expert annotation ({evt_b}) under relation {rel_type} "
                f"with {overlap:.1f} seconds overlap and {gap:.1f} seconds gap."
            )

    def generate_all_sentences(
        self,
        rel_df: pd.DataFrame,
        events_df: pd.DataFrame,
        output_csv_path: str = "outputs/generated_sentences.csv",
    ) -> pd.DataFrame:
        """
        Iterates over relationships, extracts evidence, generates sentences,
        and saves outputs/generated_sentences.csv.

        Required columns:
        - sentence_id
        - recording_id
        - relationship_id
        - event_a_id
        - event_b_id
        - relation_type
        - generated_sentence
        - evidence_reference
        """
        records = []
        sent_counter = 1

        for _, row in rel_df.iterrows():
            rel_dict = row.to_dict()
            evidence = self.evidence_extractor.extract_relationship_evidence(rel_dict, events_df)
            sentence_text = self.generate_sentence(evidence)

            sent_id = f"SENT_{evidence['recording_id']}_{sent_counter:03d}"
            s_a = float(row.get("event_a_start", 0.0))
            e_a = float(row.get("event_a_end", 0.0))
            s_b = float(row.get("event_b_start", 0.0))
            e_b = float(row.get("event_b_end", 0.0))
            overlap = float(evidence.get("overlap_seconds", 0.0))
            gap = float(evidence.get("gap_seconds", 0.0))

            records.append({
                "sentence_id": sent_id,
                "Sentence_ID": sent_id,
                "recording_id": evidence["recording_id"],
                "Recording_ID": evidence["recording_id"],
                "relationship_id": evidence["relationship_id"],
                "Relationship_ID": evidence["relationship_id"],
                "event_a_id": evidence["event_a_id"],
                "Event_A_ID": evidence["event_a_id"],
                "event_b_id": evidence["event_b_id"],
                "Event_B_ID": evidence["event_b_id"],
                "relation_type": evidence["relation_type"],
                "Relationship_Type": evidence["relation_type"],
                "event_a_start": s_a,
                "event_a_end": e_a,
                "event_b_start": s_b,
                "event_b_end": e_b,
                "overlap_seconds": overlap,
                "Overlap_Seconds": overlap,
                "gap_seconds": gap,
                "Gap_Seconds": gap,
                "generated_sentence": sentence_text,
                "Generated_Sentence": sentence_text,
                "evidence_reference": evidence["evidence_reference"],
                "Evidence_Reference": evidence["evidence_reference"],
            })
            sent_counter += 1

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                "sentence_id", "recording_id", "relationship_id",
                "event_a_id", "event_b_id", "relation_type",
                "generated_sentence", "evidence_reference",
                "Event_A_ID", "Event_B_ID", "Relationship_Type", "Generated_Sentence",
                "Overlap_Seconds", "Gap_Seconds"
            ])

        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        return df
