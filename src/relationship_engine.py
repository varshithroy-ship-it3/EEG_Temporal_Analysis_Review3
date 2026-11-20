"""
Allen Interval Algebra Relationship Engine for EEG Seizure Detection.
Implements the 13 formal interval relations of Allen's Interval Algebra:
Before, After, Meets, Met-By, Overlaps, Overlapped-By, During, Contains,
Starts, Started-By, Finishes, Finished-By, Equals.
Includes floating-point boundary tolerance and computes exact overlap/gap evidence.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd


class AllenIntervalEngine:
    """
    Computes strict Allen Interval Algebra temporal relationships between event intervals
    with configurable boundary tolerance.
    """

    def __init__(self, tolerance_sec: float = 0.5):
        """
        Args:
            tolerance_sec: Epsilon tolerance in seconds for boundary comparisons
                           (accounting for discrete window discretization).
        """
        self.tolerance_sec = float(tolerance_sec)

    def _eq(self, t1: float, t2: float) -> bool:
        return abs(t1 - t2) <= self.tolerance_sec

    def _lt(self, t1: float, t2: float) -> bool:
        return (t1 - t2) < -self.tolerance_sec

    def _gt(self, t1: float, t2: float) -> bool:
        return (t1 - t2) > self.tolerance_sec

    def classify_interval_relation(
        self,
        s_a: float,
        e_a: float,
        s_b: float,
        e_b: float,
    ) -> Tuple[str, float, float]:
        """
        Classifies the exact Allen Interval relation between Interval A and Interval B.

        Returns:
            Tuple of:
            - relation_name (str)
            - overlap_seconds (float)
            - gap_seconds (float)
        """
        # Calculate metric overlap and gap
        intersection = max(0.0, min(e_a, e_b) - max(s_a, s_b))
        gap = 0.0
        if intersection <= 0.0:
            if e_a <= s_b:
                gap = s_b - e_a
            elif e_b <= s_a:
                gap = s_a - e_b

        # 1. Equals: sA == sB and eA == eB
        if self._eq(s_a, s_b) and self._eq(e_a, e_b):
            return "EQUALS", round(intersection, 3), round(gap, 3)

        # 2. Before: eA < sB
        if self._lt(e_a, s_b):
            return "BEFORE", 0.0, round(gap, 3)

        # 3. After: sA > eB (Inverse of Before)
        if self._gt(s_a, e_b):
            return "AFTER", 0.0, round(gap, 3)

        # 4. Meets: eA == sB
        if self._eq(e_a, s_b):
            return "MEETS", 0.0, 0.0

        # 5. Met-By: sA == eB (Inverse of Meets)
        if self._eq(s_a, e_b):
            return "MET_BY", 0.0, 0.0

        # 6. Starts: sA == sB and eA < eB
        if self._eq(s_a, s_b) and self._lt(e_a, e_b):
            return "STARTS", round(intersection, 3), 0.0

        # 7. Started-By: sA == sB and eA > eB (Inverse of Starts)
        if self._eq(s_a, s_b) and self._gt(e_a, e_b):
            return "STARTED_BY", round(intersection, 3), 0.0

        # 8. Finishes: sA > sB and eA == eB
        if self._gt(s_a, s_b) and self._eq(e_a, e_b):
            return "FINISHES", round(intersection, 3), 0.0

        # 9. Finished-By: sA < sB and eA == eB (Inverse of Finishes)
        if self._lt(s_a, s_b) and self._eq(e_a, e_b):
            return "FINISHED_BY", round(intersection, 3), 0.0

        # 10. During: sA > sB and eA < eB
        if self._gt(s_a, s_b) and self._lt(e_a, e_b):
            return "DURING", round(intersection, 3), 0.0

        # 11. Contains: sA < sB and eA > eB (Inverse of During)
        if self._lt(s_a, s_b) and self._gt(e_a, e_b):
            return "CONTAINS", round(intersection, 3), 0.0

        # 12. Overlaps: sA < sB and sB < eA < eB
        if self._lt(s_a, s_b) and s_b <= e_a and self._lt(e_a, e_b):
            return "OVERLAPS", round(intersection, 3), 0.0

        # 13. Overlapped-By: sB < sA and sA < eB < eA (Inverse of Overlaps)
        if self._lt(s_b, s_a) and s_a <= e_b and self._lt(e_b, e_a):
            return "OVERLAPPED_BY", round(intersection, 3), 0.0

        # Fallback for borderline continuous overlap cases
        if intersection > 0:
            if s_a <= s_b:
                return "OVERLAPS", round(intersection, 3), 0.0
            else:
                return "OVERLAPPED_BY", round(intersection, 3), 0.0

        return "BEFORE" if e_a < s_b else "AFTER", 0.0, round(gap, 3)

    def compute_all_relationships(
        self,
        events_df: pd.DataFrame,
        output_csv_path: str = "outputs/relationships.csv",
    ) -> pd.DataFrame:
        """
        Computes pairwise temporal relationships between events belonging to the same recording.
        Specifically pairs model-predicted events with expert annotations (and vice-versa).

        Returns:
            pd.DataFrame and writes to outputs/relationships.csv
        """
        relationships: List[Dict[str, Any]] = []
        rel_counter = 1

        for rec_id, group in events_df.groupby("recording_id"):
            preds = group[group["source"] == "model_prediction"].to_dict("records")
            experts = group[group["source"] == "expert_annotation"].to_dict("records")

            if not preds or not experts:
                continue

            for p in preds:
                for exp in experts:
                    s_a, e_a = p["start_seconds"], p["end_seconds"]
                    s_b, e_b = exp["start_seconds"], exp["end_seconds"]

                    rel_type, overlap, gap = self.classify_interval_relation(s_a, e_a, s_b, e_b)

                    evidence_ref = (
                        f"REC:{rec_id} | EvtA({p['event_id']}):[{s_a:.1f}s, {e_a:.1f}s] "
                        f"| EvtB({exp['event_id']}):[{s_b:.1f}s, {e_b:.1f}s]"
                    )

                    rel_id = f"REL_{rec_id}_{rel_counter:03d}"
                    relationships.append({
                        "relationship_id": rel_id,
                        "recording_id": rec_id,
                        "event_a_id": p["event_id"],
                        "event_b_id": exp["event_id"],
                        "event_a_source": p["source"],
                        "event_b_source": exp["source"],
                        "relation_type": rel_type,
                        "event_a_start": s_a,
                        "event_a_end": e_a,
                        "event_b_start": s_b,
                        "event_b_end": e_b,
                        "overlap_seconds": overlap,
                        "gap_seconds": gap,
                        "evidence_reference": evidence_ref,
                    })
                    rel_counter += 1

        rel_df = pd.DataFrame(relationships)
        if rel_df.empty:
            rel_df = pd.DataFrame(columns=[
                "relationship_id", "recording_id", "event_a_id", "event_b_id",
                "event_a_source", "event_b_source", "relation_type",
                "event_a_start", "event_a_end", "event_b_start", "event_b_end",
                "overlap_seconds", "gap_seconds", "evidence_reference"
            ])

        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rel_df.to_csv(out_path, index=False)

        return rel_df
