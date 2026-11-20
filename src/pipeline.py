"""
Master Pipeline Orchestrator for EEG Temporal Analysis System.
Executes the full end-to-end scientific pipeline:
1. Load EDF recordings and parse expert annotations.
2. Filter and preprocess EEG signals.
3. Segment continuous recordings into overlapping 2-second windows.
4. Extract 26 physiological time- and frequency-domain features per window.
5. Train Random Forest model with leakage-aware Leave-One-Recording-Out cross-validation.
6. Generate window predictions and confidence scores.
7. Reconstruct interval-based temporal seizure events.
8. Identify strict Allen Interval Algebra temporal relationships.
9. Extract evidentiary metrics and generate deterministic relationship sentences.
10. Render NetworkX relationship graph and evaluation diagnostic plots.
11. Serialize all structured CSV, JSON, and PNG deliverables.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.data_loader import EDFDataLoader
from src.annotation_parser import AnnotationParser
from src.preprocessing import EEGPreprocessor
from src.window_segmentation import EEGWindowSegmenter
from src.feature_extraction import EEGFeatureExtractor
from src.model_training import EEGModelTrainer
from src.evaluation import EEGEvaluator
from src.window_prediction import WindowPredictor
from src.temporal_engine import TemporalEngine
from src.relationship_engine import AllenIntervalEngine
from src.evidence_extractor import EvidenceExtractor
from src.sentence_generator import SentenceGenerator
from src.graph_generator import GraphGenerator


def run_pipeline(
    data_dir: str = "data/chbmit/chb01",
    summary_filename: str = "chb01-summary.txt",
    recording_files: Optional[List[str]] = None,
    output_dir: str = "outputs",
    model_dir: str = "models",
) -> Dict[str, Any]:
    """
    Executes the complete EEG temporal analysis pipeline.
    """
    start_total_time = time.time()
    print("=" * 80)
    print("EEG TEMPORAL ANALYSIS SYSTEM — FINAL YEAR REVIEW 3 PIPELINE")
    print("=" * 80)

    if recording_files is None:
        recording_files = ["chb01_03.edf", "chb01_04.edf"]

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mod_path = Path(model_dir)
    mod_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STAGE 1: Parse Annotations and Validate Metadata
    # -------------------------------------------------------------------------
    print("\n[STAGE 1/8] Parsing Expert Annotations & Reading EDF Metadata...")
    summary_path = data_path / summary_filename
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    parser = AnnotationParser(str(summary_path))
    annotations_df = parser.parse(target_recordings=recording_files)
    print(f"  -> Found {len(annotations_df)} expert seizure annotations for target files.")
    for _, r in annotations_df.iterrows():
        print(f"     * {r['recording_id']}: [{r['start_seconds']}s - {r['end_seconds']}s] ({r['end_seconds'] - r['start_seconds']}s duration)")

    loader = EDFDataLoader(data_dir=str(data_path))
    rec_infos: Dict[str, Dict[str, Any]] = {}
    durations: Dict[str, float] = {}

    for rec_file in recording_files:
        info = loader.get_recording_info(rec_file)
        rec_id = info["recording_id"]
        rec_infos[rec_id] = info
        durations[rec_id] = info["duration_seconds"]
        print(f"  -> Recording '{rec_id}': {info['n_channels']} channels, sfreq={info['sfreq']}Hz, duration={info['duration_seconds']}s")

    # Validate annotations against duration
    parser.validate_durations(annotations_df, durations)
    expert_csv_path = out_path / "expert_annotations.csv"
    parser.save_to_csv(annotations_df, str(expert_csv_path))
    print(f"  [SAVED] {expert_csv_path}")

    # Find common channels
    common_channels, missing_report = loader.find_common_channels([str(data_path / f) for f in recording_files])
    print(f"  -> Identified {len(common_channels)} shared channels across recordings.")
    if missing_report:
        print(f"  -> Warning: Channel discrepancies found: {missing_report}")

    # -------------------------------------------------------------------------
    # STAGE 2: Signal Preprocessing, Segmentation, and Feature Extraction
    # -------------------------------------------------------------------------
    print("\n[STAGE 2/8] Preprocessing EEG Signals & Segmenting Overlapping Windows...")
    preprocessor = EEGPreprocessor(l_freq=0.5, h_freq=45.0, notch_freq=60.0)
    segmenter = EEGWindowSegmenter(window_duration_sec=2.0, step_size_sec=1.0, seizure_overlap_threshold=0.50)
    print(f"  -> Window Policy: {segmenter.policy_description}")

    extractor = EEGFeatureExtractor(sfreq=256.0)
    all_features_list = []

    for rec_file in recording_files:
        t_rec_start = time.time()
        rec_path = data_path / rec_file
        print(f"  -> Loading & filtering '{rec_file}'...")
        raw = loader.load_raw(str(rec_path), preload=True, channels=common_channels, verbose=False)

        # Preprocess
        raw_clean = preprocessor.preprocess_raw(raw, copy=False, verbose=False)
        rec_id = Path(rec_file).stem
        sfreq = raw_clean.info["sfreq"]
        data_clean = raw_clean.get_data()

        quality = preprocessor.check_signal_quality(data_clean, sfreq)
        print(f"     Signal Quality Audit: Valid={quality['is_valid']}, Saturated Ratio={quality['saturation_ratio']:.4f}, Mean RMS={quality['rms_amplitude']:.2e}V")

        # Segment windows
        windows_meta, window_signals = segmenter.segment_recording(
            raw_data=data_clean,
            sfreq=sfreq,
            recording_id=rec_id,
            channel_names=common_channels,
            annotations_df=annotations_df,
        )
        print(f"     Segmented {len(windows_meta)} windows (2.0s each, 1.0s step).")

        # Feature Extraction
        print(f"     Extracting time & spectral features across {len(windows_meta)} windows...")
        feat_df = extractor.extract_features_batch(window_signals, windows_meta)
        all_features_list.append(feat_df)

        raw.close()
        print(f"     Completed {rec_id} in {time.time() - t_rec_start:.2f}s.")

    combined_features_df = pd.concat(all_features_list, ignore_index=True)
    feature_cols = EEGFeatureExtractor.get_feature_column_names(combined_features_df)
    total_sz_windows = int(combined_features_df["label"].sum())
    print(f"  -> Feature Table: {combined_features_df.shape[0]} windows x {len(feature_cols)} features.")
    print(f"  -> Class Distribution: Non-Seizure={len(combined_features_df) - total_sz_windows}, Seizure={total_sz_windows} windows.")

    # -------------------------------------------------------------------------
    # STAGE 3: Leakage-Aware Model Training & Out-of-Fold Evaluation
    # -------------------------------------------------------------------------
    print("\n[STAGE 3/8] Leakage-Aware Machine Learning Training & Evaluation...")
    evaluator = EEGEvaluator(output_dir=str(out_path))
    trainer = EEGModelTrainer(
        random_state=42,
        n_estimators=150,
        max_depth=12,
        model_path=str(mod_path / "model.joblib"),
    )

    print("  -> Performing Leave-One-Recording-Out (LORO) cross-validation (zero window leakage)...")
    eval_metrics, oof_df = trainer.evaluate_leave_one_recording_out(combined_features_df, evaluator=evaluator)

    print(f"     Accuracy : {eval_metrics['accuracy']:.4f}")
    print(f"     Precision: {eval_metrics['precision']:.4f}")
    print(f"     Recall   : {eval_metrics['recall']:.4f}")
    print(f"     F1-Score : {eval_metrics['f1_score']:.4f}")
    print(f"     ROC-AUC  : {eval_metrics['roc_auc'] if eval_metrics['roc_auc'] is not None else 'N/A'}")
    print(f"     Confusion Matrix: {eval_metrics['confusion_matrix']}")

    # Train final unified pipeline
    print("  -> Training final production model pipeline on all verified windows...")
    final_pipeline = trainer.train_final_model(combined_features_df)
    print(f"  [SAVED] {trainer.model_path}")
    print(f"  [SAVED] {out_path / 'evaluation_metrics.json'}")
    print(f"  [SAVED] {out_path / 'confusion_matrix.png'}")
    print(f"  [SAVED] {out_path / 'roc_curve.png'}")

    # -------------------------------------------------------------------------
    # STAGE 4: Window Prediction Output
    # -------------------------------------------------------------------------
    print("\n[STAGE 4/8] Generating Window Prediction Output...")
    predictor = WindowPredictor(model_source=final_pipeline)
    pred_csv_path = out_path / "window_predictions.csv"
    window_preds_df = predictor.predict(combined_features_df, output_csv_path=str(pred_csv_path))
    pos_preds_count = int(window_preds_df["predicted_label"].sum())
    print(f"  -> Generated {len(window_preds_df)} window predictions.")
    print(f"  -> Positive predicted windows: {pos_preds_count}")
    print(f"  [SAVED] {pred_csv_path}")

    # -------------------------------------------------------------------------
    # STAGE 5: Temporal Progression Reconstruction
    # -------------------------------------------------------------------------
    print("\n[STAGE 5/8] Reconstructing Temporal Progression Events...")
    temporal_engine = TemporalEngine(max_gap_tolerance_sec=2.0, min_episode_duration_sec=2.0)
    events_csv_path = out_path / "temporal_events.csv"
    events_df = temporal_engine.compile_all_events(
        pred_df=window_preds_df,
        expert_df=annotations_df,
        output_csv_path=str(events_csv_path),
    )

    pred_events_count = len(events_df[events_df["source"] == "model_prediction"])
    expert_events_count = len(events_df[events_df["source"] == "expert_annotation"])
    print(f"  -> Compiled {len(events_df)} total temporal events:")
    print(f"     * Expert-Annotated Seizures: {expert_events_count}")
    print(f"     * Model-Predicted Seizure Episodes: {pred_events_count}")
    for _, ev in events_df.iterrows():
        print(f"       [{ev['source'].upper()}] {ev['event_id']} ({ev['recording_id']}): {ev['start_seconds']}s - {ev['end_seconds']}s (dur={ev['duration_seconds']}s, conf={ev['confidence']})")
    print(f"  [SAVED] {events_csv_path}")

    # -------------------------------------------------------------------------
    # STAGE 6: Allen Interval Algebra Relationships
    # -------------------------------------------------------------------------
    print("\n[STAGE 6/8] Identifying Allen Interval Relationships...")
    allen_engine = AllenIntervalEngine(tolerance_sec=0.5)
    rel_csv_path = out_path / "relationships.csv"
    relationships_df = allen_engine.compute_all_relationships(
        events_df=events_df,
        output_csv_path=str(rel_csv_path),
    )
    print(f"  -> Computed {len(relationships_df)} interval relationships:")
    for _, r in relationships_df.iterrows():
        print(f"     * {r['relationship_id']}: {r['event_a_id']} {r['relation_type']} {r['event_b_id']} (overlap={r['overlap_seconds']}s, gap={r['gap_seconds']}s)")
    print(f"  [SAVED] {rel_csv_path}")

    # -------------------------------------------------------------------------
    # STAGE 7: Evidence-Grounded Sentence Generation
    # -------------------------------------------------------------------------
    print("\n[STAGE 7/8] Generating Evidence-Grounded Relationship Sentences...")
    sentence_gen = SentenceGenerator()
    sentences_csv_path = out_path / "generated_sentences.csv"
    sentences_df = sentence_gen.generate_all_sentences(
        rel_df=relationships_df,
        events_df=events_df,
        output_csv_path=str(sentences_csv_path),
    )
    print(f"  -> Generated {len(sentences_df)} auditable clinical sentences:")
    for _, s in sentences_df.iterrows():
        print(f"     * [{s['relation_type']}] \"{s['generated_sentence']}\"")
    print(f"  [SAVED] {sentences_csv_path}")

    # -------------------------------------------------------------------------
    # STAGE 8: NetworkX Relationship Graph Visualization
    # -------------------------------------------------------------------------
    print("\n[STAGE 8/8] Rendering NetworkX Temporal Relationship Graph...")
    graph_gen = GraphGenerator(output_dir=str(out_path))
    G = graph_gen.build_graph(events_df, relationships_df)
    graph_img_path = graph_gen.render_and_save(G, filename="graph.png")
    print(f"  -> Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} directed edges.")
    print(f"  [SAVED] {graph_img_path}")

    # -------------------------------------------------------------------------
    # PIPELINE SUMMARY
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 80)
    print(f"Verified Output Artifacts in '{out_path}':")
    for fname in [
        "expert_annotations.csv",
        "window_predictions.csv",
        "temporal_events.csv",
        "relationships.csv",
        "generated_sentences.csv",
        "evaluation_metrics.json",
        "confusion_matrix.png",
        "roc_curve.png",
        "graph.png",
    ]:
        p = out_path / fname
        status = "EXISTS" if p.exists() else "MISSING"
        size = p.stat().st_size if p.exists() else 0
        print(f"  [{status}] {fname:<25} ({size:,} bytes)")

    return {
        "status": "success",
        "elapsed_seconds": round(elapsed, 2),
        "expert_annotations_count": len(annotations_df),
        "total_windows": len(window_preds_df),
        "temporal_events_count": len(events_df),
        "relationships_count": len(relationships_df),
        "sentences_count": len(sentences_df),
        "metrics": eval_metrics,
    }


if __name__ == "__main__":
    run_pipeline()
