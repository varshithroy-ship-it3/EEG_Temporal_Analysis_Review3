"""
Window Prediction Output Module.
Generates window-level model predictions with confidence probabilities, keeping
predictions explicitly distinct from expert ground-truth annotations.
"""

from pathlib import Path
from typing import Optional, Union
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.feature_extraction import EEGFeatureExtractor


class WindowPredictor:
    """
    Applies the trained seizure classification model to extract window-level predictions.
    """

    def __init__(self, model_source: Union[str, Pipeline] = "models/model.joblib"):
        if isinstance(model_source, str):
            model_path = Path(model_source)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.model: Pipeline = joblib.load(model_path)
        else:
            self.model = model_source

    def predict(
        self,
        features_df: pd.DataFrame,
        output_csv_path: str = "outputs/window_predictions.csv",
    ) -> pd.DataFrame:
        """
        Runs model inference on the window features and produces the required output table.

        Required columns:
        - window_id
        - recording_id
        - start_seconds
        - end_seconds
        - predicted_label
        - predicted_probability
        - expert_label
        - annotation_overlap_seconds
        - annotation_overlap_ratio

        Returns:
            pd.DataFrame and writes to outputs/window_predictions.csv
        """
        feature_cols = EEGFeatureExtractor.get_feature_column_names(features_df)
        X = features_df[feature_cols].values

        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]

        out_df = pd.DataFrame({
            "window_id": features_df["window_id"].values,
            "recording_id": features_df["recording_id"].values,
            "start_seconds": features_df["start_seconds"].values,
            "end_seconds": features_df["end_seconds"].values,
            "predicted_label": preds.astype(int),
            "predicted_probability": probs.round(4),
            "expert_label": features_df["label"].values.astype(int) if "label" in features_df.columns else None,
            "annotation_overlap_seconds": features_df.get("overlap_seconds", 0.0),
            "annotation_overlap_ratio": features_df.get("overlap_ratio", 0.0),
        })

        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

        return out_df
