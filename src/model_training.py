"""
Machine Learning Model Training and Leakage-Aware Evaluation.
Trains a Scikit-Learn pipeline (StandardScaler + RandomForestClassifier) with
strict Leave-One-Recording-Out cross-validation to prevent adjacent-window data leakage.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import EEGEvaluator
from src.feature_extraction import EEGFeatureExtractor


class EEGModelTrainer:
    """
    Manages leakage-aware model training, evaluation, and serialization.
    """

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 150,
        max_depth: Optional[int] = 12,
        model_path: str = "models/model.joblib",
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def create_pipeline(self) -> Pipeline:
        """
        Creates a clean Scikit-Learn pipeline with feature scaling and Random Forest.
        Uses class_weight='balanced' to handle severe class imbalance (rare seizure events).
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
            ),
        ])

    def evaluate_leave_one_recording_out(
        self,
        features_df: pd.DataFrame,
        evaluator: Optional[EEGEvaluator] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Performs Leave-One-Recording-Out (LORO) cross-validation across recordings.
        Prevents adjacent-window temporal leakage by isolating whole recordings.

        Returns:
            Tuple of:
            - metrics dictionary
            - DataFrame with out-of-fold predictions
        """
        if evaluator is None:
            evaluator = EEGEvaluator()

        feature_cols = EEGFeatureExtractor.get_feature_column_names(features_df)
        recordings = sorted(features_df["recording_id"].unique())

        if len(recordings) < 2:
            raise ValueError(
                "Leave-One-Recording-Out requires at least 2 distinct recordings. "
                f"Found: {recordings}"
            )

        oof_df = features_df.copy()
        oof_df["oof_pred"] = 0
        oof_df["oof_prob"] = 0.0

        for test_rec in recordings:
            train_mask = features_df["recording_id"] != test_rec
            test_mask = features_df["recording_id"] == test_rec

            X_train = features_df.loc[train_mask, feature_cols].values
            y_train = features_df.loc[train_mask, "label"].values.astype(int)

            X_test = features_df.loc[test_mask, feature_cols].values
            y_test = features_df.loc[test_mask, "label"].values.astype(int)

            # Ensure both classes exist in train partition
            if len(np.unique(y_train)) < 2:
                raise ValueError(
                    f"Training partition excluding {test_rec} lacks both classes."
                )

            fold_pipe = self.create_pipeline()
            fold_pipe.fit(X_train, y_train)

            test_preds = fold_pipe.predict(X_test)
            test_probs = fold_pipe.predict_proba(X_test)[:, 1]

            oof_df.loc[test_mask, "oof_pred"] = test_preds
            oof_df.loc[test_mask, "oof_prob"] = test_probs

        # Compute global out-of-fold metrics
        y_true = oof_df["label"].values.astype(int)
        y_pred = oof_df["oof_pred"].values.astype(int)
        y_prob = oof_df["oof_prob"].values

        notes = (
            f"Evaluated across {len(recordings)} recordings: {', '.join(recordings)}. "
            f"Total windows: {len(y_true)}, Seizure windows: {int(np.sum(y_true))}."
        )

        metrics = evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            strategy_name="Leave-One-Recording-Out Cross-Validation (Cross-Recording Split)",
            n_train_samples=len(y_true) // 2,
            n_test_samples=len(y_true),
            notes=notes,
        )

        # Generate evaluation plots and json
        evaluator.plot_confusion_matrix(y_true, y_pred)
        evaluator.plot_roc_curve(y_true, y_prob)
        evaluator.save_metrics_json(metrics)

        return metrics, oof_df

    def train_final_model(
        self,
        features_df: pd.DataFrame,
    ) -> Pipeline:
        """
        Trains the final production pipeline on all available validated windows
        and serializes it to disk.
        """
        feature_cols = EEGFeatureExtractor.get_feature_column_names(features_df)
        X = features_df[feature_cols].values
        y = features_df["label"].values.astype(int)

        final_pipe = self.create_pipeline()
        final_pipe.fit(X, y)

        joblib.dump(final_pipe, self.model_path)
        return final_pipe
