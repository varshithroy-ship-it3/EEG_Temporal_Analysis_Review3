"""
Evaluation and Metrics Module for EEG Seizure Detection.
Calculates leakage-aware validation metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
generates publication-grade evaluation plots, and saves structured JSON reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


class EEGEvaluator:
    """
    Computes performance metrics and generates diagnostic plots for seizure classification.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        strategy_name: str = "Leave-One-Recording-Out Cross-Validation",
        n_train_samples: int = 0,
        n_test_samples: int = 0,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Calculates all clinical evaluation metrics, handling edge cases gracefully.

        Returns:
            Dictionary containing metrics and metadata.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Check ROC-AUC validity (requires both classes in y_true)
        unique_classes = np.unique(y_true)
        roc_auc_val: Optional[float] = None
        roc_auc_status: str = "valid"

        if len(unique_classes) < 2:
            roc_auc_status = "undefined_single_class_present"
        elif y_prob is None:
            roc_auc_status = "undefined_no_probabilities_provided"
        else:
            try:
                roc_auc_val = float(roc_auc_score(y_true, y_prob))
            except Exception as e:
                roc_auc_status = f"error_{str(e)}"

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        clf_rep = classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Non-Seizure", "Seizure"],
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "evaluation_strategy": strategy_name,
            "n_train_samples": int(n_train_samples),
            "n_test_samples": int(n_test_samples if n_test_samples > 0 else len(y_true)),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc_val, 4) if roc_auc_val is not None else None,
            "roc_auc_status": roc_auc_status,
            "confusion_matrix": {
                "tn": int(cm[0][0]),
                "fp": int(cm[0][1]),
                "fn": int(cm[1][0]),
                "tp": int(cm[1][1]),
            },
            "classification_report": clf_rep,
            "limitations": (
                "Recordings originate from a single subject (chb01). Results demonstrate "
                "within-subject temporal generalization across distinct seizure events, but do not "
                "establish patient-independent generalization. Model predictions are investigative "
                "and do not constitute clinical diagnostic decisions."
            ),
            "notes": notes,
        }

        return metrics

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str = "confusion_matrix.png",
    ) -> str:
        """
        Generates and saves a styled confusion matrix heatmap with dynamic,
        high-contrast text colors: dark blue cells have white text, and light blue/white
        cells have dark navy text to ensure all numbers are clearly readable.
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)

        # High-contrast medical research dark-navy style
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#1E293B")

        cmap = plt.cm.Blues

        # Create base heatmap without automatic annotations
        sns.heatmap(
            cm,
            annot=False,
            cmap=cmap,
            cbar=False,
            ax=ax,
            xticklabels=["Non-Seizure", "Seizure"],
            yticklabels=["Non-Seizure", "Seizure"],
            linewidths=1.5,
            linecolor="#0F172A",
        )

        # Compute dynamic text color based on cell luminance
        norm = plt.Normalize(vmin=0, vmax=max(cm.max(), 1))
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                norm_val = norm(val)
                rgba = cmap(norm_val)
                # Perceived luminance formula (ITU-R BT.601)
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "#FFFFFF" if luminance < 0.55 else "#0F172A"

                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:,}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=16,
                    fontweight="bold",
                )

        ax.set_title("Out-of-Fold Seizure Confusion Matrix", fontsize=13, weight="bold", color="#F8FAFC", pad=12)
        ax.set_xlabel("Predicted Window Class", fontsize=11, weight="bold", color="#CBD5E1", labelpad=8)
        ax.set_ylabel("Expert Ground Truth", fontsize=11, weight="bold", color="#CBD5E1", labelpad=8)
        ax.tick_params(colors="#94A3B8", labelsize=10)

        out_path = self.output_dir / filename
        plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        return str(out_path)

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        filename: str = "roc_curve.png",
    ) -> Optional[str]:
        """
        Generates and saves the Receiver Operating Characteristic (ROC) curve with
        clear AUC display, matching dimensions, and consistent typography.
        """
        if len(np.unique(y_true)) < 2:
            return None

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#1E293B")

        # ROC line & chance baseline
        ax.plot(fpr, tpr, color="#38BDF8", lw=2.8, label=f"Random Forest (AUC = {auc_val:.4f})")
        ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Chance Baseline (AUC = 0.5000)")

        # Prominent AUC Callout Box inside plot
        ax.text(
            0.52, 0.28,
            f"ROC-AUC = {auc_val:.4f}",
            fontsize=12,
            fontweight="bold",
            color="#38BDF8",
            bbox=dict(boxstyle="round,pad=0.4", fc="#0F172A", ec="#38BDF8", alpha=0.9, lw=1.2),
        )

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight="bold", color="#CBD5E1", labelpad=8)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, weight="bold", color="#CBD5E1", labelpad=8)
        ax.set_title("Out-of-Fold ROC Curve", fontsize=13, weight="bold", color="#F8FAFC", pad=12)
        ax.grid(True, linestyle=":", alpha=0.35, color="#64748B")
        ax.tick_params(colors="#94A3B8", labelsize=10)

        legend = ax.legend(loc="lower right", facecolor="#0F172A", edgecolor="#334155", fontsize=9.5)
        for text in legend.get_texts():
            text.set_color("#F8FAFC")

        out_path = self.output_dir / filename
        plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        return str(out_path)

    def save_metrics_json(
        self,
        metrics: Dict[str, Any],
        filename: str = "evaluation_metrics.json",
    ) -> str:
        """
        Saves evaluation dictionary to formatted JSON.
        """
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        return str(out_path)
