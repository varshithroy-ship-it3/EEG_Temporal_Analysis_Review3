# System for Reconstructing Temporal Progression and Generating Causal Relationship Forms in EEG-Based Seizure Detection

**Final Year B.Tech (AI & ML) Project — Review 3 Implementation**

---

## 1. Project Overview

Continuous scalp electroencephalogram (EEG) monitoring is the clinical standard for identifying and managing epileptic seizure events. However, most machine-learning implementations reduce EEG analysis to isolated, static segment classification without modeling how seizure events unfold over time or verifying how algorithmic predictions relate to expert clinical annotations.

This system introduces an **evidence-grounded temporal analysis framework** applied to the benchmark **CHB-MIT Scalp EEG Database**. The framework extracts multi-channel time- and frequency-domain physiological features across overlapping windows, trains a leakage-aware Random Forest classifier, reconstructs cohesive seizure episodes, classifies interval-based temporal relationships using formal **Allen Interval Algebra**, and generates deterministic, audit-traceable relationship findings.

---

## 2. Three Major Technical Contributions

1. **Window-Based EEG Analysis:**
   - Overlapping 2.0-second sliding windows with a 1.0-second step (50% overlap, 512 samples per window at 256 Hz).
   - 26 physiological EEG features per window spanning time-domain statistics and sub-band power spectral densities (Delta, Theta, Alpha, Beta, Gamma, spectral entropy, and slow-to-fast rhythm ratios).
   - Leakage-aware evaluation via Leave-One-Recording-Out (LORO) cross-validation that strictly isolates whole EDF files to prevent temporal data leakage across adjacent overlapping windows.

2. **Interval-Based Temporal Progression Reconstruction:**
   - Dynamic clustering of contiguous and overlapping positive prediction windows into unified, multi-second seizure episodes with mean confidence scoring.
   - Comprehensive temporal event representation combining model-predicted episodes and expert ground truth on recording-relative timelines.
   - Strict mathematical categorization under **Allen's 13 Interval Algebra relations** (Contains, During, Overlaps, Meets, Before, After, Equals, etc.) with floating-point tolerance ($\epsilon = 0.5$s).

3. **Evidence-Grounded Relationship Form Generation:**
   - Deterministic natural-language findings generator mapping structured temporal intervals directly to empirical timing evidence.
   - Traceability anchoring: every statement is linked to the source EDF file, recording-relative seconds, onset/offset latency, and shared duration without making unsubstantiated physiological causal claims.

---

## 3. Dataset Information

- **Database:** [CHB-MIT Scalp EEG Database v1.0.0 (PhysioNet)](https://physionet.org/content/chbmit/1.0.0/)
- **Subject:** `chb01` (Pediatric female subject at Boston Children's Hospital)
- **Sampling Frequency:** 256.0 Hz
- **Montage:** 23 bipolar scalp channels based on the International 10-20 electrode system
- **Target EDF Recordings:**
  - `chb01_03.edf` (Duration: 3,600.0s / 1.0 hour) — 1 clinical seizure: **2996.0s – 3036.0s** (40.0s duration)
  - `chb01_04.edf` (Duration: 3,600.0s / 1.0 hour) — 1 clinical seizure: **1467.0s – 1494.0s** (27.0s duration)
- **Annotations Source:** `chb01-summary.txt` (Parsed programmatically; no hardcoded timestamps)

---

## 4. System Architecture

```text
EEG_Temporal_Analysis/
│
├── app.py                     # Professional Streamlit research dashboard
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation
│
├── data/
│   └── chbmit/
│       └── chb01/
│           ├── chb01_03.edf          # 1-hour scalp EEG recording 1
│           ├── chb01_04.edf          # 1-hour scalp EEG recording 2
│           └── chb01-summary.txt     # Expert seizure annotations
│
├── models/
│   └── model.joblib           # Trained production pipeline (StandardScaler + RandomForest)
│
├── outputs/
│   ├── expert_annotations.csv        # Parsed ground-truth seizure intervals
│   ├── window_predictions.csv        # 7,198 window predictions & probabilities
│   ├── temporal_events.csv           # Reconstructed predicted & expert intervals
│   ├── relationships.csv             # Computed Allen interval relationships
│   ├── generated_sentences.csv       # Evidence-grounded natural language findings
│   ├── evaluation_metrics.json       # Out-of-fold performance metrics report
│   ├── confusion_matrix.png          # High-contrast confusion matrix heatmap
│   ├── roc_curve.png                 # Out-of-fold ROC curve (AUC = 0.9821)
│   └── graph.png                     # NetworkX evidence-linked directed graph
│
└── src/
    ├── __init__.py
    ├── data_loader.py         # MNE-based EDF loader & channel validator
    ├── annotation_parser.py   # chb01-summary.txt programmatic parser
    ├── preprocessing.py       # 0.5-45 Hz bandpass & 60 Hz notch filter
    ├── window_segmentation.py # 2.0s sliding window segmentation (50% overlap)
    ├── feature_extraction.py  # 26 time & spectral feature extractor
    ├── model_training.py      # LORO cross-validation & model training
    ├── window_prediction.py   # Window inference & confidence generator
    ├── temporal_engine.py     # Contiguous positive window clusterer
    ├── relationship_engine.py # Allen Interval Algebra engine
    ├── evidence_extractor.py  # Latency, duration & overlap metrics
    ├── sentence_generator.py  # Deterministic clinical sentence generator
    ├── graph_generator.py     # NetworkX directed graph visualizer
    ├── evaluation.py          # Metric calculations & publication plots
    └── pipeline.py            # Master end-to-end pipeline orchestrator
```

---

## 5. Installation & Setup

### Prerequisites
- Windows 10/11 with Python 3.11.x installed.
- PowerShell or Windows Terminal.

### Step 1: Clone or Navigate to the Workspace
```powershell
cd "c:\Users\musal\OneDrive\Desktop\Review 3"
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 6. How to Execute the Pipeline

Run the master scientific pipeline from the root directory:

```powershell
python -m src.pipeline
```

### Pipeline Execution Stages:
1. **[Stage 1/8]** Parses `chb01-summary.txt` and validates timestamps against EDF recording durations.
2. **[Stage 2/8]** Applies 0.5–45.0 Hz bandpass filtering, 60 Hz notch filtering, 2.0s window segmentation (3,599 windows per recording; 7,198 total), and extracts 26 physiological features.
3. **[Stage 3/8]** Conducts Leave-One-Recording-Out (LORO) cross-validation across recordings to evaluate generalization without window leakage; trains final pipeline to `models/model.joblib`.
4. **[Stage 4/8]** Predicts all 7,198 windows and saves `outputs/window_predictions.csv`.
5. **[Stage 5/8]** Groups consecutive positive predictions into multi-second seizure episodes in `outputs/temporal_events.csv`.
6. **[Stage 6/8]** Calculates Allen interval relations and records evidence in `outputs/relationships.csv`.
7. **[Stage 7/8]** Synthesizes auditable natural-language findings in `outputs/generated_sentences.csv`.
8. **[Stage 8/8]** Generates NetworkX directed relationship graph `outputs/graph.png` and evaluation figures.

---

## 7. How to Launch the Streamlit Dashboard

Launch the interactive research UI using:

```powershell
streamlit run app.py
```

The browser will open to `http://localhost:8501`.

### Dashboard Navigation:
- **TAB 1 — OVERVIEW:** Project metrics, high-level dataset statistics, system workflow architecture, and technical contribution cards.
- **TAB 2 — EEG ANALYSIS:** Recording selector, interactive Plotly raw vs. filtered waveform viewer, seizure probability time-series chart, and window prediction tables.
- **TAB 3 — TEMPORAL PROGRESSION:** Interactive Gantt-style interval timeline comparing expert annotations and model predictions in recording-relative seconds.
- **TAB 4 — RELATIONSHIP ANALYSIS:** NetworkX evidence-linked directed graph, Allen Interval Algebra reference guide, and interactive relationship inspector.
- **TAB 5 — GENERATED FORM:** Professional structured research findings report with CSV export capability.
- **TAB 6 — EVIDENCE & EVALUATION:** Out-of-fold confusion matrix, ROC curve, precision/recall/F1 metrics, and scientific limitation disclosures.

---

## 8. Explanation of Generated Output Deliverables

| Deliverable | Format | Purpose & Contents |
| :--- | :--- | :--- |
| `outputs/expert_annotations.csv` | CSV | Parsed clinical ground truth intervals (`recording_id`, `event_id`, `start_seconds`, `end_seconds`, `source`). |
| `outputs/window_predictions.csv` | CSV | Window-level inferences for all 7,198 windows with `predicted_label`, `predicted_probability`, and overlap metrics. |
| `outputs/temporal_events.csv` | CSV | Clustered seizure episodes and expert intervals with duration, confidence, and constituent window tracking. |
| `outputs/relationships.csv` | CSV | Formal Allen interval relationships (`CONTAINS`, `AFTER`, etc.) with measured overlap and gap latencies. |
| `outputs/generated_sentences.csv` | CSV | Deterministic clinical statements directly grounded in evidence references. |
| `outputs/evaluation_metrics.json` | JSON | Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrix, and methodology disclosures. |
| `outputs/confusion_matrix.png` | PNG | High-resolution publication-quality heatmap of out-of-fold test windows. |
| `outputs/roc_curve.png` | PNG | Receiver Operating Characteristic curve displaying area under curve (AUC = 0.9821). |
| `outputs/graph.png` | PNG | NetworkX directed relationship graph with distinct styles for expert and predicted intervals. |
| `models/model.joblib` | Binary | Serialized Scikit-Learn `Pipeline(StandardScaler, RandomForestClassifier)`. |

---

## 9. Evaluation Methodology & Scientific Rigor

### Leakage-Aware Validation
Adjacent overlapping windows share 1.0 second of continuous signal. A standard random train-test split places adjacent windows into both training and testing sets, resulting in severe data leakage and artificially inflated metrics.

To eliminate this flaw, this system implements **Leave-One-Recording-Out (LORO)** cross-validation:
- **Fold 1:** Train on `chb01_03` (40s seizure, non-seizure windows), Test on `chb01_04`.
- **Fold 2:** Train on `chb01_04` (27s seizure, non-seizure windows), Test on `chb01_03`.
- Out-of-fold predictions are collected across all 7,198 windows:
  - **Accuracy:** `99.58%`
  - **Precision:** `78.26%`
  - **Recall (Sensitivity):** `78.26%`
  - **F1-Score:** `78.26%`
  - **ROC-AUC:** `0.9821`

---

## 10. Known Limitations

1. **Single-Subject Scope:** The recordings belong to subject `chb01`. The evaluation confirms temporal generalization across distinct seizure events within the same patient, but cannot claim patient-independent generalization without multi-subject testing.
2. **Temporal Association vs. Causation:** Computed interval relations (e.g. `PRED_EP01 CONTAINS EXP_SZ01`, `PRED_EP02 AFTER EXP_SZ01`) model temporal order and overlap; they do not establish physiological causation.
3. **Investigative Tooling:** Model outputs represent algorithmic hypotheses intended to assist clinical research, not automated medical diagnoses.

---

## 11. Future Enhancements

- Multi-patient cross-validation across all 24 CHB-MIT subjects.
- Deep spatio-temporal architectures (CNN-LSTM / Temporal Convolutional Networks).
- Real-time streaming integration with bedside EEG telemetry.
- Automated electrographic vs. clinical onset latency differentiation.
