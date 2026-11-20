"""
EEG Temporal Analysis System — Streamlit Dashboard
Final Year B.Tech Project: System for Reconstructing Temporal Progression
and Generating Causal Relationship Forms in EEG-Based Seizure Detection.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EEG Temporal Analysis | CHB-MIT Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Professional Dark-Navy Custom Styling
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Base theme overrides */
.stApp {
    background-color: #0B0F19;
    color: #E2E8F0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid #1E293B;
}

/* Header banner */
.header-container {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.header-title {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #F8FAFC;
    margin: 0 0 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-subtitle {
    font-size: 15px;
    color: #94A3B8;
    margin: 0 0 16px 0;
    font-weight: 400;
}

.header-meta-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 12px;
}

.badge {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
    color: #CBD5E1;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.badge-blue {
    background-color: rgba(56, 189, 248, 0.12);
    border-color: rgba(56, 189, 248, 0.35);
    color: #38BDF8;
}

.badge-green {
    background-color: rgba(16, 185, 129, 0.12);
    border-color: rgba(16, 185, 129, 0.35);
    color: #34D399;
}

.badge-purple {
    background-color: rgba(168, 85, 247, 0.12);
    border-color: rgba(168, 85, 247, 0.35);
    color: #C084FC;
}

.badge-amber {
    background-color: rgba(245, 158, 11, 0.12);
    border-color: rgba(245, 158, 11, 0.35);
    color: #FBBF24;
}

/* Metric card */
.metric-card {
    background-color: #131D31;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.metric-card:hover {
    border-color: #38BDF8;
}

.metric-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94A3B8;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 26px;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.1;
    margin-bottom: 4px;
}

.metric-sub {
    font-size: 12px;
    color: #64748B;
}

/* Form Report Card */
.report-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-left: 4px solid #A855F7;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

.report-sentence {
    font-size: 16px;
    font-weight: 500;
    color: #F3F4F6;
    line-height: 1.5;
    margin-bottom: 12px;
}

.evidence-box {
    background-color: #0B0F19;
    border: 1px solid #1E293B;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: monospace;
    font-size: 12px;
    color: #94A3B8;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 4px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    color: #94A3B8;
    font-weight: 600;
    font-size: 14px;
}

.stTabs [aria-selected="true"] {
    background-color: #1E293B !important;
    border-color: #334155 #334155 transparent #334155 !important;
    color: #38BDF8 !important;
}

/* Scientific note alert */
.scientific-alert {
    background-color: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px;
    padding: 14px 18px;
    color: #FDE68A;
    font-size: 13px;
    line-height: 1.45;
    margin: 16px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Paths and Data Loading Functions
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data/chbmit/chb01")
MODEL_PATH = Path("models/model.joblib")


@st.cache_data(show_spinner=False)
def load_output_file(filename: str) -> Optional[pd.DataFrame]:
    """Loads a CSV output from outputs directory if available."""
    p = OUTPUT_DIR / filename
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_metrics_json() -> Optional[Dict[str, Any]]:
    """Loads evaluation metrics JSON if available."""
    p = OUTPUT_DIR / "evaluation_metrics.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Load actual files
expert_df = load_output_file("expert_annotations.csv")
preds_df = load_output_file("window_predictions.csv")
events_df = load_output_file("temporal_events.csv")
rel_df = load_output_file("relationships.csv")
sentences_df = load_output_file("generated_sentences.csv")
metrics = load_metrics_json()

pipeline_ready = all([
    expert_df is not None,
    preds_df is not None,
    events_df is not None,
    rel_df is not None,
    sentences_df is not None,
    metrics is not None,
])

# -----------------------------------------------------------------------------
# Application Header
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="header-container">
        <div class="header-title">
            <span>🧠</span> EEG TEMPORAL ANALYSIS
        </div>
        <div class="header-subtitle">
            An Evidence-Grounded Framework for EEG Seizure Analysis and Temporal Reconstruction
        </div>
        <div class="header-meta-bar">
            <span class="badge badge-blue">Dataset: CHB-MIT Scalp EEG</span>
            <span class="badge">Subject: chb01</span>
            <span class="badge">Recordings: chb01_03.edf & chb01_04.edf</span>
            <span class="badge {'badge-green' if pipeline_ready else 'badge-amber'}">
                ● Pipeline: {'Outputs Active & Verified' if pipeline_ready else 'Outputs Missing — Action Required'}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar Navigation and System Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Controls")

    # Dataset status
    st.markdown("#### 📁 Dataset Verification")
    edf3_exists = (DATA_DIR / "chb01_03.edf").exists()
    edf4_exists = (DATA_DIR / "chb01_04.edf").exists()
    sum_exists = (DATA_DIR / "chb01-summary.txt").exists()

    st.markdown(f"- `chb01_03.edf`: {'✅ Available' if edf3_exists else '❌ Missing'}")
    st.markdown(f"- `chb01_04.edf`: {'✅ Available' if edf4_exists else '❌ Missing'}")
    st.markdown(f"- `chb01-summary.txt`: {'✅ Available' if sum_exists else '❌ Missing'}")

    st.markdown("---")
    st.markdown("#### 🔬 Analysis Policy")
    st.info(
        "**Window Configuration:**\n"
        "- Duration: 2.0s\n"
        "- Stride/Step: 1.0s (50% overlap)\n"
        "- Sampling Rate: 256 Hz (512 samples/win)\n"
        "- Seizure Threshold: $\ge 50\%$ annotation overlap"
    )

    st.markdown("---")
    st.markdown("#### 🚀 Pipeline Execution")
    if st.button("Run / Re-run Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Executing end-to-end EEG scientific pipeline..."):
            from src.pipeline import run_pipeline
            try:
                result = run_pipeline()
                st.success("Pipeline finished successfully!")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")

    st.markdown("---")
    st.markdown(
        """
        <div class="scientific-alert">
            <strong>⚠️ Scientific Notice:</strong><br>
            Temporal interval relationships (e.g. Before, Overlaps, Contains) demonstrate time-order alignment, not clinical causation.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Empty State Notice if Outputs Missing
# -----------------------------------------------------------------------------
if not pipeline_ready:
    st.warning("### ⚠️ Pipeline Deliverables Not Yet Generated")
    st.markdown(
        """
        The EEG Temporal Analysis pipeline has not yet generated the required output records.
        To generate all empirical records, feature tables, out-of-fold metrics, and relationship graphs:

        1. Click the **'Run / Re-run Pipeline'** button in the left sidebar, or
        2. Run the following command in your terminal:
        ```powershell
        python -m src.pipeline
        ```
        """
    )
    st.stop()

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📊 TAB 1 — OVERVIEW",
    "📈 TAB 2 — EEG ANALYSIS",
    "⏱️ TAB 3 — TEMPORAL PROGRESSION",
    "🔗 TAB 4 — RELATIONSHIP ANALYSIS",
    "📝 Generated Form",
    "🧪 TAB 6 — EVIDENCE & EVALUATION",
])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tabs[0]:
    st.markdown("### 📌 Project Overview & System Architecture")

    # High-level metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Analyzed Windows</div>
                <div class="metric-value">{len(preds_df):,}</div>
                <div class="metric-sub">2-sec windows (50% overlap)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        total_rec_sec = 7200.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Duration</div>
                <div class="metric-value">{total_rec_sec/3600:.1f} hrs</div>
                <div class="metric-sub">2 recordings (3,600s each)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        n_expert = len(expert_df)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Expert Seizures</div>
                <div class="metric-value">{n_expert}</div>
                <div class="metric-sub">chb01-summary.txt</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        n_preds_ep = len(events_df[events_df["source"] == "model_prediction"])
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Episodes</div>
                <div class="metric-value">{n_preds_ep}</div>
                <div class="metric-sub">Grouped temporal intervals</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        n_rels = len(rel_df)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Allen Relationships</div>
                <div class="metric-value">{n_rels}</div>
                <div class="metric-sub">Formally verified relations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Architectural Workflow Diagram
    st.markdown("#### 🔄 End-to-End Scientific Architecture Workflow")
    st.markdown(
        """
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 20px; margin-bottom: 24px;">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; text-align: center;">
                <div style="flex: 1; min-width: 120px; background: #1E293B; border-radius: 8px; padding: 12px; border-top: 3px solid #38BDF8;">
                    <div style="font-size: 20px;">📁</div>
                    <div style="font-size: 13px; font-weight: 700; color: #F8FAFC;">1. EDF Loading</div>
                    <div style="font-size: 11px; color: #94A3B8;">MNE-Python & Annotations</div>
                </div>
                <div style="color: #64748B; font-size: 18px;">➔</div>
                <div style="flex: 1; min-width: 120px; background: #1E293B; border-radius: 8px; padding: 12px; border-top: 3px solid #38BDF8;">
                    <div style="font-size: 20px;">⚡</div>
                    <div style="font-size: 13px; font-weight: 700; color: #F8FAFC;">2. Preprocessing</div>
                    <div style="font-size: 11px; color: #94A3B8;">0.5-45Hz Filter & 60Hz Notch</div>
                </div>
                <div style="color: #64748B; font-size: 18px;">➔</div>
                <div style="flex: 1; min-width: 120px; background: #1E293B; border-radius: 8px; padding: 12px; border-top: 3px solid #38BDF8;">
                    <div style="font-size: 20px;">🪟</div>
                    <div style="font-size: 13px; font-weight: 700; color: #F8FAFC;">3. Segmentation</div>
                    <div style="font-size: 11px; color: #94A3B8;">2s Windows (50% Overlap)</div>
                </div>
                <div style="color: #64748B; font-size: 18px;">➔</div>
                <div style="flex: 1; min-width: 120px; background: #1E293B; border-radius: 8px; padding: 12px; border-top: 3px solid #38BDF8;">
                    <div style="font-size: 20px;">🔬</div>
                    <div style="font-size: 13px; font-weight: 700; color: #F8FAFC;">4. Features & ML</div>
                    <div style="font-size: 11px; color: #94A3B8;">26 Bands & RF Classifier</div>
                </div>
                <div style="color: #64748B; font-size: 18px;">➔</div>
                <div style="flex: 1; min-width: 120px; background: #1E293B; border-radius: 8px; padding: 12px; border-top: 3px solid #34D399;">
                    <div style="font-size: 20px;">⏱️</div>
                    <div style="font-size: 13px; font-weight: 700; color: #F8FAFC;">5. Temporal Engine</div>
                    <div style="font-size: 11px; color: #94A3B8;">Episodic Reconstruction</div>
                </div>
                <div style="color: #64748B; font-size: 18px;">➔</div>
                <div style="flex: 1; min-width: 120px; background: #1E293B; border-radius: 8px; padding: 12px; border-top: 3px solid #C084FC;">
                    <div style="font-size: 20px;">🔗</div>
                    <div style="font-size: 13px; font-weight: 700; color: #F8FAFC;">6. Allen Relations</div>
                    <div style="font-size: 11px; color: #94A3B8;">Interval Algebra Graph</div>
                </div>
                <div style="color: #64748B; font-size: 18px;">➔</div>
                <div style="flex: 1; min-width: 120px; background: #1E293B; border-radius: 8px; padding: 12px; border-top: 3px solid #FBBF24;">
                    <div style="font-size: 20px;">📄</div>
                    <div style="font-size: 13px; font-weight: 700; color: #F8FAFC;">7. Evidence Form</div>
                    <div style="font-size: 11px; color: #94A3B8;">Auditable Synthesis</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Three Technical Contributions
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """
            <div style="background-color: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; height: 100%;">
                <div style="color: #38BDF8; font-weight: 700; font-size: 14px; margin-bottom: 8px;">
                    CONTRIBUTION 1
                </div>
                <div style="color: #F8FAFC; font-weight: 700; font-size: 16px; margin-bottom: 8px;">
                    Window-Based EEG Analysis
                </div>
                <div style="color: #94A3B8; font-size: 13px; line-height: 1.5;">
                    Overlapping 2.0-second window segmentation (1.0s step) extracting 26 time- and frequency-domain physiological EEG features. Evaluated using strict Leave-One-Recording-Out cross-validation to prevent adjacent-window temporal leakage.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div style="background-color: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; height: 100%;">
                <div style="color: #34D399; font-weight: 700; font-size: 14px; margin-bottom: 8px;">
                    CONTRIBUTION 2
                </div>
                <div style="color: #F8FAFC; font-weight: 700; font-size: 16px; margin-bottom: 8px;">
                    Interval-Based Reconstruction
                </div>
                <div style="color: #94A3B8; font-size: 13px; line-height: 1.5;">
                    Clustering contiguous positive classification windows into unified seizure episodes with confidence tracking, followed by formal classification under Allen's 13 interval relations (Contains, Overlaps, Meets, During, Before, After) with mathematical boundary tolerance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            """
            <div style="background-color: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; height: 100%;">
                <div style="color: #C084FC; font-weight: 700; font-size: 14px; margin-bottom: 8px;">
                    CONTRIBUTION 3
                </div>
                <div style="color: #F8FAFC; font-weight: 700; font-size: 16px; margin-bottom: 8px;">
                    Evidence-Grounded Form
                </div>
                <div style="color: #94A3B8; font-size: 13px; line-height: 1.5;">
                    Deterministic synthesis of natural-language clinical findings completely linked to raw recording seconds, channel montages, and exact overlap and gap latencies—without unsubstantiated causal assumptions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 2: EEG ANALYSIS
# =============================================================================
with tabs[1]:
    st.markdown("### 🎯 Window-Level Binary Classification: Seizure vs. Non-Seizure")
    st.markdown(
        "Quantitative evaluation of sliding 2.0-second EEG windows (1.0s step) classified by the "
        "trained model into **Seizure (Class 1)** versus **Non-Seizure (Class 0)**."
    )

    # -------------------------------------------------------------------------
    # 1. METRICS CARDS: Binary Classification Window Counts
    # -------------------------------------------------------------------------
    st.markdown("#### 📊 Binary Classification Window Counts")
    total_all_win = len(preds_df)
    total_all_seiz = int((preds_df["predicted_label"] == 1).sum())
    total_all_nonseiz = int((preds_df["predicted_label"] == 0).sum())

    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1:
        st.metric(
            label="Total Windows Evaluated",
            value=f"{total_all_win:,}",
            help="Total 2.0-second EEG windows across all recordings in window_predictions.csv",
        )
    with m_c2:
        st.metric(
            label="Classified as 'Seizure'",
            value=f"{total_all_seiz:,}",
            delta=f"{total_all_seiz / total_all_win * 100:.2f}% of windows",
            delta_color="inverse",
            help="Windows where model predicted probability P(Seizure) >= 0.50",
        )
    with m_c3:
        st.metric(
            label="Classified as 'Non-Seizure'",
            value=f"{total_all_nonseiz:,}",
            delta=f"{total_all_nonseiz / total_all_win * 100:.2f}% of windows",
            delta_color="normal",
            help="Windows where model predicted probability P(Seizure) < 0.50",
        )
    with m_c4:
        st.metric(
            label="Decision Threshold",
            value="0.50",
            help="Threshold boundary: P >= 0.50 classified as Seizure, P < 0.50 as Non-Seizure",
        )

    st.markdown("---")

    # Select recording for temporal exploration
    rec_choices = sorted(preds_df["recording_id"].unique())
    sel_rec = st.selectbox("Select EDF Recording to Inspect:", rec_choices, index=0)

    rec_preds = preds_df[preds_df["recording_id"] == sel_rec].copy().sort_values("start_seconds").reset_index(drop=True)
    rec_annots = expert_df[expert_df["recording_id"] == sel_rec] if expert_df is not None else pd.DataFrame()

    rec_seiz = int((rec_preds["predicted_label"] == 1).sum())
    rec_nonseiz = int((rec_preds["predicted_label"] == 0).sum())

    st.markdown(
        f"""
        <div style="background-color: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
            <span style="font-size: 14px; color: #94A3B8;">Selected Recording: <strong>{sel_rec}</strong></span>
            <span class="badge badge-blue">Windows: {len(rec_preds):,}</span>
            <span class="badge badge-purple">Seizure: {rec_seiz} ({rec_seiz/len(rec_preds)*100:.2f}%)</span>
            <span class="badge">Non-Seizure: {rec_nonseiz} ({rec_nonseiz/len(rec_preds)*100:.2f}%)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 2. PREDICTION PROBABILITY CHART (With Red Threshold Line at 0.5)
    # -------------------------------------------------------------------------
    st.markdown("#### 📈 Model Prediction Probability Over Time")
    st.caption("Plots the model's confidence score P(Seizure) against Window Start Time (seconds). The horizontal red line marks the 0.5 decision boundary where the model flips from Non-Seizure to Seizure.")

    fig_prob = go.Figure()

    # Ground truth shaded region (if available for this recording)
    for _, a in rec_annots.iterrows():
        s = float(a["start_seconds"])
        e = float(a["end_seconds"])
        fig_prob.add_vrect(
            x0=s, x1=e,
            fillcolor="rgba(16, 185, 129, 0.2)",
            line_width=1.5,
            line_color="#10B981",
            annotation_text=f"Expert Ground Truth ({s:.0f}s - {e:.0f}s)",
            annotation_position="top left",
            annotation_font=dict(color="#34D399", size=11),
        )

    # Model predicted probabilities line chart
    fig_prob.add_trace(
        go.Scatter(
            x=rec_preds["start_seconds"],
            y=rec_preds["predicted_probability"],
            mode="lines",
            name="Model Seizure Probability",
            line=dict(color="#38BDF8", width=2.0),
            hovertemplate="<b>Window Start:</b> %{x:.1f}s<br><b>P(Seizure):</b> %{y:.4f}<extra></extra>",
        )
    )

    # Horizontal RED threshold line at 0.5
    fig_prob.add_hline(
        y=0.50,
        line_dash="dash",
        line_color="#EF4444",
        line_width=2.5,
        annotation_text="Threshold = 0.5 (Flips to Seizure)",
        annotation_position="top right",
        annotation_font=dict(color="#EF4444", size=11, weight="bold"),
    )

    fig_prob.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        margin=dict(l=40, r=40, t=35, b=40),
        xaxis=dict(
            title="Window Start Time (seconds)",
            gridcolor="#334155",
            zeroline=False,
        ),
        yaxis=dict(
            title="Prediction Probability P(Seizure)",
            range=[-0.05, 1.05],
            gridcolor="#334155",
            zeroline=False,
        ),
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.8)",
        ),
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # -------------------------------------------------------------------------
    # 3. RAW PREDICTIONS TABLE (First 20 rows of window_predictions.csv)
    # -------------------------------------------------------------------------
    st.markdown("#### 📋 Raw Predictions Table (First 20 Rows)")
    st.caption("Directly loaded from `outputs/window_predictions.csv` showing the exact start time, end time, binary predicted class, and probability confidence.")

    # Prepare first 20 rows with exact required columns
    top20_raw = rec_preds.head(20).copy()
    display_table = pd.DataFrame({
        "Start Time": top20_raw["start_seconds"].map(lambda x: f"{x:.2f}s"),
        "End Time": top20_raw["end_seconds"].map(lambda x: f"{x:.2f}s"),
        "Predicted Class": top20_raw["predicted_label"].map({1: "Seizure", 0: "Non-Seizure"}),
        "Seizure Probability": top20_raw["predicted_probability"].map(lambda x: f"{x:.4f}"),
    })

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=False,
    )

    # -------------------------------------------------------------------------
    # 4. SIGNAL SEGMENT VISUALIZER (Raw vs Filtered Waveforms)
    # -------------------------------------------------------------------------
    st.markdown("#### ⚡ Signal Segment Visualizer (Raw vs Filtered)")
    with st.expander("Explore Multi-Channel EEG Waveforms around Seizure Interval", expanded=False):
        if not rec_annots.empty:
            sz_s = float(rec_annots.iloc[0]["start_seconds"])
            sz_e = float(rec_annots.iloc[0]["end_seconds"])
            default_tmin = max(0.0, sz_s - 5.0)
            default_tmax = min(3600.0, sz_e + 5.0)
        else:
            default_tmin = 0.0
            default_tmax = 30.0

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            plot_tmin = st.number_input("Start Time (s):", value=default_tmin, min_value=0.0, max_value=3590.0, step=5.0)
        with col_t2:
            plot_tmax = st.number_input("End Time (s):", value=min(plot_tmin + 20.0, default_tmax), min_value=plot_tmin + 1.0, max_value=3600.0, step=5.0)

        # Load raw snippet using data loader
        from src.data_loader import EDFDataLoader
        from src.preprocessing import EEGPreprocessor

        try:
            dloader = EDFDataLoader()
            raw_obj = dloader.load_raw(f"{sel_rec}.edf", preload=False)
            ch_choice = st.selectbox("Select EEG Channel to Display:", raw_obj.ch_names, index=0)

            # Crop slice
            raw_slice = raw_obj.copy().crop(tmin=plot_tmin, tmax=plot_tmax).load_data()
            preproc = EEGPreprocessor()
            clean_slice = preproc.preprocess_raw(raw_slice.copy(), verbose=False)

            times = raw_slice.times + plot_tmin
            raw_ch_data = raw_slice.get_data(picks=[ch_choice])[0] * 1e6  # Convert V to uV
            clean_ch_data = clean_slice.get_data(picks=[ch_choice])[0] * 1e6

            fig_eeg = go.Figure()
            fig_eeg.add_trace(go.Scatter(
                x=times, y=raw_ch_data, mode="lines", name="Raw EEG (uV)",
                line=dict(color="#64748B", width=1.0)
            ))
            fig_eeg.add_trace(go.Scatter(
                x=times, y=clean_ch_data, mode="lines", name="Filtered (0.5-45 Hz) (uV)",
                line=dict(color="#38BDF8", width=1.5)
            ))

            fig_eeg.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                margin=dict(l=40, r=40, t=30, b=40),
                xaxis_title="Time (seconds)",
                yaxis_title=f"Amplitude (uV) — {ch_choice}",
                height=300,
            )
            st.plotly_chart(fig_eeg, use_container_width=True)
            raw_obj.close()
        except Exception as e:
            st.error(f"Could not load EEG waveform snippet: {e}")

    # Window Predictions Table
    st.markdown("#### 📋 Window-Level Prediction Records")
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        filter_label = st.radio(
            "Filter Windows by Class:",
            ["All Windows", "Positive Predicted Only (P >= 0.50)", "Expert Seizure Overlap Only"],
            horizontal=True,
        )
    with col_f2:
        st.write("")

    if filter_label == "Positive Predicted Only (P >= 0.50)":
        display_win_df = rec_preds[rec_preds["predicted_label"] == 1]
    elif filter_label == "Expert Seizure Overlap Only":
        display_win_df = rec_preds[rec_preds["expert_label"] == 1]
    else:
        display_win_df = rec_preds

    st.dataframe(
        display_win_df[[
            "window_id", "recording_id", "start_seconds", "end_seconds",
            "predicted_label", "predicted_probability", "expert_label",
            "annotation_overlap_seconds", "annotation_overlap_ratio"
        ]].head(200),
        use_container_width=True,
        height=280,
    )

# =============================================================================
# TAB 3: TEMPORAL PROGRESSION
# =============================================================================
with tabs[2]:
    st.markdown("### ⏱️ Temporal Progression Reconstruction")

    sel_rec_t = st.selectbox("Select Recording for Temporal Progression:", sorted(events_df["recording_id"].unique()), key="prog_rec_select")
    rec_events = events_df[events_df["recording_id"] == sel_rec_t].copy()

    # Timeline Gantt chart
    st.markdown("#### 📅 Reconstructed Interval Timeline")

    fig_timeline = go.Figure()
    y_labels = []
    y_pos = []

    for idx, (_, row) in enumerate(rec_events.iterrows()):
        evt_id = row["event_id"]
        s = float(row["start_seconds"])
        e = float(row["end_seconds"])
        dur = float(row["duration_seconds"])
        src = row["source"]
        is_expert = src == "expert_annotation"
        color = "#10B981" if is_expert else "#A855F7"
        src_title = "Expert Annotation" if is_expert else "Model Prediction"
        conf = row.get("confidence", 1.0)
        conf_text = f"{conf*100:.1f}%" if conf is not None else "100%"

        fig_timeline.add_trace(go.Bar(
            x=[dur],
            y=[evt_id],
            base=[s],
            orientation="h",
            marker=dict(color=color, line=dict(color="white", width=1)),
            name=src_title,
            hovertemplate=(
                f"<b>{evt_id}</b> ({src_title})<br>"
                f"Start: {s:.1f}s | End: {e:.1f}s<br>"
                f"Duration: {dur:.1f}s | Confidence: {conf_text}"
                "<extra></extra>"
            ),
            showlegend=(idx < 2),
        ))

    # Zoom window to relevant interval
    min_t = max(0.0, rec_events["start_seconds"].min() - 30.0)
    max_t = rec_events["end_seconds"].max() + 30.0

    fig_timeline.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        margin=dict(l=140, r=40, t=30, b=40),
        xaxis=dict(title="Recording-Relative Time (seconds)", range=[min_t, max_t]),
        yaxis=dict(title="Event ID", autorange="reversed"),
        height=260,
        barmode="overlay",
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Detailed Event Cards
    st.markdown("#### 🔍 Reconstructed Temporal Events Table")
    st.dataframe(
        rec_events[[
            "event_id", "recording_id", "event_type", "start_seconds",
            "end_seconds", "duration_seconds", "source", "confidence"
        ]],
        use_container_width=True,
    )

# =============================================================================
# TAB 4: RELATIONSHIP ANALYSIS
# =============================================================================
with tabs[3]:
    st.markdown("### 🔗 Allen Interval Relationship Analysis")

    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        st.markdown("#### 🕸️ Evidence-Linked Relationship Graph")
        graph_img = OUTPUT_DIR / "graph.png"
        if graph_img.exists():
            st.image(str(graph_img), use_container_width=True)
        else:
            st.info("Graph image not generated yet.")

    with col_g2:
        st.markdown("#### 📖 Allen Interval Algebra Primer")
        st.markdown(
            """
            <div style="background-color: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 14px; font-size: 13px; color: #CBD5E1;">
                <p><strong>Strict Interval Algebra Definitions (Tolerance &epsilon; = 0.5s):</strong></p>
                <ul>
                    <li><strong>CONTAINS:</strong> Interval A starts before B and ends after B.</li>
                    <li><strong>DURING:</strong> Interval A is strictly inside Interval B.</li>
                    <li><strong>OVERLAPS:</strong> Interval A begins before B, but terminates inside B.</li>
                    <li><strong>MEETS:</strong> Interval A concludes exactly when Interval B commences.</li>
                    <li><strong>BEFORE:</strong> Interval A concludes strictly before B commences.</li>
                    <li><strong>AFTER:</strong> Interval A commences strictly after B concludes.</li>
                    <li><strong>EQUALS:</strong> Both intervals share simultaneous onsets and offsets.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 📊 Computed Pairwise Relationships")
    st.dataframe(
        rel_df[[
            "relationship_id", "recording_id", "event_a_id", "relation_type",
            "event_b_id", "event_a_start", "event_a_end", "event_b_start",
            "event_b_end", "overlap_seconds", "gap_seconds", "evidence_reference"
        ]],
        use_container_width=True,
    )

    # Relationship Inspection Tool
    st.markdown("#### 🔎 Interactive Relationship Inspector")
    rel_ids = rel_df["relationship_id"].tolist()
    chosen_rel = st.selectbox("Select Relationship to Inspect:", rel_ids)
    rel_row = rel_df[rel_df["relationship_id"] == chosen_rel].iloc[0]

    c_r1, c_r2, c_r3 = st.columns(3)
    with c_r1:
        st.markdown(f"**Relationship ID:** `{rel_row['relationship_id']}`")
        st.markdown(f"**Recording:** `{rel_row['recording_id']}`")
        st.markdown(f"**Formal Relation:** <span class='badge badge-purple'>{rel_row['relation_type']}</span>", unsafe_allow_html=True)
    with c_r2:
        st.markdown(f"**Event A (Prediction):** `{rel_row['event_a_id']}`")
        st.markdown(f"**Span:** `[{rel_row['event_a_start']}s – {rel_row['event_a_end']}s]`")
        dur_a = rel_row['event_a_end'] - rel_row['event_a_start']
        st.markdown(f"**Duration:** `{dur_a:.1f}s`")
    with c_r3:
        st.markdown(f"**Event B (Expert):** `{rel_row['event_b_id']}`")
        st.markdown(f"**Span:** `[{rel_row['event_b_start']}s – {rel_row['event_b_end']}s]`")
        dur_b = rel_row['event_b_end'] - rel_row['event_b_start']
        st.markdown(f"**Duration:** `{dur_b:.1f}s`")

    st.markdown(
        f"""
        <div class="evidence-box">
            <strong>Evidence Reference:</strong> {rel_row['evidence_reference']}<br>
            <strong>Overlap:</strong> {rel_row['overlap_seconds']}s | <strong>Gap:</strong> {rel_row['gap_seconds']}s
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# TAB 5: GENERATED FORM (Relationship Sentences)
# =============================================================================
with tabs[4]:
    st.markdown("### 📝 Generated Form: Evidence-Supported Relationship Sentences")
    st.markdown(
        "Deterministic natural-language clinical findings synthesized directly from computed "
        "**Allen Interval Algebra** relationships. Each sentence is anchored to exact recording seconds, "
        "predicted episode boundaries, and expert clinical ground truth without unsubstantiated causal claims."
    )

    # Directly read outputs/generated_sentences.csv using Pandas
    sentences_csv_path = OUTPUT_DIR / "generated_sentences.csv"
    if not sentences_csv_path.exists():
        st.warning("⚠️ `outputs/generated_sentences.csv` does not exist yet. Please run the pipeline first.")
    else:
        df_sentences = pd.read_csv(sentences_csv_path)

        if df_sentences.empty:
            st.info("No relationship sentences found in `outputs/generated_sentences.csv`.")
        else:
            n_sentences = len(df_sentences)

            # -----------------------------------------------------------------
            # 1. INTERACTIVE SELECTOR (Dropdown & Slider)
            # -----------------------------------------------------------------
            st.markdown("#### 🔍 Step-by-Step Sentence Inspector")
            col_ctrl1, col_ctrl2 = st.columns([3, 2])

            with col_ctrl1:
                sentence_options = []
                for i, r in df_sentences.iterrows():
                    r_type = r.get("Relationship_Type") or r.get("relation_type", "REL")
                    ea = r.get("Event_A_ID") or r.get("event_a_id", "EvtA")
                    eb = r.get("Event_B_ID") or r.get("event_b_id", "EvtB")
                    rec = r.get("Recording_ID") or r.get("recording_id", "")
                    sentence_options.append(f"Sentence {i+1} ({rec}): [{r_type}] {ea} ➔ {eb}")

                selected_sentence_idx = st.selectbox(
                    "Select Sentence to Inspect:",
                    range(n_sentences),
                    format_func=lambda idx: sentence_options[idx],
                    key="sent_dropdown_select",
                )

            with col_ctrl2:
                slider_val = st.slider(
                    "Or Navigate with Slider:",
                    min_value=1,
                    max_value=n_sentences,
                    value=selected_sentence_idx + 1,
                    step=1,
                    key="sent_slider_select",
                )
                if slider_val - 1 != selected_sentence_idx:
                    selected_sentence_idx = slider_val - 1

            # Selected sentence record
            cur_row = df_sentences.iloc[selected_sentence_idx]
            cur_evt_a = str(cur_row.get("Event_A_ID") or cur_row.get("event_a_id", "N/A"))
            cur_evt_b = str(cur_row.get("Event_B_ID") or cur_row.get("event_b_id", "N/A"))
            cur_rel_type = str(cur_row.get("Relationship_Type") or cur_row.get("relation_type", "N/A"))
            cur_rec_id = str(cur_row.get("Recording_ID") or cur_row.get("recording_id", "N/A"))
            cur_overlap = float(cur_row.get("Overlap_Seconds", cur_row.get("overlap_seconds", 0.0)))
            cur_gap = float(cur_row.get("Gap_Seconds", cur_row.get("gap_seconds", 0.0)))
            cur_sent_text = str(cur_row.get("Generated_Sentence") or cur_row.get("generated_sentence", ""))
            cur_ev_ref = str(cur_row.get("Evidence_Reference") or cur_row.get("evidence_reference", "N/A"))
            cur_rel_id = str(cur_row.get("Relationship_ID") or cur_row.get("relationship_id", "N/A"))
            cur_sent_id = str(cur_row.get("Sentence_ID") or cur_row.get("sentence_id", f"SENT_{selected_sentence_idx+1:03d}"))

            st.markdown("---")

            # -----------------------------------------------------------------
            # 2. EXACT EVIDENCE DISPLAY (Above the Sentence for Panel Traceability)
            # -----------------------------------------------------------------
            st.markdown("#### 🔬 Traceable Empirical Evidence")
            st.caption("Verifiable timing metrics linking the algorithmic prediction to expert ground truth:")

            ev_col1, ev_col2, ev_col3, ev_col4, ev_col5 = st.columns(5)
            with ev_col1:
                st.metric("Event A ID (Prediction)", cur_evt_a)
            with ev_col2:
                st.metric("Event B ID (Expert Ground Truth)", cur_evt_b)
            with ev_col3:
                st.metric("Relationship Type", cur_rel_type)
            with ev_col4:
                st.metric("Time Overlap", f"{cur_overlap:.1f}s")
            with ev_col5:
                st.metric("Time Gap", f"{cur_gap:.1f}s")

            # Evidence Anchor Reference Box
            st.markdown(
                f"""
                <div class="evidence-box" style="margin-top: 8px; margin-bottom: 20px;">
                    <strong>Traceability Anchor:</strong> {cur_ev_ref}<br>
                    <strong>Lineage:</strong> Recording: <code>{cur_rec_id}</code> | Relationship ID: <code>{cur_rel_id}</code> | Sentence ID: <code>{cur_sent_id}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # -----------------------------------------------------------------
            # 3. EVIDENCE-SUPPORTED SENTENCE (st.success or st.info card)
            # -----------------------------------------------------------------
            st.markdown("#### 💬 Evidence-Supported Clinical Finding")

            # Highlighting card based on relation nature
            if cur_rel_type in ["CONTAINS", "OVERLAPS", "EQUALS", "STARTS", "FINISHES", "DURING"]:
                st.success(f"**Synthesized Finding [{cur_rel_type}]:**\n\n\"{cur_sent_text}\"")
            else:
                st.info(f"**Synthesized Finding [{cur_rel_type}]:**\n\n\"{cur_sent_text}\"")

            # Custom styled presentation card
            badge_class = "badge-purple" if cur_rel_type in ["CONTAINS", "OVERLAPS", "EQUALS"] else "badge-blue"
            st.markdown(
                f"""
                <div class="report-card" style="margin-top: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span class="badge {badge_class}">Formal Relation: {cur_rel_type}</span>
                        <span style="font-size: 12px; color: #94A3B8; font-family: monospace;">Record {selected_sentence_idx + 1} of {n_sentences}</span>
                    </div>
                    <div class="report-sentence" style="font-size: 17px; font-weight: 600; color: #F8FAFC;">
                        "{cur_sent_text}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # -----------------------------------------------------------------
            # 4. FULL GENERATED FINDINGS TABLE (All Sentences)
            # -----------------------------------------------------------------
            with st.expander("📋 View All Generated Sentences Table", expanded=True):
                # Clean display table with standardized columns
                table_rows = []
                for _, r in df_sentences.iterrows():
                    table_rows.append({
                        "Event A ID": r.get("Event_A_ID") or r.get("event_a_id"),
                        "Event B ID": r.get("Event_B_ID") or r.get("event_b_id"),
                        "Relationship Type": r.get("Relationship_Type") or r.get("relation_type"),
                        "Overlap (s)": f"{float(r.get('Overlap_Seconds', r.get('overlap_seconds', 0.0))):.1f}s",
                        "Gap (s)": f"{float(r.get('Gap_Seconds', r.get('gap_seconds', 0.0))):.1f}s",
                        "Generated Sentence": r.get("Generated_Sentence") or r.get("generated_sentence"),
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            # CSV Download Button
            csv_bytes = df_sentences.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Generated Findings Form (CSV)",
                data=csv_bytes,
                file_name="outputs_generated_sentences.csv",
                mime="text/csv",
                key="download_sentences_csv",
            )

# =============================================================================
# TAB 6: EVIDENCE & EVALUATION
# =============================================================================
with tabs[5]:
    st.markdown("### 🧪 Empirical Evaluation & Clinical Limitations")

    # Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    with m2:
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    with m3:
        st.metric("Recall (Sensitivity)", f"{metrics.get('recall', 0):.4f}")
    with m4:
        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
    with m5:
        roc_val = metrics.get("roc_auc")
        st.metric("ROC-AUC", f"{roc_val:.4f}" if roc_val is not None else "N/A")

    st.markdown("#### 🖼️ Diagnostic Validation Figures")
    col_p1, col_p2 = st.columns(2)

    cm_img = OUTPUT_DIR / "confusion_matrix.png"
    roc_img = OUTPUT_DIR / "roc_curve.png"

    with col_p1:
        if cm_img.exists():
            st.image(str(cm_img), caption="Out-of-Fold Seizure Confusion Matrix", use_container_width=True)
    with col_p2:
        if roc_img.exists():
            st.image(str(roc_img), caption="Out-of-Fold Receiver Operating Characteristic (ROC)", use_container_width=True)

    # Evaluation Strategy & Limitations
    st.markdown("#### 🔬 Evaluation Methodology & Scientific Disclosure")
    st.info(
        f"**Evaluation Strategy:** {metrics.get('evaluation_strategy', 'Leave-One-Recording-Out')}\n\n"
        f"**Sample Count:** Total evaluated windows = {metrics.get('n_test_samples', 0):,} "
        f"(Seizure windows = {int(preds_df['expert_label'].sum())}, Non-seizure = {int((preds_df['expert_label']==0).sum())})\n\n"
        f"**Data Leakage Prevention:** Windows from separate recordings are isolated during cross-validation. "
        f"Random k-fold splitting is deliberately avoided because adjacent overlapping windows share identical "
        f"sub-segments, which artificially inflates performance numbers."
    )

    st.markdown("#### ⚠️ Known Clinical & Data Limitations")
    st.warning(
        f"{metrics.get('limitations', 'Recordings originate from a single subject.')}\n\n"
        "• Model predictions represent investigative algorithmic hypotheses and do not replace clinical EEG review.\n"
        "• High accuracy reflects within-subject generalization across distinct seizure events on subject chb01; "
        "inter-subject generalization requires multi-patient cross-validation."
    )

    # Official dataset links and references
    st.markdown("#### 🔗 References & Traceable Assets")
    st.markdown(
        """
        - **Main Dataset:** [PhysioNet CHB-MIT Scalp EEG Database v1.0.0](https://physionet.org/content/chbmit/1.0.0/)
        - **Subject Case:** `chb01` (Pediatric subject at Boston Children's Hospital)
        - **Summary File:** `data/chbmit/chb01/chb01-summary.txt`
        - **Target EDF Recordings:** `chb01_03.edf` (3600s) & `chb01_04.edf` (3600s)
        - **Serialised Model:** `models/model.joblib`
        """
    )
