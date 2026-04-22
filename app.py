"""
app.py — ClinIQ Clinical Intelligence Platform v2.0
=====================================================
Redesigned: modern UI · proper module names · fixed charts · clinical context
Run:  streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import re
from datetime import datetime as _dt
try:
    import shap as _shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False
try:
    from fpdf import FPDF as _FPDF
    _FPDF_OK = True
except ImportError:
    _FPDF_OK = False

from core.profiler import profile as run_profile
from core.cleaner import clean as run_clean
import modules.breast_cancer  as mod_bc
import modules.heart_disease   as mod_hd
import modules.diabetes        as mod_db
import modules.stroke          as mod_sk
import modules.respiratory     as mod_rp
import modules.universal       as mod_u
from core.target_detector import detect_target


# ── Helpers ─────────────────────────────────────────────────────────────────
def parse_clf_report(report_str: str) -> pd.DataFrame:
    """Parse sklearn classification_report string into a tidy DataFrame."""
    rows = []
    for line in report_str.strip().split("\n"):
        line = line.strip()
        if not line or "precision" in line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            support   = int(parts[-1])
            f1        = float(parts[-2])
            recall    = float(parts[-3])
            precision = float(parts[-4])
            name      = " ".join(parts[:-4])
            if name:
                rows.append({"Class": name, "Precision": precision,
                             "Recall": recall, "F1-Score": f1, "Support": support})
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows)

def smart_read(source) -> pd.DataFrame:
    """Read CSV or XLS/XLSX (including CSVs saved with .xls extension)."""
    name = getattr(source, "name", str(source)).lower()
    df = None
    if name.endswith(".xlsx"):
        df = pd.read_excel(source, engine="openpyxl")
    elif name.endswith(".xls"):
        try:
            df = pd.read_excel(source, engine="xlrd")
        except Exception:
            df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    return df


def _ascii(text: str) -> str:
    """Strip non-ASCII characters (emoji, Unicode) for PDF compatibility."""
    return re.sub(r"[^\x00-\x7F]+", "", str(text)).strip()


def generate_pdf_report(result, prof, module_name: str, target_col: str) -> bytes:
    """Build a clinical PDF report and return raw bytes."""
    if not _FPDF_OK:
        return b""
    pdf = _FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title block ──────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(0, 180, 216)
    pdf.cell(0, 12, "ClinIQ Clinical Intelligence Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 7, f"Generated: {_dt.now().strftime('%B %d, %Y at %H:%M')}", ln=True, align="C")
    pdf.cell(0, 7, f"Module: {_ascii(module_name)}   |   Target: {_ascii(target_col)}", ln=True, align="C")
    pdf.ln(4)
    pdf.set_draw_color(0, 180, 216)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    def _sec(title: str):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(0, 180, 216)
        pdf.cell(0, 9, _ascii(title), ln=True)
        pdf.set_draw_color(30, 58, 95)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)

    def _row(label: str, value: str, label2: str = "", value2: str = ""):
        pdf.cell(50, 7, _ascii(label) + ":", ln=False)
        pdf.cell(45, 7, _ascii(value), ln=False)
        if label2:
            pdf.cell(50, 7, _ascii(label2) + ":", ln=False)
            pdf.cell(45, 7, _ascii(value2), ln=False)
        pdf.ln(7)

    # ── Dataset Summary ───────────────────────────────────────────────────────
    _sec("Dataset Summary")
    _row("Total Records", f"{prof.n_rows:,}", "Features", str(prof.n_cols))
    _row("Quality Score", f"{prof.quality_score}/100", "Quality Label", prof.quality_label)
    _row("Duplicate Rows", str(prof.n_duplicates), "", "")
    pdf.ln(3)

    # ── Model Performance ─────────────────────────────────────────────────────
    _sec("Model Performance")
    total = len(result.predictions_df)
    n_high = int((result.predictions_df["prediction"] == 1).sum())
    n_low = total - n_high
    avg_prob = result.predictions_df["risk_probability"].mean()
    _row("Best Model", result.best_model_name, "Test ROC-AUC", f"{result.test_auc:.4f}")
    _row("High Risk Patients", f"{n_high} ({n_high/total:.1%})", "Low Risk Patients", f"{n_low} ({n_low/total:.1%})")
    _row("Avg Risk Probability", f"{avg_prob:.1%}", "", "")
    pdf.ln(3)

    # ── Cross-Validation Scores ───────────────────────────────────────────────
    _sec("5-Fold Cross-Validation AUC — All Models")
    for model, score in sorted(result.cv_scores.items(), key=lambda x: -x[1]):
        tag = " <-- WINNER" if model == result.best_model_name else ""
        pdf.cell(0, 7, f"  {_ascii(model)}{tag}: {score:.4f}", ln=True)
    pdf.ln(3)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    _sec("Confusion Matrix")
    tn, fp, fn, tp = result.confusion_matrix.ravel()
    _row("True Negatives (TN)", str(tn), "False Positives (FP)", str(fp))
    _row("False Negatives (FN)", str(fn), "True Positives (TP)", str(tp))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    _row("Precision", f"{prec:.3f}", "Recall", f"{rec:.3f}")
    _row("F1-Score", f"{f1:.3f}", "", "")
    pdf.ln(3)

    # ── Top Features ─────────────────────────────────────────────────────────
    if not result.feature_importances.empty:
        _sec("Top 10 Predictive Features")
        for rank, (_, row) in enumerate(result.feature_importances.head(10).iterrows(), 1):
            pdf.cell(10, 7, f"{rank}.", ln=False)
            pdf.cell(120, 7, _ascii(row["feature"]), ln=False)
            pdf.cell(0, 7, f"{row['importance']:.4f}", ln=True)
        pdf.ln(3)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_y(-18)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, "ClinIQ v2.0 -- Clinical Intelligence Platform  |  Confidential Clinical Report", align="C")

    return bytes(pdf.output())


# ── Global plot colour constants (used across multiple tabs) ─────────────────
_DARK_BG   = "#0D1E35"
_DARK_PLOT = "#071020"


# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinIQ — Clinical Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Vibrant dark theme (multi-colour palette) ── */
    [data-testid="stAppViewContainer"] {
        background: #071020 !important;
        background-image: radial-gradient(circle, #0F2A45 1px, transparent 1px) !important;
        background-size: 22px 22px !important;
    }
    [data-testid="stMainBlockContainer"] { background: transparent; }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#040A17 0%,#071020 60%,#091428 100%);
        border-right: 1px solid #0D2A48;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div { color: #94A3B8 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #E0F7FF !important; }
    [data-testid="stSidebar"] hr { border-color: #0D2A48; }
    /* Hero */
    .cliniq-hero {
        background: linear-gradient(135deg,#040A17 0%,#092840 40%,#0A4A6E 72%,#006B85 100%);
        padding:32px 44px; border-radius:20px; margin-bottom:28px;
        box-shadow:0 8px 40px rgba(0,0,0,0.55);
        border: 1px solid #0D3A5A;
    }
    .cliniq-title { font-size:3rem; font-weight:900; color:#FFFFFF;
                    margin:0; letter-spacing:-1px; }
    .cliniq-sub   { font-size:1.1rem; color:#5EEAD4; margin-top:6px; }
    .cliniq-pills { margin-top:14px; }
    .cpill { display:inline-block; background:rgba(0,180,216,0.13);
             color:#5EEAD4; padding:4px 14px; border-radius:20px;
             font-size:0.78rem; font-weight:600; margin-right:8px;
             border:1px solid rgba(0,180,216,0.32); }
    /* Section headings */
    .sec-head { font-size:1.05rem; font-weight:700; color:#00D4F5;
                border-left:5px solid #00B4D8; padding-left:12px;
                margin:20px 0 14px 0; }
    /* Generic cards */
    .card { background:#0D1E35; border-radius:16px; padding:24px 28px;
            box-shadow:0 4px 24px rgba(0,0,0,0.4); border:1px solid #1E3A5F;
            transition:transform 0.2s,box-shadow 0.2s; }
    .card:hover { transform:translateY(-3px);
                  box-shadow:0 10px 40px rgba(0,180,216,0.15); }
    .mod-icon { font-size:2.6rem; margin-bottom:12px; }
    .mod-name { font-size:1.1rem; font-weight:800; color:#E2E8F0; margin-bottom:6px; }
    .mod-desc { font-size:0.87rem; color:#64748B; line-height:1.55; }
    .mod-tag  { display:inline-block; border-radius:20px; font-size:0.75rem;
                font-weight:700; padding:3px 12px; margin-top:12px; }
    /* Step cards (welcome screen) */
    .step-card { background:#0D1E35; border-radius:14px; padding:20px;
                 box-shadow:0 2px 12px rgba(0,0,0,0.3); border:1px solid #1E3A5F;
                 text-align:center; }
    .step-num  { width:38px;height:38px;background:#006B85;color:#FFF;
                 border-radius:50%;font-weight:800;font-size:1rem;
                 display:flex;align-items:center;justify-content:center;
                 margin:0 auto 10px auto; }
    .step-ttl  { font-weight:700;color:#E2E8F0;font-size:0.95rem;margin-bottom:4px; }
    .step-dsc  { font-size:0.82rem;color:#64748B; }
    /* Standard KPI card */
    .kpi-card  { background:#0D1E35;border-radius:14px;padding:22px 26px;
                 box-shadow:0 3px 18px rgba(0,0,0,0.3);
                 border:1px solid #1E3A5F;border-top:5px solid #00B4D8;
                 text-align:center; }
    .kpi-val   { font-size:2.2rem;font-weight:900;color:#E8813A;line-height:1; }
    .kpi-lbl   { font-size:0.8rem;color:#64748B;font-weight:600;
                 text-transform:uppercase;letter-spacing:.5px;margin-top:6px; }
    /* Overview Dashboard KPI (left column) */
    .db-kpi { background:#0D1E35; border-radius:12px; padding:14px 18px;
              border:1px solid #1E3A5F; margin-bottom:10px; border-left:4px solid; }
    .db-kpi-lbl { font-size:0.70rem; font-weight:700; color:#64748B;
                  text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px; }
    .db-kpi-val { font-size:2rem; font-weight:900; line-height:1.1; }
    .db-kpi-ctx { font-size:0.74rem; color:#475569; margin-top:3px; }
    /* Chart section label */
    .chart-lbl { font-size:0.72rem; font-weight:700; color:#00B4D8;
                 text-transform:uppercase; letter-spacing:0.7px; margin-bottom:4px; }
    /* Dashboard title bar */
    .db-header { background:linear-gradient(90deg,#0D2A48,#071020);
                 border:1px solid #1E3A5F; border-radius:14px;
                 padding:14px 28px; margin-bottom:20px; text-align:center; }
    .db-header-title { font-size:1.35rem; font-weight:800; color:#00D4F5; }
    .db-header-sub   { font-size:0.82rem; color:#64748B; margin-top:3px; }
    /* Type badges */
    .badge-num  { background:#1E3A5F;color:#5EEAD4;border-radius:999px;
                  padding:2px 10px;font-size:0.74rem;font-weight:700; }
    .badge-cat  { background:#2D1B69;color:#C4B5FD;border-radius:999px;
                  padding:2px 10px;font-size:0.74rem;font-weight:700; }
    .badge-bool { background:#3B2500;color:#FCD34D;border-radius:999px;
                  padding:2px 10px;font-size:0.74rem;font-weight:700; }
    .badge-id   { background:#1E293B;color:#94A3B8;border-radius:999px;
                  padding:2px 10px;font-size:0.74rem;font-weight:700; }
    /* Info box */
    .info-box { background:#0D1E35;border-left:5px solid #00B4D8;
                padding:14px 18px;border-radius:0 12px 12px 0;
                font-size:0.88rem;color:#5EEAD4;margin-top:16px;
                border:1px solid #1E3A5F;border-left:5px solid #00B4D8; }
    /* Tables */
    .metric-tbl { width:100%;border-collapse:collapse; }
    .metric-tbl th { background:#040A17;color:#00B4D8;font-size:0.8rem;
                     font-weight:700;text-transform:uppercase;letter-spacing:.5px;
                     padding:10px 14px;text-align:left;
                     border-bottom:2px solid #1E3A5F; }
    .metric-tbl td { padding:10px 14px;border-bottom:1px solid #1E3A5F;
                     font-size:0.9rem;color:#CBD5E1; }
    .metric-tbl tr:hover td { background:#0D2A48; }
    /* Buttons */
    .stButton > button {
        background:linear-gradient(135deg,#006B85,#00B4D8) !important;
        color:#FFF !important;border:none !important;border-radius:10px !important;
        padding:12px 24px !important;font-weight:700 !important;
        font-size:0.9rem !important;width:100% !important;
        box-shadow:0 4px 14px rgba(0,180,216,0.38) !important;
    }
    .stButton > button:hover {
        background:linear-gradient(135deg,#00B4D8,#E8813A) !important;
        box-shadow:0 6px 22px rgba(0,180,216,0.55) !important;
    }
    /* Streamlit metric containers */
    div[data-testid="metric-container"] {
        background:#0D1E35;border-radius:12px;padding:18px;
        box-shadow:0 3px 14px rgba(0,0,0,0.35);border:1px solid #1E3A5F;
        border-top:3px solid #00B4D8;
    }
    div[data-testid="stMetricValue"] { color:#E8813A !important; }
    div[data-testid="stMetricLabel"] > div { color:#64748B !important; }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap:4px;background:#0D1E35;border-radius:12px;padding:5px;
        border:1px solid #1E3A5F;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius:8px;font-weight:500;color:#64748B;padding:8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background:#006B85 !important;color:#FFFFFF !important;
        font-weight:700 !important;box-shadow:0 2px 10px rgba(0,180,216,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────────────────────
for key in ["analysis_done", "profile", "clean_log", "result",
            "df_raw", "df_clean", "module_name", "target_col",
            "problem_type", "auto_confidence", "auto_candidates",
            "shap_vals", "shap_feat_names", "shap_X"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.analysis_done is None:
    st.session_state.analysis_done = False

# ── Module Definitions ─────────────────────────────────────────────────────
MODULES = {
    "🎗️  Breast Cancer": {
        "key": "BC",
        "name": "Breast Cancer Analysis",
        "subtitle": "Diagnosis · Staging · Survival",
        "default_target": "diagnosis",
        "runner": mod_bc.run,
        "sample_csv": "sample_data/breast_cancer/breast_cancer_wisconsin.csv",
        "description": "Predict Malignant vs Benign tumour diagnosis, cancer staging, or survival outcome from any breast cancer CSV — Wisconsin, TCGA, clinical staging records.",
        "icon": "🎗️",
        "color": "#EC4899",
        "tag_bg": "#FCE7F3",
        "tag_color": "#9D174D",
    },
    "🫀  Heart Disease": {
        "key": "HD",
        "name": "Heart Disease Analysis",
        "subtitle": "Disease Presence · Mortality",
        "default_target": "target",
        "runner": mod_hd.run,
        "sample_csv": "sample_data/heart_disease/heart.xls",
        "description": "Predict heart disease presence or cardiac mortality from cardiovascular, ECG, cholesterol, and clinical lab features.",
        "icon": "�",
        "color": "#DC2626",
        "tag_bg": "#FEE2E2",
        "tag_color": "#991B1B",
    },
    "🩸  Diabetes": {
        "key": "DB",
        "name": "Diabetes Analysis",
        "subtitle": "Diagnosis · Readmission",
        "default_target": "Outcome",
        "runner": mod_db.run,
        "sample_csv": "sample_data/diabetes/diabetes.csv",
        "description": "Predict diabetes diagnosis, 30-day readmission, or complication risk from glucose, insulin, BMI, and lifestyle features.",
        "icon": "🩸",
        "color": "#D97706",
        "tag_bg": "#FEF3C7",
        "tag_color": "#92400E",
    },
    "🧠  Stroke": {
        "key": "SK",
        "name": "Stroke Prediction",
        "subtitle": "Stroke Occurrence · Severity",
        "default_target": "stroke",
        "runner": mod_sk.run,
        "sample_csv": "sample_data/stroke/healthcare-dataset-stroke-data.xls",
        "description": "Predict stroke occurrence or severity from age, vascular risk factors, glucose levels, BMI, and lifestyle data.",
        "icon": "🧠",
        "color": "#7C3AED",
        "tag_bg": "#EDE9FE",
        "tag_color": "#5B21B6",
    },
    "🫁  Respiratory": {
        "key": "RP",
        "name": "Respiratory / Pneumonia",
        "subtitle": "Pneumonia · COVID-19 · ICU",
        "default_target": "pneumonia",
        "runner": mod_rp.run,
        "sample_csv": "sample_data/respiratory/Respiratory.datafin.csv",
        "description": "Predict pneumonia diagnosis, COVID-19 positivity, or ICU admission risk from respiratory vitals, oxygen saturation, and inflammatory markers.",
        "icon": "🫁",
        "color": "#059669",
        "tag_bg": "#D1FAE5",
        "tag_color": "#065F46",
    },
    "🌐  Universal Auto": {
        "key": "U",
        "name": "Universal Auto-Analysis",
        "subtitle": "Any Dataset · Zero Config",
        "default_target": None,
        "runner": None,
        "sample_csv": None,
        "description": "Upload ANY health or clinical CSV. ClinIQ auto-detects your target column, selects features, engineers variables, and trains the best model — zero configuration needed.",
        "icon": "🌐",
        "color": "#0EA5E9",
        "tag_bg": "#E0F2FE",
        "tag_color": "#0369A1",
    },
}

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 ClinIQ")
    st.markdown("**Clinical Intelligence Platform**")
    st.markdown("---")

    st.markdown("### 📦 Select Module")
    module_name = st.radio(
        "Analysis type", list(MODULES.keys()), label_visibility="collapsed"
    )
    mod_cfg = MODULES[module_name]
    st.caption(mod_cfg["description"])
    st.markdown("---")

    st.markdown("### 📂 Upload Data")
    uploaded = st.file_uploader(
        "Drop your CSV or Excel file (clean or messy)",
        type=["csv", "xls", "xlsx"], label_visibility="collapsed"
    )

    target_col_ui = None
    if uploaded:
        df_preview = smart_read(uploaded)
        uploaded.seek(0)

        # Auto-detect target for ALL modules
        auto_target, auto_conf, auto_candidates = detect_target(df_preview)
        conf_pct = int(auto_conf * 100)
        if auto_target:
            conf_color = (
                "#22C55E" if conf_pct >= 70
                else "#F59E0B" if conf_pct >= 40
                else "#EF4444"
            )
            st.markdown(f"""
            <div style="background:#0D1E35;border-radius:10px;
                 padding:12px 14px;border-left:4px solid {conf_color};
                 margin-bottom:8px">
              <div style="font-size:0.68rem;color:#64748B;font-weight:700;
                   text-transform:uppercase;letter-spacing:.5px">
                🤖 Auto-Detected Target</div>
              <div style="font-size:1.05rem;font-weight:800;
                   color:#E2E8F0;margin:4px 0">{auto_target}</div>
              <div style="font-size:0.76rem;color:{conf_color};
                   font-weight:600">Confidence: {conf_pct}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            auto_target = df_preview.columns[-1]
            st.warning("⚠️ Could not auto-detect target. Please select below.")

        default_idx = (
            list(df_preview.columns).index(auto_target)
            if auto_target in df_preview.columns else 0
        )
        target_col_ui = st.selectbox(
            "🔧 Override target (optional)",
            options=df_preview.columns.tolist(),
            index=default_idx,
        )

    run_btn  = st.button("▶  Run ClinIQ Analysis", disabled=(uploaded is None))
    st.markdown("---")
    st.markdown("### 🎲 Try Demo Data")
    _sample_path = mod_cfg.get("sample_csv")
    _has_demo    = bool(_sample_path and Path(_sample_path).exists())
    demo_btn = st.button(
        "⚡ Load & Analyse Demo",
        disabled=not _has_demo,
        help="Load the built-in sample dataset" if _has_demo else "No sample data for this module yet — upload your own CSV",
    )

    st.markdown("---")
    st.caption("ClinIQ v2.0 · 5-Disease Clinical ML Platform")


# ── Run analysis helper (unified for all modules) ────────────────────────────
def run_disease_analysis(df: pd.DataFrame, target_override, mod_key: str, runner):
    with st.spinner("🔍 Profiling data..."):
        prof = run_profile(df)
    with st.spinner("🧹 Cleaning data..."):
        df_clean, log = run_clean(df, prof)
    with st.spinner("🤖 Training models (CV in progress)..."):
        try:
            if mod_key == "BC":
                result, target_used, problem_type, confidence, candidates = runner(
                    df_clean, target_col=target_override
                )
                st.session_state["problem_type"] = problem_type
            elif mod_key == "U":
                result, target_used, confidence, candidates = mod_u.run(
                    df_clean, target_col=target_override
                )
            else:
                result, target_used, confidence, candidates = runner(
                    df_clean, target_col=target_override
                )
        except ValueError as e:
            st.error(f"❌ {e}")
            return
    st.session_state.profile            = prof
    st.session_state.clean_log          = log
    st.session_state.result             = result
    st.session_state.df_raw             = df
    st.session_state.df_clean           = df_clean
    st.session_state.module_name        = module_name
    st.session_state.target_col         = target_used
    st.session_state.analysis_done      = True
    st.session_state["auto_confidence"] = confidence
    st.session_state["auto_candidates"] = candidates


# ── Trigger analysis ───────────────────────────────────────────────────────
if run_btn and uploaded:
    df_in = smart_read(uploaded)
    run_disease_analysis(df_in, target_col_ui, mod_cfg["key"], mod_cfg["runner"])

if demo_btn and _has_demo:
    df_demo = smart_read(_sample_path)
    run_disease_analysis(df_demo, None, mod_cfg["key"], mod_cfg["runner"])


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="cliniq-hero">
  <div class="cliniq-title">🏥 ClinIQ</div>
  <div class="cliniq-sub">Clinical Intelligence Platform &nbsp;·&nbsp;
    Plug-in your data &rarr; auto-clean &rarr; auto-train &rarr; instant insights</div>
  <div class="cliniq-pills">
    <span class="cpill">❆ 5 Disease Modules</span>
    <span class="cpill">❆ Universal Auto-Mode</span>
    <span class="cpill">❆ Auto Target Detection</span>
    <span class="cpill">❆ 4-Model Selection</span>
    <span class="cpill">❆ Interactive Dashboard</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Welcome Screen ─────────────────────────────────────────────────────────
if not st.session_state.analysis_done:
    st.markdown('<p style="font-size:1.05rem;color:#475569;margin-bottom:24px">'
                'Select a disease module in the sidebar, upload your CSV '
                '(or click <b>⚡ Load &amp; Analyse Demo</b> for Breast Cancer), and ClinIQ handles '
                'everything — auto-detecting your target, cleaning the data, engineering features, '
                'selecting the best model, and delivering an interactive clinical dashboard.</p>',
                unsafe_allow_html=True)

    # Module cards — 2 rows of 3
    mod_items = list(MODULES.items())
    for row_start in range(0, len(mod_items), 3):
        row_items = mod_items[row_start:row_start + 3]
        cols = st.columns(len(row_items), gap="large")
        for col, (_, cfg) in zip(cols, row_items):
            with col:
                st.markdown(f"""
                <div class="card" style="border-top:5px solid {cfg['color']}">
                  <div class="mod-icon">{cfg['icon']}</div>
                  <div class="mod-name">{cfg['name']}</div>
                  <div class="mod-desc">{cfg['description']}</div>
                  <div class="mod-tag" style="background:{cfg['tag_bg']};color:{cfg['tag_color']}">
                    {cfg['subtitle']}
                  </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="sec-head">⚙️ How ClinIQ Works</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Upload",    "Drop any CSV — clean or messy, any medical format"),
        ("2", "Profile",   "Auto-detects column types, missing values, and invalid data"),
        ("3", "Clean",     "Removes duplicates, fixes negatives, imputes missing values"),
        ("4", "Engineer",  "Builds 20+ clinical features automatically"),
        ("5", "Train",     "Runs 5-fold CV on 4 models and picks the best by AUC"),
        ("6", "Explore",   "Full interactive dashboard with predictions & explanations"),
    ]
    cols = st.columns(6, gap="small")
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
              <div class="step-num">{num}</div>
              <div class="step-ttl">{title}</div>
              <div class="step-dsc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()


# ── Results Tabs ───────────────────────────────────────────────────────────
prof    = st.session_state.profile
log     = st.session_state.clean_log
result  = st.session_state.result
df_raw  = st.session_state.df_raw
df_clean = st.session_state.df_clean

# ── Pre-computed values (shared across all tabs) ───────────────────────────
_pred     = result.predictions_df.copy()
_total    = len(_pred)
_n_high   = int((_pred["prediction"] == 1).sum())
_n_low    = _total - _n_high
_avg_prob = _pred["risk_probability"].mean()
_imp_df   = result.feature_importances.head(20)
_mod_label = (st.session_state.module_name.split("  ")[1]
              if "  " in (st.session_state.module_name or "")
              else (st.session_state.module_name or "—"))

# Aliases for backward-compat with existing tab code
pred_df  = _pred
total    = _total
n_high   = _n_high
n_low    = _n_low
avg_prob = _avg_prob
imp_df   = _imp_df

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Overview",
    "📊 Data Profile",
    "🧹 Auto-Clean Report",
    "🤖 Model Results",
    "🎯 Predictions",
    "🔍 Feature Insights",
    "🩺 Patient Predictor",
    "📡 Data Drift",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — Overview Dashboard  (matches reference image style)
# ════════════════════════════════════════════════════════════════════════════
with tab0:
    # ── Title bar ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="db-header">
        <div class="db-header-title">📋 ClinIQ Clinical Intelligence Dashboard</div>
        <div class="db-header-sub">
            Module: <b>{_mod_label}</b> &nbsp;·&nbsp;
            Target: <code style='background:#1E3A5F;color:#93C5FD;
                padding:1px 6px;border-radius:4px'>{st.session_state.target_col}</code>
            &nbsp;·&nbsp; Best Model: <b style='color:#FBBF24'>{result.best_model_name}</b>
            &nbsp;·&nbsp; AUC: <b style='color:#34D399'>{result.test_auc:.4f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main layout: left KPI column + right chart grid ───────────────────────
    left_db, right_db = st.columns([1, 3], gap="medium")

    with left_db:
        kpis = [
            ("TOTAL PATIENTS",      f"{_total:,}",             "#E8813A", "#E8813A",  "Test-set size"),
            ("HIGH RISK",           f"{_n_high:,}",            "#EF4444", "#EF4444",  f"{_n_high/_total:.1%} of test set"),
            ("LOW RISK",            f"{_n_low:,}",             "#00C875", "#00C875",  f"{_n_low/_total:.1%} of test set"),
            ("TEST ROC-AUC",        f"{result.test_auc:.3f}",  "#FFC107", "#FFC107",  result.best_model_name),
            ("TRAINING ROWS",       f"{len(df_clean):,}",      "#C084FC", "#C084FC",  f"{prof.n_cols} features"),
            ("DATA QUALITY SCORE",  f"{prof.quality_score}",   "#00B4D8", "#00B4D8",  prof.quality_label),
        ]
        for lbl, val, col, border, ctx in kpis:
            st.markdown(f"""
            <div class="db-kpi" style="border-left-color:{border}">
                <div class="db-kpi-lbl">{lbl}</div>
                <div class="db-kpi-val" style="color:{col}">{val}</div>
                <div class="db-kpi-ctx">{ctx}</div>
            </div>
            """, unsafe_allow_html=True)

    with right_db:
        # Row 1 – three compact charts
        r1a, r1b, r1c = st.columns(3)

        with r1a:
            st.markdown('<div class="chart-lbl">Risk Distribution</div>',
                        unsafe_allow_html=True)
            fig_d1 = px.histogram(
                _pred, x="risk_probability", color="risk_label", nbins=30,
                color_discrete_map={"High Risk": "#EF4444", "Low Risk": "#22C55E"},
                template="plotly_dark", barmode="overlay", opacity=0.8,
            )
            fig_d1.update_layout(
                height=195, showlegend=False,
                paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
                margin=dict(l=6,r=6,t=4,b=26),
                xaxis=dict(showgrid=False, tickfont=dict(size=8), title=""),
                yaxis=dict(showgrid=True, gridcolor="#1E3A5F",
                           tickfont=dict(size=8), title=""),
            )
            st.plotly_chart(fig_d1, use_container_width=True)

        with r1b:
            st.markdown('<div class="chart-lbl">High vs Low Risk</div>',
                        unsafe_allow_html=True)
            fig_d2 = go.Figure(go.Pie(
                labels=["Low Risk", "High Risk"],
                values=[_n_low, _n_high], hole=0.62,
                marker_colors=["#22C55E", "#EF4444"],
                textinfo="percent", textfont_size=10,
                direction="clockwise",
            ))
            fig_d2.update_layout(
                height=195, showlegend=False,
                paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
                margin=dict(l=6,r=6,t=4,b=4),
                annotations=[dict(
                    text=f"<b style='color:#EF4444'>{_n_high/_total:.0%}</b><br>"
                         f"<span style='color:#64748B;font-size:10px'>High Risk</span>",
                    font=dict(size=13,color="#EF4444"), showarrow=False,
                )],
            )
            st.plotly_chart(fig_d2, use_container_width=True)

        with r1c:
            # Age group if available, else gender, else comorbidity count
            if "age_group" in df_clean.columns:
                st.markdown('<div class="chart-lbl">Patients by Age Group</div>',
                            unsafe_allow_html=True)
                ag = df_clean["age_group"].value_counts().reset_index()
                ag.columns = ["group", "n"]
                fig_d3 = px.bar(
                    ag, x="group", y="n",
                    color="n",
                    color_continuous_scale=["#0A3A5A", "#E8813A", "#FFAB76"],
                    template="plotly_dark",
                )
                fig_d3.update_layout(
                    height=195, showlegend=False, coloraxis_showscale=False,
                    paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
                    margin=dict(l=6,r=6,t=4,b=26),
                    xaxis=dict(showgrid=False, tickfont=dict(size=8), title=""),
                    yaxis=dict(showgrid=True, gridcolor="#1E3A5F",
                               tickfont=dict(size=8), title=""),
                )
                st.plotly_chart(fig_d3, use_container_width=True)
            elif "gender" in df_clean.columns:
                st.markdown('<div class="chart-lbl">Patients by Gender</div>',
                            unsafe_allow_html=True)
                gg = df_clean["gender"].value_counts().reset_index()
                gg.columns = ["g", "n"]
                fig_d3 = px.bar(
                    gg, x="g", y="n",
                    color="g",
                    color_discrete_sequence=["#4472C4", "#E8813A", "#64748B"],
                    template="plotly_dark",
                )
                fig_d3.update_layout(
                    height=195, showlegend=False,
                    paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
                    margin=dict(l=6,r=6,t=4,b=26),
                    xaxis=dict(showgrid=False, tickfont=dict(size=8), title=""),
                    yaxis=dict(showgrid=True, gridcolor="#1E3A5F",
                               tickfont=dict(size=8), title=""),
                )
                st.plotly_chart(fig_d3, use_container_width=True)
            else:
                st.markdown('<div class="chart-lbl">Comorbidity Count</div>',
                            unsafe_allow_html=True)
                if "comorbidity_count" in df_clean.columns:
                    cc = (df_clean["comorbidity_count"]
                          .value_counts().sort_index().reset_index())
                    cc.columns = ["count","n"]
                    fig_d3 = px.bar(cc, x="count", y="n",
                                    color="n",
                                    color_continuous_scale=["#0A3A5A","#E8813A"],
                                    template="plotly_dark")
                    fig_d3.update_layout(
                        height=195, showlegend=False, coloraxis_showscale=False,
                        paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
                        margin=dict(l=6,r=6,t=4,b=26),
                    )
                    st.plotly_chart(fig_d3, use_container_width=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Row 2 – three compact charts
        r2a, r2b, r2c = st.columns(3)

        with r2a:
            st.markdown('<div class="chart-lbl">Top Predictive Features</div>',
                        unsafe_allow_html=True)
            if not _imp_df.empty:
                top5_ov = _imp_df.head(5)
                fig_d4 = go.Figure(go.Bar(
                    x=top5_ov["importance"],
                    y=top5_ov["feature"],
                    orientation="h",
                    marker_color=["#EF4444","#F97316","#FBBF24","#34D399","#60A5FA"],
                    text=[f"{v:.3f}" for v in top5_ov["importance"]],
                    textposition="outside",
                    textfont=dict(size=8, color="#E2E8F0"),
                ))
                fig_d4.update_layout(
                    height=195, template="plotly_dark",
                    paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
                    margin=dict(l=6,r=44,t=4,b=6),
                    xaxis=dict(showgrid=False, tickfont=dict(size=8), title=""),
                    yaxis=dict(tickfont=dict(size=8), title=""),
                    showlegend=False,
                )
                st.plotly_chart(fig_d4, use_container_width=True)

        with r2b:
            st.markdown('<div class="chart-lbl">Model Comparison (CV AUC)</div>',
                        unsafe_allow_html=True)
            cv_ov = pd.DataFrame({
                "Model": list(result.cv_scores.keys()),
                "AUC":   list(result.cv_scores.values()),
            }).sort_values("AUC")
            m_colors = ["#E8813A" if m == result.best_model_name else "#1E3A5F"
                        for m in cv_ov["Model"]]
            fig_d5 = go.Figure(go.Bar(
                x=cv_ov["AUC"], y=cv_ov["Model"], orientation="h",
                marker_color=m_colors,
                text=[f"{v:.3f}" for v in cv_ov["AUC"]],
                textposition="outside",
                textfont=dict(size=9, color="#E2E8F0"),
            ))
            fig_d5.update_layout(
                height=195, template="plotly_dark",
                paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
                margin=dict(l=6,r=44,t=4,b=6),
                xaxis=dict(range=[0,1.08], showgrid=False,
                           tickfont=dict(size=8), title=""),
                yaxis=dict(tickfont=dict(size=8), title=""),
                showlegend=False,
            )
            st.plotly_chart(fig_d5, use_container_width=True)

        with r2c:
            st.markdown('<div class="chart-lbl">Confusion Matrix</div>',
                        unsafe_allow_html=True)
            _tn, _fp, _fn, _tp = result.confusion_matrix.ravel()
            fig_d6 = go.Figure(go.Heatmap(
                z=result.confusion_matrix,
                x=["Pred: Low", "Pred: High"],
                y=["Act: Low",  "Act: High"],
                colorscale=[[0,"#071020"],[0.4,"#0A3A5A"],[1,"#00748A"]],
                showscale=False,
            ))
            fig_d6.update_layout(
                annotations=[
                    dict(x=0,y=0,text=f"<b>{_tn}</b><br><sub>TN</sub>",
                         showarrow=False,font=dict(size=14,color="#60A5FA")),
                    dict(x=1,y=0,text=f"<b>{_fp}</b><br><sub>FP</sub>",
                         showarrow=False,font=dict(size=14,color="#EF4444")),
                    dict(x=0,y=1,text=f"<b>{_fn}</b><br><sub>FN</sub>",
                         showarrow=False,font=dict(size=14,color="#EF4444")),
                    dict(x=1,y=1,text=f"<b>{_tp}</b><br><sub>TP</sub>",
                         showarrow=False,font=dict(size=14,color="#22C55E")),
                ],
                height=195, template="plotly_dark",
                paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
                margin=dict(l=6,r=6,t=4,b=26),
                xaxis=dict(tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=8)),
            )
            st.plotly_chart(fig_d6, use_container_width=True)

        # ROC curve full-width at bottom
        st.markdown('<div class="chart-lbl">ROC Curve</div>', unsafe_allow_html=True)
        fig_d7 = go.Figure()
        fig_d7.add_trace(go.Scatter(
            x=np.concatenate([result.fpr,[1,0]]),
            y=np.concatenate([result.tpr,[0,0]]),
            fill="toself", fillcolor="rgba(0,180,216,0.15)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig_d7.add_trace(go.Scatter(
            x=result.fpr, y=result.tpr, mode="lines",
            name=f"{result.best_model_name}  AUC={result.test_auc:.3f}",
            line=dict(color="#00B4D8", width=2.5),
        ))
        fig_d7.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            name="Random (0.50)",
            line=dict(color="#EF4444", dash="dot", width=1.5),
        ))
        fig_d7.update_layout(
            height=180, template="plotly_dark",
            paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_PLOT,
            margin=dict(l=30,r=10,t=4,b=30),
            xaxis_title="FPR", yaxis_title="TPR",
            legend=dict(x=0.55,y=0.08,font=dict(size=9)),
            xaxis=dict(tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8)),
        )
        st.plotly_chart(fig_d7, use_container_width=True)

    # ── PDF Export ────────────────────────────────────────────────────────────
    st.markdown("---")
    pdf_col, _ = st.columns([1, 3])
    with pdf_col:
        if _FPDF_OK:
            if st.button("📄 Download PDF Report", key="pdf_btn"):
                with st.spinner("Generating PDF…"):
                    pdf_bytes = generate_pdf_report(
                        result, prof,
                        st.session_state.module_name or "",
                        st.session_state.target_col or "",
                    )
                st.download_button(
                    label="⬇ Save Clinical Report PDF",
                    data=pdf_bytes,
                    file_name=f"ClinIQ_Report_{_dt.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    key="pdf_download",
                )
        else:
            st.caption("Install `fpdf2` to enable PDF export.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Profile
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    q_color = {"Excellent": "#00B4D8", "Good": "#E8813A",
               "Fair": "#FFC107", "Poor": "#EF4444"}[prof.quality_label]

    # KPI row
    st.markdown('<div class="sec-head">📊 Dataset Overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📊 Total Rows",      f"{prof.n_rows:,}")
    k2.metric("🗒️ Columns",        prof.n_cols)
    k3.metric("🔄 Duplicate Rows",  f"{prof.n_duplicates:,}")
    k4.metric("⚠️ Quality Label",   prof.quality_label)

    # Gauge + missing chart side by side
    left_g, right_g = st.columns([1, 2])
    with left_g:
        st.markdown('<div class="sec-head">🎯 Quality Score</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prof.quality_score,
            number={"suffix": "/100", "font": {"size": 36, "color": q_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": q_color, "thickness": 0.28},
                "bgcolor": "white",
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 50],  "color": "#FEE2E2"},
                    {"range": [50, 70], "color": "#FEF3C7"},
                    {"range": [70, 85], "color": "#D1FAE5"},
                    {"range": [85,100], "color": "#DCFCE7"},
                ],
            },
            title={"text": prof.quality_label, "font": {"size": 14, "color": q_color}},
        ))
        fig_gauge.update_layout(
            height=260, paper_bgcolor="white",
            margin=dict(l=20, r=20, t=50, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with right_g:
        st.markdown('<div class="sec-head">🔴 Missing Values per Column</div>', unsafe_allow_html=True)
        miss_df = pd.DataFrame({
            "column":      [cp.name for cp in prof.columns.values()],
            "missing_pct": [cp.missing_pct for cp in prof.columns.values()],
        }).query("missing_pct > 0").sort_values("missing_pct")
        if miss_df.empty:
            st.success("✅ No missing values detected in the raw dataset.")
        else:
            fig_miss = px.bar(
                miss_df, x="missing_pct", y="column", orientation="h",
                color="missing_pct",
                color_continuous_scale=["#FCA5A5", "#EF4444", "#991B1B"],
                labels={"missing_pct": "Missing (%)", "column": ""},
                template="plotly_white",
            )
            fig_miss.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
            fig_miss.update_layout(
                height=max(260, len(miss_df) * 30),
                coloraxis_showscale=False,
                margin=dict(l=0, r=60, t=10, b=10),
                xaxis_title="Missing (%)",
            )
            st.plotly_chart(fig_miss, use_container_width=True)

    if prof.warnings:
        for w in prof.warnings:
            st.warning(f"⚠️  {w}")

    # Column details table with type badges
    st.markdown('<div class="sec-head">🗒️ Column Details</div>', unsafe_allow_html=True)
    badge_map = {"numeric": "badge-num", "categorical": "badge-cat",
                 "boolean": "badge-bool", "id": "badge-id", "datetime": "badge-id"}
    col_html = """<table class='metric-tbl'>
    <tr><th>Column</th><th>Type</th><th>Missing</th>
    <th>Invalid</th><th>Unique</th><th>Sample Values</th></tr>"""
    for cp in prof.columns.values():
        badge = badge_map.get(cp.dtype_detected, "badge-id")
        miss_cell = (f'<span style="color:#DC2626;font-weight:600">'
                     f'{cp.missing_count} ({cp.missing_pct}%)</span>'
                     if cp.missing_count > 0 else '—')
        inv_cell  = (f'<span style="color:#D97706;font-weight:600">{cp.invalid_count}</span>'
                     if cp.invalid_count > 0 else '—')
        col_html += (f"<tr><td><b>{cp.name}</b></td>"
                     f"<td><span class='{badge}'>{cp.dtype_detected}</span></td>"
                     f"<td>{miss_cell}</td><td>{inv_cell}</td>"
                     f"<td>{cp.n_unique}</td>"
                     f"<td style='color:#64748B;font-size:0.82rem'>{str(cp.sample_values[:3])}</td></tr>")
    col_html += "</table>"
    st.markdown(col_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Auto-Clean Report
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    total_fixed = sum(e["rows_affected"] for e in log)
    dupes_removed = next((e["rows_affected"] for e in log
                          if e["step"] == "Duplicate Removal"), 0)
    invalids_fixed = sum(e["rows_affected"] for e in log
                         if e["step"] == "Invalid Value Fix")
    imputed = sum(e["rows_affected"] for e in log
                  if "Imputation" in e["step"])

    st.markdown('<div class="sec-head">🧹 Auto-Cleaning Summary</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔄 Cleaning Steps",    len(log))
    k2.metric("♻️ Duplicates Removed", f"{dupes_removed:,}")
    k3.metric("⛔ Invalid Values Fixed", f"{invalids_fixed:,}")
    k4.metric("📊 Missing Imputed",    f"{imputed:,}")

    # Cleaning impact bar chart (cells fixed per column — always visible)
    st.markdown('<div class="sec-head">📉 Cells Fixed per Column</div>', unsafe_allow_html=True)
    before_miss = {c: int(df_raw[c].isna().sum()) for c in df_raw.columns}
    after_miss  = {c: int(df_clean[c].isna().sum())
                   for c in df_clean.columns if c in df_raw.columns}

    fixed_data = pd.DataFrame({
        "Column":      list(before_miss.keys()),
        "Cells Fixed": [before_miss[c] - after_miss.get(c, 0)
                         for c in before_miss.keys()],
    }).query("`Cells Fixed` > 0").sort_values("Cells Fixed")

    if fixed_data.empty:
        st.success("✅ Dataset was already clean — no imputation was needed.")
    else:
        fig_fix = px.bar(
            fixed_data, x="Cells Fixed", y="Column", orientation="h",
            color="Cells Fixed",
            color_continuous_scale=["#6EE7B7", "#059669", "#064E3B"],
            text="Cells Fixed",
            template="plotly_white",
            labels={"Cells Fixed": "Cells Imputed / Fixed", "Column": ""},
            title="How many cells were cleaned per column (0 remaining after clean)",
        )
        fig_fix.update_traces(textposition="outside")
        fig_fix.update_layout(
            height=max(280, len(fixed_data) * 30),
            coloraxis_showscale=False,
            margin=dict(l=0, r=60, t=40, b=10),
        )
        st.plotly_chart(fig_fix, use_container_width=True)
        st.success(f"✅ All {total_fixed:,} issues fixed. Dataset is now 100% clean and ready for modelling.")

    # Cleaning action log table (colour-coded step types)
    st.markdown('<div class="sec-head">📝 Cleaning Action Log</div>', unsafe_allow_html=True)
    step_colors = {
        "Duplicate Removal":  "#1A0A0A",
        "Invalid Value Fix":  "#1A1500",
        "Median Imputation":  "#001A2E",
        "Mode Imputation":    "#0D0A1A",
        "No Action Needed":   "#001A10",
    }
    log_html = """<table class='metric-tbl'>
    <tr><th>Step</th><th>Column</th><th>Action Taken</th><th>Rows Affected</th></tr>"""
    for entry in log:
        bg = step_colors.get(entry["step"], "#F8FAFC")
        log_html += (f"<tr style='background:{bg}'>"
                     f"<td><b>{entry['step']}</b></td>"
                     f"<td>{entry['column']}</td>"
                     f"<td>{entry['detail']}</td>"
                     f"<td style='text-align:center;font-weight:700'>{entry['rows_affected']:,}</td></tr>")
    log_html += "</table>"
    st.markdown(log_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Results
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    # KPI banner
    st.markdown('<div class="sec-head">🥇 Best Model Selected</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🥇 Best Model",      result.best_model_name)
    k2.metric("📈 Test ROC-AUC",    f"{result.test_auc:.4f}")
    k3.metric("🔄 CV AUC (Best)",    f"{result.cv_scores[result.best_model_name]:.4f}")
    k4.metric("📊 Models Compared", len(result.cv_scores))

    # Model comparison
    st.markdown('<div class="sec-head">📊 5-Fold Cross-Validated AUC — All Models</div>',
                unsafe_allow_html=True)
    cv_df = pd.DataFrame({
        "Model":   list(result.cv_scores.keys()),
        "CV AUC": list(result.cv_scores.values()),
        "Winner":  [m == result.best_model_name for m in result.cv_scores.keys()],
    }).sort_values("CV AUC", ascending=True)
    colors = ["#E8813A" if w else "#1E4A6E" for w in cv_df["Winner"]]
    fig_cv = go.Figure(go.Bar(
        x=cv_df["CV AUC"], y=cv_df["Model"], orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in cv_df["CV AUC"]],
        textposition="outside",
    ))
    fig_cv.add_vline(x=0.5, line_dash="dash", line_color="#EF4444",
                     annotation_text="● Random (0.50)", annotation_font_color="#EF4444")
    fig_cv.update_layout(
        height=220, template="plotly_white",
        xaxis={"range": [0, 1.05], "title": "ROC-AUC Score"},
        yaxis_title="", margin=dict(l=0, r=60, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="sec-head">🟙 Confusion Matrix</div>', unsafe_allow_html=True)
        cm = result.confusion_matrix
        tn, fp, fn, tp = cm.ravel()
        annotations = [
            dict(x=0, y=0, text=f"<b>{tn}</b><br><sub>True Neg</sub>",
                 showarrow=False, font=dict(size=16, color="#1D4ED8")),
            dict(x=1, y=0, text=f"<b>{fp}</b><br><sub>False Pos</sub>",
                 showarrow=False, font=dict(size=16, color="#DC2626")),
            dict(x=0, y=1, text=f"<b>{fn}</b><br><sub>False Neg</sub>",
                 showarrow=False, font=dict(size=16, color="#DC2626")),
            dict(x=1, y=1, text=f"<b>{tp}</b><br><sub>True Pos</sub>",
                 showarrow=False, font=dict(size=16, color="#059669")),
        ]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Predicted: Low", "Predicted: High"],
            y=["Actual: Low", "Actual: High"],
            colorscale=[[0, "#061828"], [0.5, "#006B85"], [1, "#00B4D8"]],
            showscale=False, zmin=0,
        ))
        fig_cm.update_layout(
            annotations=annotations, height=340,
            template="plotly_white",
            xaxis=dict(side="bottom"),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with right:
        st.markdown('<div class="sec-head">📉 ROC Curve</div>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        # Shaded area under ROC curve
        fig_roc.add_trace(go.Scatter(
            x=np.concatenate([result.fpr, [1, 0]]),
            y=np.concatenate([result.tpr, [0, 0]]),
            fill="toself", fillcolor="rgba(0,180,216,0.12)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig_roc.add_trace(go.Scatter(
            x=result.fpr, y=result.tpr, mode="lines",
            name=f"{result.best_model_name}  AUC = {result.test_auc:.3f}",
            line=dict(color="#00B4D8", width=3),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Baseline (0.50)",
            line=dict(color="#EF4444", dash="dot", width=2),
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=340, template="plotly_white",
            legend=dict(x=0.42, y=0.07, bgcolor="rgba(255,255,255,0.85)"),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Parsed classification report as styled table
    st.markdown('<div class="sec-head">📝 Performance Metrics by Class</div>',
                unsafe_allow_html=True)
    report_df = parse_clf_report(result.classification_report)
    if not report_df.empty:
        score_color = lambda v: (
            "#059669" if v >= 0.80 else "#D97706" if v >= 0.65 else "#DC2626"
        )
        rpt_html = """<table class='metric-tbl'>
        <tr><th>Class</th><th>Precision</th><th>Recall</th>
        <th>F1-Score</th><th>Support</th></tr>"""
        for _, row in report_df.iterrows():
            def fmt(v):
                c = score_color(v)
                bar_w = int(v * 60)
                return (f'<td><span style="color:{c};font-weight:700">{v:.2f}</span>'
                        f'<div style="height:4px;width:{bar_w}px;'
                        f'background:{c};border-radius:2px;margin-top:3px"></div></td>')
            rpt_html += (f"<tr><td><b>{row['Class']}</b></td>"
                         + fmt(row["Precision"]) + fmt(row["Recall"])
                         + fmt(row["F1-Score"])
                         + f"<td style='color:#64748B'>{int(row['Support'])}</td></tr>")
        rpt_html += "</table>"
        st.markdown(rpt_html, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="margin-top:14px">
    <b>Reading the metrics:</b> &nbsp;
    <b>Precision</b> = of all patients flagged as High Risk, how many truly are. &nbsp;
    <b>Recall</b> = of all actual High Risk patients, how many did we catch. &nbsp;
    <b>F1</b> = harmonic mean of Precision &amp; Recall.
    In clinical settings, <em>high Recall</em> is usually preferred to avoid missing sick patients.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Predictions
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    pred_df  = result.predictions_df.copy()
    total    = len(pred_df)
    n_high   = int((pred_df["prediction"] == 1).sum())
    n_low    = total - n_high
    avg_prob = pred_df["risk_probability"].mean()

    st.markdown('<div class="sec-head">🎯 Prediction Summary — Test Set</div>',
                unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👥 Total Patients",       f"{total:,}")
    k2.metric("🔴 High Risk",            f"{n_high:,}  ({n_high/total:.1%})")
    k3.metric("🟢 Low Risk",             f"{n_low:,}  ({n_low/total:.1%})")
    k4.metric("⚖️ Avg Risk Probability", f"{avg_prob:.1%}")

    # Risk distribution + donut side by side
    left_p, right_p = st.columns([2, 1])
    with left_p:
        st.markdown('<div class="sec-head">📊 Risk Score Distribution</div>',
                    unsafe_allow_html=True)
        fig_hist = px.histogram(
            pred_df, x="risk_probability", color="risk_label", nbins=40,
            color_discrete_map={"High Risk": "#EF4444", "Low Risk": "#22C55E"},
            labels={"risk_probability": "Predicted Risk Probability",
                    "count": "Patients", "risk_label": ""},
            template="plotly_white", barmode="overlay",
            opacity=0.75,
        )
        fig_hist.update_layout(height=300, legend=dict(x=0.7, y=0.95),
                               margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)
    with right_p:
        st.markdown('<div class="sec-head">🍡 Risk Split</div>',
                    unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=["Low Risk", "High Risk"],
            values=[n_low, n_high],
            hole=0.60,
            marker_colors=["#22C55E", "#EF4444"],
            textinfo="label+percent",
            textfont_size=13,
        ))
        fig_donut.update_layout(
            height=300, showlegend=False, template="plotly_white",
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[dict(text=f"{avg_prob:.0%}<br>avg risk",
                              font_size=14, showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Colour-coded patient table
    st.markdown('<div class="sec-head">📄 Patient-Level Predictions</div>',
                unsafe_allow_html=True)
    show_cols  = ["risk_label", "risk_probability", "actual"]
    extra_cols = [c for c in ["age", "gender", "disease_type",
                               "primary_diagnosis", "admission_type"]
                  if c in pred_df.columns]
    display_df = pred_df[extra_cols + show_cols].copy()
    display_df["risk_probability"] = display_df["risk_probability"].map("{:.1%}".format)
    display_df["actual"] = display_df["actual"].map(
        lambda v: "High Risk" if v == 1 else "Low Risk")
    st.dataframe(
        display_df.style.apply(
            lambda row: [
                "background-color:#FEE2E2;color:#991B1B"
                if row["risk_label"] == "High Risk"
                else "background-color:#DCFCE7;color:#166534"
            ] * len(row), axis=1,
        ),
        use_container_width=True, height=380,
    )
    csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Full Predictions CSV",
        data=csv_bytes, file_name="cliniq_predictions.csv", mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Feature Insights
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    imp_df = result.feature_importances.head(20)
    if imp_df.empty:
        st.info("Feature importance not available for this model type.")
    else:
        # Top-5 driver metric cards
        st.markdown('<div class="sec-head">🔥 Top 5 Clinical Risk Drivers</div>',
                    unsafe_allow_html=True)
        top5 = imp_df.head(5)
        cols = st.columns(5, gap="small")
        driver_colors = ["#E8813A", "#E91E8C", "#FFC107", "#00B4D8", "#C084FC"]
        for i, (col, (_, row)) in enumerate(zip(cols, top5.iterrows())):
            pct = int(row["importance"] / imp_df["importance"].max() * 100)
            with col:
                st.markdown(f"""
                <div class="card" style="border-top:5px solid {driver_colors[i]};
                    padding:18px 14px; text-align:center">
                  <div style="font-size:0.72rem;font-weight:700;color:{driver_colors[i]};
                    text-transform:uppercase;letter-spacing:1px">#{i+1} Driver</div>
                  <div style="font-size:1.5rem;font-weight:900;color:#E2E8F0;
                    margin:6px 0">{row['importance']:.3f}</div>
                  <div style="font-size:0.82rem;color:#475569;font-weight:600;
                    word-break:break-word">{row['feature']}</div>
                  <div style="margin-top:10px;height:6px;border-radius:3px;
                    background:linear-gradient(90deg,{driver_colors[i]},{driver_colors[i]}33);
                    width:{pct}%"></div>
                </div>
                """, unsafe_allow_html=True)

        # Full importance chart
        st.markdown('<div class="sec-head">📊 Feature Importance — All Top 20</div>',
                    unsafe_allow_html=True)
        plot_df = imp_df.sort_values("importance").copy()
        plot_df["rank"] = range(len(plot_df), 0, -1)
        plot_df["color"] = [
            driver_colors[0] if i < 3 else
            driver_colors[2] if i < 8 else "#4472C4"
            for i in range(len(plot_df) - 1, -1, -1)
        ]
        fig_imp = go.Figure(go.Bar(
            x=plot_df["importance"],
            y=plot_df["feature"],
            orientation="h",
            marker_color=plot_df["color"],
            text=[f"{v:.3f}" for v in plot_df["importance"]],
            textposition="outside",
        ))
        fig_imp.update_layout(
            height=max(420, len(plot_df) * 28),
            template="plotly_white",
            xaxis_title="Importance Score",
            yaxis=dict(tickfont=dict(size=11)),
            margin=dict(l=0, r=60, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # Clinical interpretation guide
        st.markdown('<div class="sec-head">🧠 Clinical Interpretation Guide</div>',
                    unsafe_allow_html=True)
        CLINICAL_MEANINGS = {
            "avg_glucose":       ("Blood Glucose", "Fasting glucose > 126 mg/dL indicates diabetes, a major risk driver."),
            "avg_hba1c":         ("HbA1c", "Long-term glycaemic control marker. > 7% suggests poor management."),
            "hyperglycaemia":    ("Hyperglycaemia Flag", "Auto-engineered: 1 if avg_glucose > 126 mg/dL."),
            "poor_glycaemic_control":("Poor Glycaemic Control","Auto-engineered: 1 if HbA1c > 7.0%."),
            "avg_hemoglobin":    ("Haemoglobin", "Low values (< 12 g/dL) indicate anaemia, linked to poor outcomes."),
            "anaemia":           ("Anaemia Flag", "Auto-engineered: 1 if haemoglobin below normal threshold."),
            "avg_creatinine":    ("Creatinine", "Elevated (> 1.3 mg/dL) signals reduced kidney function."),
            "avg_wbc":           ("WBC Count", "High WBC (> 11 ×10³/μL) indicates active infection or inflammation."),
            "avg_cholesterol":   ("Cholesterol", "Total cholesterol > 200 mg/dL linked to cardiovascular risk."),
            "has_heart_disease": ("Heart Disease", "Presence of documented cardiovascular disease."),
            "has_diabetes":      ("Diabetes","Type 2 diabetes is a primary multi-system risk factor."),
            "has_hypertension":  ("Hypertension", "Elevated BP increases stroke and heart failure risk."),
            "has_ckd":           ("Chronic Kidney Disease", "Reduced kidney function worsens systemic health outcomes."),
            "comorbidity_count": ("Comorbidity Count", "Number of co-existing chronic conditions. Higher = higher risk."),
            "n_meds":            ("Medication Count", "High number of medications indicates complex disease management."),
            "polypharmacy":      ("Polypharmacy", "Auto-engineered: 1 if patient takes ≥5 medications."),
            "n_visits":          ("Visit Count", "Frequent visits may indicate worsening or poorly managed condition."),
            "high_utilisation":  ("High Utilisation", "Auto-engineered: 1 if ≥5 visits."),
            "age_group":         ("Age Group", "Elderly patients carry higher inherent clinical risk."),
            "bmi_category":      ("BMI Category", "Obesity (BMI > 30) is linked to diabetes, hypertension, and CV disease."),
            "med_adherence_score":("Medication Adherence","Lower adherence strongly predicts disease progression."),
            "low_adherence":     ("Low Adherence Flag", "Auto-engineered: 1 if adherence score < 0.70."),
            "length_of_stay_days":("Length of Stay", "Longer stays indicate greater severity and readmission risk."),
            "prev_admissions_12m":("Prior Admissions","History of multiple admissions is the strongest readmission predictor."),
            "hospitalizations_6m":("Recent Hospitalisations","Recent hospital episodes signal actively deteriorating condition."),
        }
        top_names = list(imp_df["feature"].head(8))
        found = [(f, CLINICAL_MEANINGS[f]) for f in top_names if f in CLINICAL_MEANINGS]
        if found:
            tbl_html = "<table class='metric-tbl'><tr><th>Feature</th><th>Clinical Name</th><th>Clinical Meaning</th></tr>"
            for feat, (name, meaning) in found:
                tbl_html += (f"<tr><td><code style='background:#EFF6FF;color:#1D4ED8;"
                             f"padding:2px 6px;border-radius:4px'>{feat}</code></td>"
                             f"<td><b>{name}</b></td><td>{meaning}</td></tr>")
            tbl_html += "</table>"
            st.markdown(tbl_html, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <b>💡 How to use feature insights clinically:</b> Features ranked highest have the
        most influence on the model’s risk predictions. Clinical teams can prioritise
        monitoring and intervention on patients with abnormal values in these variables.
        Red bars = strongest drivers. The importance score is the mean decrease in
        impurity (Random Forest) or coefficient magnitude (Logistic Regression).
        </div>""", unsafe_allow_html=True)

        # ── SHAP Explainability ───────────────────────────────────────────────
        st.markdown('<div class="sec-head">🔮 SHAP Explainability</div>',
                    unsafe_allow_html=True)
        if not _SHAP_OK:
            st.info("Install `shap` to unlock SHAP waterfall plots: `pip install shap`")
        else:
            st.caption("SHAP (SHapley Additive exPlanations) shows exactly how much each "
                       "feature pushes a patient's risk score up or down.")
            if st.button("▶ Compute SHAP Values", key="shap_compute"):
                with st.spinner("Computing SHAP values (up to 30 s for large datasets)…"):
                    try:
                        _feat_cols = result.numeric_cols + result.categorical_cols
                        _avail     = [c for c in _feat_cols
                                      if c in result.predictions_df.columns]
                        X_raw      = result.predictions_df[_avail].head(120)
                        X_bg_raw   = (result.X_train_sample[_avail]
                                      if result.X_train_sample is not None
                                      else X_raw.head(50))
                        pre        = result.pipeline.named_steps["preprocessor"]
                        clf        = result.pipeline.named_steps["classifier"]
                        X_t        = pre.transform(X_raw)
                        X_bg_t     = pre.transform(X_bg_raw.head(50))
                        if result.categorical_cols:
                            try:
                                _cat_names = list(
                                    pre.named_transformers_["cat"]
                                    .named_steps["encoder"]
                                    .get_feature_names_out(result.categorical_cols))
                            except Exception:
                                _cat_names = result.categorical_cols
                        else:
                            _cat_names = []
                        fn_out = result.numeric_cols + _cat_names
                        if "Logistic" in result.best_model_name:
                            exp = _shap.LinearExplainer(clf, X_bg_t)
                            sv  = exp.shap_values(X_t)
                        else:
                            exp = _shap.TreeExplainer(clf, X_bg_t[:20],
                                                      feature_perturbation="interventional")
                            sv  = exp.shap_values(X_t)
                            if isinstance(sv, list):
                                sv = sv[1]
                        st.session_state.shap_vals       = sv
                        st.session_state.shap_feat_names = fn_out
                        st.session_state.shap_X          = X_raw.reset_index(drop=True)
                    except Exception as _e:
                        st.error(f"SHAP computation failed: {_e}")

            if st.session_state.shap_vals is not None:
                _sv  = st.session_state.shap_vals
                _sfn = st.session_state.shap_feat_names
                _sX  = st.session_state.shap_X

                # Mean |SHAP| summary bar
                mean_abs = np.abs(_sv).mean(axis=0)
                n_show   = min(20, len(_sfn))
                idx_sort = np.argsort(mean_abs)[-n_show:]
                shap_summary_df = pd.DataFrame({
                    "feature":   [_sfn[i] for i in idx_sort],
                    "mean_shap": [mean_abs[i] for i in idx_sort],
                })
                fig_shap_sum = go.Figure(go.Bar(
                    x=shap_summary_df["mean_shap"],
                    y=shap_summary_df["feature"],
                    orientation="h",
                    marker_color="#E8813A",
                ))
                fig_shap_sum.update_layout(
                    title="Mean |SHAP Value| — Global Feature Impact",
                    height=max(380, n_show * 22),
                    template="plotly_white",
                    xaxis_title="Mean |SHAP|",
                    yaxis=dict(tickfont=dict(size=10)),
                    margin=dict(l=0, r=60, t=40, b=10),
                )
                st.plotly_chart(fig_shap_sum, use_container_width=True)

                # Individual patient waterfall
                st.markdown("**Individual Patient SHAP Waterfall**")
                pat_idx = st.slider("Select patient row (test set index)",
                                    0, len(_sv) - 1, 0, key="shap_pat_slider")
                sv_row    = _sv[pat_idx]
                top_n     = min(15, len(_sfn))
                abs_order = np.argsort(np.abs(sv_row))[-top_n:]
                w_feats   = [_sfn[i] for i in abs_order]
                w_vals    = [sv_row[i] for i in abs_order]
                w_colors  = ["#EF4444" if v > 0 else "#00C875" for v in w_vals]
                fig_wf = go.Figure(go.Bar(
                    x=w_vals, y=w_feats, orientation="h",
                    marker_color=w_colors,
                    text=[f"{v:+.3f}" for v in w_vals],
                    textposition="outside",
                ))
                fig_wf.update_layout(
                    title=f"Patient #{pat_idx} — SHAP Waterfall "
                          f"({'High Risk' if _sv[pat_idx].sum() > 0 else 'Low Risk'})",
                    height=max(380, top_n * 28),
                    template="plotly_white",
                    xaxis_title="SHAP Value (red = raises risk, green = lowers risk)",
                    yaxis=dict(tickfont=dict(size=10)),
                    margin=dict(l=0, r=80, t=40, b=10),
                )
                st.plotly_chart(fig_wf, use_container_width=True)
                st.markdown("""
                <div class="info-box">
                <b>Reading SHAP:</b> Each bar shows how much that feature
                <span style="color:#EF4444"><b>raises (red)</b></span> or
                <span style="color:#00C875"><b>lowers (green)</b></span> this
                patient's predicted risk score relative to the average patient.
                Longer bars = stronger influence on this specific prediction.
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Patient Predictor
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="sec-head">🩺 Single Patient Risk Predictor</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B;font-size:0.95rem'>"
        "Fill in the patient's clinical values below and click <b>Predict Risk</b> "
        "to get an instant High / Low Risk classification from the trained model.</p>",
        unsafe_allow_html=True,
    )

    _num_cols = result.numeric_cols
    _cat_cols = result.categorical_cols

    if not _num_cols and not _cat_cols:
        st.warning("Feature column metadata not available. Re-run the analysis to enable this tab.")
    else:
        # Build reference DataFrame for ranges / unique values
        _feat_avail = [c for c in (_num_cols + _cat_cols)
                       if c in result.predictions_df.columns]
        _ref = result.predictions_df[_feat_avail].copy()
        if result.X_train_sample is not None:
            _tr_avail = [c for c in _feat_avail
                         if c in result.X_train_sample.columns]
            _ref = pd.concat([_ref, result.X_train_sample[_tr_avail]],
                             ignore_index=True)

        _show_num = [c for c in _num_cols if c in _feat_avail][:18]
        _show_cat = [c for c in _cat_cols if c in _feat_avail][:8]

        with st.form("patient_predictor_form"):
            _pvals: dict = {}

            if _show_num:
                st.markdown("##### 🔢 Numeric Features")
                _nc = st.columns(3)
                for _i, _c in enumerate(_show_num):
                    with _nc[_i % 3]:
                        _lo  = float(_ref[_c].min()) if _c in _ref else 0.0
                        _hi  = float(_ref[_c].max()) if _c in _ref else 100.0
                        _med = float(_ref[_c].median()) if _c in _ref else (_lo + _hi) / 2
                        _lo, _hi = min(_lo, _med), max(_hi, _med)
                        _pvals[_c] = st.number_input(
                            _c, min_value=_lo, max_value=_hi,
                            value=_med, format="%.3f", key=f"pp_num_{_c}",
                        )

            if _show_cat:
                st.markdown("##### 🏷️ Categorical Features")
                _cc = st.columns(3)
                for _i, _c in enumerate(_show_cat):
                    with _cc[_i % 3]:
                        _opts = sorted(_ref[_c].dropna().unique().tolist()) \
                                if _c in _ref else ["Unknown"]
                        _pvals[_c] = st.selectbox(_c, _opts, key=f"pp_cat_{_c}")

            _submitted = st.form_submit_button(
                "🔍  Predict Risk", use_container_width=True)

        if _submitted:
            # Fill any missing feature cols with median / mode from ref
            _full_row: dict = {}
            for _c in _num_cols:
                _full_row[_c] = (_pvals[_c] if _c in _pvals
                                 else float(_ref[_c].median()) if _c in _ref else 0.0)
            for _c in _cat_cols:
                if _c in _pvals:
                    _full_row[_c] = _pvals[_c]
                elif _c in _ref:
                    _mode = _ref[_c].mode()
                    _full_row[_c] = str(_mode[0]) if len(_mode) > 0 else "Unknown"

            try:
                _input_df = pd.DataFrame([_full_row])
                _proba    = float(result.pipeline.predict_proba(_input_df)[0][1])
                _label    = "High Risk" if _proba >= 0.5 else "Low Risk"
                _rcol     = "#EF4444" if _proba >= 0.5 else "#00C875"

                # KPI row
                st.markdown("<br>", unsafe_allow_html=True)
                _k1, _k2, _k3 = st.columns(3)
                with _k1:
                    st.markdown(f"""
                    <div class="db-kpi" style="border-left-color:{_rcol};
                         text-align:center;padding:22px 18px">
                        <div class="db-kpi-lbl">RISK CLASSIFICATION</div>
                        <div class="db-kpi-val" style="color:{_rcol};font-size:2rem">
                            {_label}</div>
                        <div class="db-kpi-ctx">Model: {result.best_model_name}</div>
                    </div>""", unsafe_allow_html=True)
                with _k2:
                    st.markdown(f"""
                    <div class="db-kpi" style="border-left-color:#FFC107;
                         text-align:center;padding:22px 18px">
                        <div class="db-kpi-lbl">RISK PROBABILITY</div>
                        <div class="db-kpi-val" style="color:#FFC107;font-size:2rem">
                            {_proba:.1%}</div>
                        <div class="db-kpi-ctx">Confidence score</div>
                    </div>""", unsafe_allow_html=True)
                with _k3:
                    st.markdown(f"""
                    <div class="db-kpi" style="border-left-color:#00B4D8;
                         text-align:center;padding:22px 18px">
                        <div class="db-kpi-lbl">DECISION THRESHOLD</div>
                        <div class="db-kpi-val" style="color:#00B4D8;font-size:2rem">
                            50%</div>
                        <div class="db-kpi-ctx">Standard clinical cut-off</div>
                    </div>""", unsafe_allow_html=True)

                # Gauge chart
                _fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=_proba * 100,
                    number={"suffix": "%", "font": {"size": 52, "color": _rcol}},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%",
                                 "tickfont": {"color": "#94A3B8"}},
                        "bar":  {"color": _rcol, "thickness": 0.28},
                        "bgcolor": _DARK_PLOT,
                        "bordercolor": "#1E3A5F",
                        "steps": [
                            {"range": [0,  30], "color": "#002A14"},
                            {"range": [30, 50], "color": "#1A2800"},
                            {"range": [50, 70], "color": "#2A1800"},
                            {"range": [70, 100],"color": "#2A0000"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 3},
                                      "thickness": 0.8, "value": 50},
                    },
                    title={"text": f"<b>{_label}</b>",
                           "font": {"size": 20, "color": _rcol}},
                ))
                _fig_g.update_layout(
                    height=320, paper_bgcolor=_DARK_BG,
                    font={"color": "#E2E8F0"},
                    margin=dict(l=40, r=40, t=60, b=20),
                )
                _gc, _ = st.columns([1, 1])
                with _gc:
                    st.plotly_chart(_fig_g, use_container_width=True)

                # Clinical recommendation box
                if _proba >= 0.7:
                    _advice = ("⚠️ <b>High priority review recommended.</b> This patient shows "
                               "strong risk indicators. Consider immediate clinical assessment.")
                    _box_col = "#3A0000"
                    _txt_col = "#EF4444"
                elif _proba >= 0.5:
                    _advice = ("🟡 <b>Moderate risk — monitor closely.</b> This patient is above "
                               "the risk threshold. Schedule follow-up within 2–4 weeks.")
                    _box_col = "#2A1800"
                    _txt_col = "#FFC107"
                else:
                    _advice = ("✅ <b>Low risk.</b> Current indicators suggest low clinical risk. "
                               "Continue standard monitoring schedule.")
                    _box_col = "#002A14"
                    _txt_col = "#00C875"

                st.markdown(f"""
                <div style="background:{_box_col};border-left:5px solid {_txt_col};
                     border-radius:0 12px 12px 0;padding:14px 20px;margin-top:16px;
                     border:1px solid #1E3A5F;border-left:5px solid {_txt_col}">
                    <span style="color:{_txt_col};font-size:0.95rem">{_advice}</span>
                </div>""", unsafe_allow_html=True)

            except Exception as _err:
                st.error(f"❌ Prediction failed: {_err}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — Data Drift Monitor
# ════════════════════════════════════════════════════════════════════════════
with tab7:
    import json as _json
    import io   as _io
    from pathlib import Path as _Path

    st.markdown('<div class="sec-head">📡 Data Drift Monitor</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Track whether incoming patient data has shifted from the training distribution. "
        "Upload a CSV of recent data below to run a live drift check against the stored baseline.",
        unsafe_allow_html=False,
    )

    _DRIFT_DIR = _Path("drift_reports")

    def _load_reports():
        reports = []
        if not _DRIFT_DIR.exists():
            return reports
        for d in sorted(_DRIFT_DIR.iterdir()):
            rpt_path = d / "latest_report.json"
            base_path = d / "baseline.parquet"
            if rpt_path.exists():
                with open(rpt_path, encoding="utf-8") as _f:
                    reports.append(_json.load(_f))
            elif base_path.exists():
                reports.append({"disease": d.name, "baseline_only": True})
        return reports

    _reports = _load_reports()

    if not _reports:
        st.info("No drift reports found. Train a model via the **Analyse** tab — "
                "the training data is automatically saved as the baseline.")
    else:
        # ── Summary table ─────────────────────────────────────────────────────
        st.markdown("#### Drift Status by Disease Model")

        _summary_rows = []
        for _r in _reports:
            if _r.get("baseline_only"):
                _summary_rows.append({
                    "Disease":       _r["disease"].replace("_", " ").title(),
                    "Last Check":    "—",
                    "Drift Ratio":   "—",
                    "Features Drifted": "—",
                    "Status":        "⬜ Baseline only",
                })
            else:
                _ratio = _r.get("drift_ratio", 0)
                if _r.get("dataset_drift"):
                    _badge = "🚨 CRITICAL"
                elif _r.get("alert"):
                    _badge = "⚠️ WARNING"
                else:
                    _badge = "✅ OK"
                _summary_rows.append({
                    "Disease":       _r["disease"].replace("_", " ").title(),
                    "Last Check":    _r.get("generated_at", "—")[:19].replace("T", " "),
                    "Drift Ratio":   f"{_ratio:.1%}",
                    "Features Drifted": f"{_r.get('columns_drifted',0)} / {_r.get('columns_checked',0)}",
                    "Status":        _badge,
                })

        import pandas as _pd_drift
        st.dataframe(
            _pd_drift.DataFrame(_summary_rows),
            use_container_width=True,
            hide_index=True,
        )

        # ── Per-disease detail ────────────────────────────────────────────────
        _full_reports = [_r for _r in _reports if not _r.get("baseline_only")]
        if _full_reports:
            _sel_disease = st.selectbox(
                "View per-feature detail for:",
                [_r["disease"] for _r in _full_reports],
                format_func=lambda x: x.replace("_", " ").title(),
                key="_drift_sel",
            )
            _sel = next(_r for _r in _full_reports if _r["disease"] == _sel_disease)

            _ratio = _sel.get("drift_ratio", 0)
            _c1, _c2, _c3, _c4 = st.columns(4)
            _c1.metric("Drift Ratio",      f"{_ratio:.1%}")
            _c2.metric("Features Drifted", f"{_sel.get('columns_drifted',0)} / {_sel.get('columns_checked',0)}")
            _c3.metric("Reference Rows",   f"{_sel.get('reference_rows',0):,}")
            _c4.metric("Current Rows",     f"{_sel.get('current_rows',0):,}")

            _alert_color = ("#FF4444" if _sel.get("dataset_drift")
                            else "#FFC107" if _sel.get("alert") else "#00C875")
            st.markdown(
                f'<div style="background:#0E1117;border-left:5px solid {_alert_color};'
                f'padding:12px 18px;border-radius:0 8px 8px 0;margin-bottom:16px;">'
                f'<span style="color:{_alert_color}">{_sel.get("summary","")}</span></div>',
                unsafe_allow_html=True,
            )

            # Per-feature bar chart
            _col_reports = _sel.get("column_reports", [])
            if _col_reports:
                import plotly.graph_objects as _go_d
                _feat_names  = [c["column"] for c in _col_reports]
                _feat_scores = [c.get("drift_score", 0) for c in _col_reports]
                _feat_drift  = [c.get("drift_detected", False) for c in _col_reports]
                _bar_colors  = ["#FF4444" if d else "#00C875" for d in _feat_drift]

                _fig_drift = _go_d.Figure(_go_d.Bar(
                    x=_feat_scores,
                    y=_feat_names,
                    orientation="h",
                    marker_color=_bar_colors,
                    text=[f"{s:.3f}" for s in _feat_scores],
                    textposition="outside",
                ))
                _fig_drift.update_layout(
                    title="Per-Feature Drift Score (KS statistic / PSI)",
                    xaxis_title="Drift Score (higher = more drift)",
                    yaxis_title="",
                    height=max(300, len(_feat_names) * 36 + 80),
                    plot_bgcolor="#0E1117",
                    paper_bgcolor="#0E1117",
                    font_color="#FAFAFA",
                    margin=dict(l=10, r=80, t=50, b=40),
                )
                _fig_drift.add_vline(x=0.1, line_dash="dash",
                                     line_color="#FFC107", opacity=0.5,
                                     annotation_text="Warning threshold",
                                     annotation_font_color="#FFC107")
                st.plotly_chart(_fig_drift, use_container_width=True)

                _drifted = [c["column"] for c in _col_reports if c.get("drift_detected")]
                if _drifted:
                    st.markdown(f"**Drifted features:** {', '.join(_drifted)}")

            if _sel.get("auto_retrain_triggered"):
                st.success("♻️ An automated retrain was triggered for this model when drift was detected.")

    # ── Live drift check ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Run a Live Drift Check")

    _disease_opts = [d.name for d in _DRIFT_DIR.iterdir()
                     if d.is_dir() and (d / "baseline.parquet").exists()] \
                    if _DRIFT_DIR.exists() else []

    if not _disease_opts:
        st.info("Train a model first to create a baseline, then upload new data here.")
    else:
        _d_col, _u_col = st.columns([1, 3])
        with _d_col:
            _live_disease = st.selectbox(
                "Disease model", _disease_opts,
                format_func=lambda x: x.replace("_", " ").title(),
                key="_live_drift_disease",
            )
        with _u_col:
            _live_file = st.file_uploader(
                "Upload recent patient data (CSV)",
                type=["csv"],
                key="_live_drift_file",
            )

        if _live_file is not None:
            try:
                import pandas as _pd_live
                from core.drift_detector import compute_drift as _compute_drift

                _live_df = _pd_live.read_csv(_io.BytesIO(_live_file.read()))
                with st.spinner("Running drift detection…"):
                    _live_report = _compute_drift(_live_disease, _live_df)

                _lr = _live_report
                _ra = _lr.get("drift_ratio", 0)
                _ac1, _ac2, _ac3 = st.columns(3)
                _ac1.metric("Drift Ratio",  f"{_ra:.1%}")
                _ac2.metric("Alert",        "Yes 🚨" if _lr.get("alert") else "No ✅")
                _ac3.metric("Dataset Drift","Yes 🚨" if _lr.get("dataset_drift") else "No ✅")

                _ac = ("#FF4444" if _lr.get("dataset_drift")
                       else "#FFC107" if _lr.get("alert") else "#00C875")
                st.markdown(
                    f'<div style="background:#0E1117;border-left:5px solid {_ac};'
                    f'padding:12px 18px;border-radius:0 8px 8px 0">'
                    f'<span style="color:{_ac}">{_lr.get("summary","")}</span></div>',
                    unsafe_allow_html=True,
                )
                if _lr.get("drifted_features"):
                    st.markdown(f"**Drifted features:** {', '.join(_lr['drifted_features'])}")

            except Exception as _de:
                st.error(f"Drift detection failed: {_de}")
