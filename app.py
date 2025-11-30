import io
from typing import Optional

import pandas as pd
import streamlit as st

import os
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("carewatch_app")

# ================== Load secrets / config from environment ==================
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")  # NOTE: we will NOT log this
DB_HOSTNAME = os.getenv("DB_HOSTNAME")
DB_PORT = os.getenv("DB_PORT")

# Features used for EDA and ML training
EDA_FEATURE_NAMES = os.getenv("EDA_FEATURE_NAMES", "glucose,systolic_bp,cholesterol")

# ML hyperparameters (string in .env → converted to Python dict)
ML_HYPERPARAMS = json.loads(os.getenv("ML_HYPERPARAMS", '{"learning_rate": 0.001, "batch_size": 32}'))

# Expected accuracy in experiment
EXPECTED_ACCURACY = float(os.getenv("EXPECTED_ACCURACY", "0.0"))

# Number of epochs
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "0"))

# Experiment metadata
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "carewatch_experiment_default")
EXPERIMENT_VERSION = os.getenv("EXPERIMENT_VERSION", "0.0.0")

# Streamlit port for Docker
STREAMLIT_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))

# Docker Hub username (not a secret)
DOCKERHUB_USERNAME = os.getenv("DOCKERHUB_USERNAME", "amohammed5877") 

# ================== Log loaded config (without sensitive values) ==================
logger.info("DB config loaded (host=%s, port=%s, user=%s)", DB_HOSTNAME, DB_PORT, DB_USERNAME)
logger.info(
    "Experiment config loaded: name=%s, version=%s, epochs=%s, expected_accuracy=%.3f",
    EXPERIMENT_NAME,
    EXPERIMENT_VERSION,
    TRAIN_EPOCHS,
    EXPECTED_ACCURACY,
)
logger.info("EDA features in use: %s", EDA_FEATURE_NAMES)
logger.info("ML hyperparameters: %s", ML_HYPERPARAMS)
logger.info("Streamlit server port (for Docker): %s", STREAMLIT_PORT)

from medical_report_core import (
    load_foods_table,
    parse_lab_report_text,
    classify_lab_metrics,
    lab_metrics_to_dataframe,
    get_risk_flags,
    recommend_foods,
    foods_to_avoid,
)

# Real OCR: Tesseract + pdf2image (already installed in your env earlier)
try:
    import pytesseract
    from PIL import Image

    _HAS_TESSERACT = True
except Exception:
    pytesseract = None
    Image = None
    _HAS_TESSERACT = False

try:
    from pdf2image import convert_from_bytes

    _HAS_PDF2IMAGE = True
except Exception:
    convert_from_bytes = None
    _HAS_PDF2IMAGE = False


# -----------------------------------------------------------------------------
# Streamlit config + styling (attractive dashboard)
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="CareWatch – Health & Diet Dashboard",
    layout="wide",
    page_icon="🩺",
)

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fb 0%, #e3f2fd 40%, #e8f5e9 100%);
    }
    .big-title {
        font-size: 32px;
        font-weight: 700;
        color: #1b4b82;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 15px;
        color: #455a64;
        text-align: center;
        margin-bottom: 1.6rem;
    }
    .card {
        background: rgba(255, 255, 255, 0.94);
        border-radius: 18px;
        padding: 1.0rem 1.2rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    }
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #1b4b82;
        margin-bottom: 0.4rem;
    }
    .risk-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .risk-low {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .risk-medium {
        background-color: #fff3e0;
        color: #ef6c00;
    }
    .risk-high {
        background-color: #ffebee;
        color: #c62828;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='big-title'>CareWatch – Health & Diet Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload your medical report to see health metrics and personalised diet suggestions.</div>",
    unsafe_allow_html=True,
)
with st.expander("Experiment configuration", expanded=False):
    st.write(f"Experiment: {EXPERIMENT_NAME} (v{EXPERIMENT_VERSION})")
    st.write(f"Expected accuracy: {EXPECTED_ACCURACY}")
    st.write(f"Training epochs: {TRAIN_EPOCHS}")
    st.write(f"Features used in EDA/model: {EDA_FEATURE_NAMES}")
    st.write(f"Docker Hub image owner: {DOCKERHUB_USERNAME}")

# -----------------------------------------------------------------------------
# OCR helper – uses real OCR engine
# -----------------------------------------------------------------------------

def run_ocr_on_file(uploaded_file) -> Optional[str]:
    """
    Run OCR on the uploaded file and return raw text.
    Uses real OCR (Tesseract). If you have your own custom OCR pipeline,
    you can replace this function and keep the rest of the app the same.
    """
    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.read()
    mime = uploaded_file.type or ""

    # PDF -> convert to images then OCR each page
    if "pdf" in mime.lower():
        if not (_HAS_PDF2IMAGE and _HAS_TESSERACT):
            return None
        pages = convert_from_bytes(file_bytes)
        text_chunks = [pytesseract.image_to_string(page) for page in pages]
        return "\n".join(text_chunks)

    # Image input
    if not _HAS_TESSERACT or Image is None:
        return None

    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)


# -----------------------------------------------------------------------------
# Load foods once
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_foods_table() -> pd.DataFrame:
    return load_foods_table()


foods_df = get_foods_table()

# -----------------------------------------------------------------------------
# Layout – left: upload, right: health overview
# -----------------------------------------------------------------------------

left_col, right_col = st.columns([1.2, 1.8])

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>1. Upload medical report</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a lab report image or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Process OCR + lab metrics
lab_metrics = None
metric_summary = []
metric_df = None
ocr_error = None

if uploaded is not None:
    ocr_text = run_ocr_on_file(uploaded)
    if not ocr_text:
        ocr_error = "We could not read text from this file. Please try another image or PDF."
    else:
        lab_metrics = parse_lab_report_text(ocr_text)
        metric_summary = classify_lab_metrics(lab_metrics)
        metric_df = lab_metrics_to_dataframe(lab_metrics)
        if not metric_summary:
            ocr_error = "No familiar glucose, blood pressure, or cholesterol values were found in this report."

# Right column – metrics + visualisation
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>2. Health overview</div>", unsafe_allow_html=True)

    if uploaded is None:
        st.info("Upload your medical report to see your metrics and diet suggestions.")
    elif ocr_error is not None:
        # Only useful message – no raw OCR text
        st.warning(ocr_error)
    else:
        # Metric cards
        if metric_summary:
            c1, c2, c3 = st.columns(3)
            cards = [c1, c2, c3]
            for idx, metric in enumerate(metric_summary[:3]):
                with cards[idx]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.metric(
                        label=metric["metric"],
                        value=metric["value"],
                        delta=metric["status"],
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

        # Visualisation (bar chart)
        if metric_df is not None and not metric_df.empty:
            import altair as alt

            chart = (
                alt.Chart(metric_df)
                .mark_bar()
                .encode(
                    x=alt.X("metric:N", title="Metric"),
                    y=alt.Y("value:Q", title="Value"),
                    color=alt.Color(
                        "status:N",
                        scale=alt.Scale(
                            domain=["Normal", "Borderline", "Elevated", "High", "High (Stage 1)", "High (Stage 2)"],
                            range=["#43a047", "#ffb300", "#fb8c00", "#e53935", "#e53935", "#b71c1c"],
                        ),
                        legend=None,
                    ),
                    tooltip=["metric", "value", "status"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)

        # Risk pills
        if lab_metrics:
            has_diabetes_risk, has_bp_risk, has_chol_risk = get_risk_flags(lab_metrics)
            parts = []
            if not (has_diabetes_risk or has_bp_risk or has_chol_risk):
                parts.append("<span class='risk-pill risk-low'>Overall risk looks low</span>")
            else:
                if has_diabetes_risk:
                    parts.append("<span class='risk-pill risk-high'>Diabetes risk</span>")
                if has_bp_risk:
                    parts.append("<span class='risk-pill risk-high'>Blood pressure risk</span>")
                if has_chol_risk:
                    parts.append("<span class='risk-pill risk-high'>Cholesterol risk</span>")

            if parts:
                st.markdown("<br/>" + " ".join(parts), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Diet suggestions – only if we have metrics
# -----------------------------------------------------------------------------

if lab_metrics:
    st.markdown("<br/>", unsafe_allow_html=True)

    col_rec, col_avoid = st.columns(2)

    with col_rec:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Recommended foods</div>", unsafe_allow_html=True)
        rec_foods = recommend_foods(lab_metrics, foods_df, top_n=12)
        st.dataframe(
            rec_foods[["description", "sugar_g", "sodium_mg", "sat_fat_g", "cholesterol_mg"]],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_avoid:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Foods to limit</div>", unsafe_allow_html=True)
        avoid_foods = foods_to_avoid(lab_metrics, foods_df, top_n=12)
        st.dataframe(
            avoid_foods[["description", "sugar_g", "sodium_mg", "sat_fat_g", "cholesterol_mg"]],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
