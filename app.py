import pandas as pd
import streamlit as st
import logging

import io
from typing import Optional

import os
import json

from medical_report_core import extract_metrics_from_report
from recommender import hybrid_recommend

# -----------------------------------------------------------------------------
# Application-level logging setup
# -----------------------------------------------------------------------------
app_logger = logging.getLogger("carewatch.app")
if not app_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)

app_logger.info("CareWatch Streamlit application started")


from mlops_services import (
    get_food_recommendation,
    refresh_food_dataset,
    train_food_health_model,
    upload_and_extract_report,
    compute_risk_score,
    check_and_confirm_latest_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("carewatch_app")

# ================== Load secrets / config from environment ==================
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")  
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


# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="CareWatch – AI Health & Food Recommender",
    layout="wide",
)


def show_sidebar():
    with st.sidebar:
        st.title("CareWatch 🩺🍽️")

        st.markdown(
            """
            **How it works (Hybrid Option B + C):**

            1. Upload your **medical report** (PDF or image).
            2. App runs **real OCR** on the report.
            3. We extract key metrics:
               - Glucose
               - Blood Pressure
               - Cholesterol
            4. We classify your **health risk**:
               - Normal / Borderline / High
            5. You enter a **food you crave** (e.g., *pizza*).
            6. App finds **similar foods by nutrients** (sugar, sodium, sat fat, etc.).
            7. It then labels them as:
               - ✅ Recommended
               - ⚠️ Limit
            """
        )

        st.markdown("---")
        st.markdown("**Tech stack:**")
        st.markdown(
            """
            - OCR: `pytesseract`, `pdf2image`
            - ML Similarity: `scikit-learn` (cosine similarity)
            - UI: `Streamlit`
            """
        )


def main():
    show_sidebar()

    st.title("CareWatch – AI-Powered Health & Food Recommendations")

    st.markdown(
        """
        Upload your medical report, then tell the app what you're craving.
        We'll use your **health metrics + food nutrients** to suggest what to eat
        and what to limit.
        """
    )

    # ---------------------------
    # File Upload: Medical Report
    # ---------------------------
    st.subheader("1️⃣ Upload your medical report (PDF or image)")

    uploaded_file = st.file_uploader(
        "Upload lab report / medical summary (PDF, JPG, PNG)",
        type=["pdf", "png", "jpg", "jpeg"],
    )

    metrics = {}
    ocr_text_preview = ""

    if uploaded_file is not None:
        # Read file bytes safely (works on every rerun)
        file_bytes = uploaded_file.getvalue()

        with st.spinner("Running OCR on your report..."):
            metrics, ocr_text = extract_metrics_from_report(
                file_bytes=file_bytes,
                file_name=uploaded_file.name,
            )

        ocr_text_preview = ocr_text[:800]  # first 800 chars

        st.success("✅ OCR complete.")

        # ---------------------------
        # Show Extracted Metrics + Visualization
        # ---------------------------
        st.subheader("2️⃣ Extracted health metrics")

        if metrics:
            # Display as a small table
            metrics_df = (
                pd.DataFrame([metrics])
                .T.reset_index()
                .rename(columns={"index": "Metric", 0: "Value"})
            )

            st.write("These values are parsed from your report:")
            st.table(metrics_df)

            # Simple bar chart of metrics
            st.markdown("**Metrics visualization**")
            metrics_chart_df = metrics_df.set_index("Metric")
            st.bar_chart(metrics_chart_df)

        else:
            st.warning(
                "Could not automatically detect metrics from the report. "
                "You can still get recommendations (we'll assume normal risk), "
                "but for best results, make sure terms like 'glucose', 'cholesterol', "
                "and 'blood pressure' appear clearly in the report."
            )

        # OCR text preview (for debugging / transparency)
        with st.expander("🔍 View OCR text preview (for debugging)"):
            st.text_area("OCR Text", ocr_text_preview, height=200)

    # ---------------------------
    # Food Query + Recommendations
    # ---------------------------
    st.subheader("3️⃣ Tell us what you're craving")

    food_query = st.text_input(
        "Example: pizza, burger, fried chicken, biryani, cookies, etc.",
        value="pizza",
    )

    # IMPORTANT: this must be aligned with the left margin inside `main()`
    if st.button("✨ Get AI Recommendations"):
        if not food_query.strip():
            st.error("Please type a food name first.")
            return

        # If no metrics found, fall back to empty dict (treated as 'normal' risk)
        metrics_to_use = metrics if metrics else {}

        try:
            with st.spinner("Computing hybrid recommendations..."):
                risk_level, df_rec, df_lim = hybrid_recommend(
                    food_query=food_query,
                    metrics=metrics_to_use,
                    top_n_similar=20,
                    top_n_output=5,
                )

            st.success(f"Detected health risk level: **{risk_level.upper()}**")

            # Layout: Recommended on left, Limit on right
            col1, col2 = st.columns(2)

            # --- Recommended ---
            with col1:
                st.markdown("### ✅ Recommended alternatives")
                if df_rec is not None and not df_rec.empty:
                    cols_show = [
                        c
                        for c in [
                            "description",
                            "rec_label",
                            "risk_score",
                            "similarity",
                            "calories_kcal",
                            "sugar_g",
                            "sodium_mg",
                            "sat_fat_g",
                            "cholesterol_mg",
                        ]
                        if c in df_rec.columns
                    ]
                    st.dataframe(df_rec[cols_show])
                else:
                    st.info("No strong 'Recommended' alternatives found for this search.")

            # --- Limit ---
            with col2:
                st.markdown("### ⚠️ Foods to limit")
                if df_lim is not None and not df_lim.empty:
                    cols_show = [
                        c
                        for c in [
                            "description",
                            "rec_label",
                            "risk_score",
                            "similarity",
                            "calories_kcal",
                            "sugar_g",
                            "sodium_mg",
                            "sat_fat_g",
                            "cholesterol_mg",
                        ]
                        if c in df_lim.columns
                    ]
                    st.dataframe(df_lim[cols_show])
                else:
                    st.info("No clear 'Limit' items detected among the similar foods.")

            # ---------- Nutrient comparison chart ----------
            st.subheader("4️⃣ Nutrient comparison: Recommended vs Limit")

            if (
                df_rec is not None
                and not df_rec.empty
                and df_lim is not None
                and not df_lim.empty
            ):
                nutrient_cols = [
                    "calories_kcal",
                    "sugar_g",
                    "sodium_mg",
                    "sat_fat_g",
                    "cholesterol_mg",
                ]
                nutrient_cols = [
                    c
                    for c in nutrient_cols
                    if c in df_rec.columns and c in df_lim.columns
                ]

                if nutrient_cols:
                    comp_df = pd.DataFrame(
                        {
                            "nutrient": nutrient_cols,
                            "Recommended (avg)": [
                                df_rec[c].mean() for c in nutrient_cols
                            ],
                            "Limit (avg)": [df_lim[c].mean() for c in nutrient_cols],
                        }
                    ).set_index("nutrient")

                    st.markdown(
                        "Lower bars on the **Recommended** side are better for sugar, sodium, "
                        "saturated fat and cholesterol."
                    )
                    st.bar_chart(comp_df)
                else:
                    st.info("Nutrient columns not available for comparison.")
            else:
                st.info("Not enough data to plot nutrient comparison.")

        except Exception as e:
            st.error(f"Error generating recommendations: {e}")


if __name__ == "__main__":
    main()
# -------------------------------------------------------------------------
# MLOps Pipeline Demo Section (Food Recommendation Service)
# -------------------------------------------------------------------------

st.markdown("---")
st.subheader("MLOps Pipeline Demo (Food Recommendation Service)")

with st.form("mlops_food_demo_form"):
    demo_food_name = st.text_input("Enter a food name for MLOps demo:", "Pizza")
    demo_age = st.number_input("Age (optional)", min_value=0, max_value=120, value=30)
    demo_weight = st.number_input(
        "Weight in kg (optional)", min_value=0.0, max_value=300.0, value=70.0
    )
    demo_conditions = st.multiselect(
        "Health conditions (optional)",
        ["diabetes", "hypertension", "high_cholesterol"],
    )

    submitted = st.form_submit_button("Run MLOps Food Recommendation Pipeline")

if submitted:
    app_logger.info(
        f"MLOps Food pipeline triggered from UI with "
        f"food_name={demo_food_name}, age={demo_age}, "
        f"weight={demo_weight}, conditions={demo_conditions}"
    )

    # Call the service layer (pipeline stub)
    result = get_food_recommendation(
        food_name=demo_food_name,
        age=int(demo_age) if demo_age else None,
        weight=float(demo_weight) if demo_weight else None,
        conditions=demo_conditions if demo_conditions else None,
    )

    app_logger.info("MLOps Food pipeline completed and response returned to UI")

    st.write("### Service Output")
    st.json(result)


# -------------------------------------------------------------------------
# MLOps Pipeline Demo Section (Medical Report – OCR + Risk Score)
# -------------------------------------------------------------------------

st.markdown("---")
st.subheader("MLOps Pipeline Demo (Medical Report – OCR + Risk Analysis)")

uploaded_demo_report = st.file_uploader("Upload a sample medical report PDF for MLOps demo", type=["pdf"])

if uploaded_demo_report:
    app_logger.info(
        f"MLOps Medical pipeline triggered from UI with file_name={uploaded_demo_report.name}"
    )

    temp_path = "temp_demo_report.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_demo_report.getbuffer())

    st.write("📄 **Step 1: Running OCR…**")
    ocr_result = upload_and_extract_report(temp_path)
    st.json(ocr_result)

    st.write("📊 **Step 2: Computing Risk Score…**")
    sample_metrics = {
        "glucose_fasting": 140,
        "cholesterol_total": 250,
        "systolic_bp": 150,
        "diastolic_bp": 95
    }

    risk_result = compute_risk_score(sample_metrics)
    st.json(risk_result)

    app_logger.info("MLOps Medical pipeline completed and response returned to UI")

    st.write("✨ **MLOps Medical Pipeline Executed Successfully!**")
