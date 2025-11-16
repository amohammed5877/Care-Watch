import streamlit as st
from medical_report_core import upload_report, run_ocr, analyze_metrics
from logger_setup import get_logger

logger = get_logger("carewatch.gui")

st.title("CareWatch – Medical Report Analyzer")

uploaded_file = st.file_uploader("Upload a medical report", type=["pdf", "jpg", "png"])

if uploaded_file is not None:
    file_path = f"temp_{uploaded_file.name}"

    # Save file locally
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")
    logger.info(f"User uploaded file: {file_path}")

    # Run pipeline
    upload_report(file_path)
    metrics = run_ocr(file_path)
    findings = analyze_metrics(metrics)

    st.subheader("Extracted Metrics")
    st.json(metrics)

    st.subheader("Analysis Findings")
    for f in findings:
        st.write("DN", f)

    logger.info("Displayed analysis to user")
