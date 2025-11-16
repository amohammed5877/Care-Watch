from time import sleep
from logger_setup import get_logger

logger = get_logger("carewatch.medical_report")


def upload_report(file_path: str):
    logger.info(f"Starting upload of medical report: {file_path}")
    # Simulate some work
    sleep(0.5)
    logger.info("Report upload completed and saved to temporary storage")


def run_ocr(file_path: str):
    logger.info(f"Running OCR on report: {file_path}")
    sleep(0.5)

    # Fake extracted metrics
    metrics_raw = {"glucose": "7.8 mmol/L", "cholesterol": "5.2 mmol/L"}
    logger.info(f"OCR extraction finished with {len(metrics_raw)} metrics found")

    return metrics_raw


def analyze_metrics(metrics_raw: dict):
    logger.info("Analyzing extracted metrics")
    findings = []

    if "glucose" in metrics_raw:
        findings.append("Glucose slightly elevated")

    if "cholesterol" in metrics_raw:
        findings.append("Cholesterol within normal range")

    logger.info(f"Generated {len(findings)} findings from metrics")
    return findings


def main():
    logger.info("CareWatch Medical Report pipeline started")

    file_path = "sample_report.pdf"
    upload_report(file_path)

    metrics = run_ocr(file_path)
    analyze_metrics(metrics)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()

