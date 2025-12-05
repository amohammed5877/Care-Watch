# medical_report_core.py

import io
import re
from typing import Dict, Tuple, Optional

from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes


def ocr_text_from_bytes(file_bytes: bytes, file_name: str) -> str:
    """
    Run real OCR on an uploaded file (PDF or image) and return text.
    - For PDFs: converts first 1–2 pages to images and OCRs them.
    - For images: OCR directly.
    """
    file_name = file_name.lower()

    if file_name.endswith(".pdf"):
        # Convert first 2 pages of PDF to images
        images = convert_from_bytes(file_bytes, dpi=300)
        if not images:
            return ""

        text_pages = []
        for img in images[:2]:
            text_pages.append(pytesseract.image_to_string(img))

        text = "\n".join(text_pages)

    else:
        # Treat as image (jpg, png, etc.)
        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img)

    return text


def _extract_number(pattern: str, text: str) -> Optional[float]:
    """
    Helper to find the first number matching a regex pattern.
    Returns float or None.
    """
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_metrics_from_text(text: str) -> Dict[str, float]:
    """
    Parse OCR text and extract key medical metrics.

    Tries to detect:
    - glucose_fasting        (mg/dL)
    - glucose_pp             (mg/dL)
    - hba1c                  (%)
    - total_cholesterol      (mg/dL)
    - ldl_cholesterol        (mg/dL)
    - hdl_cholesterol        (mg/dL)
    - triglycerides          (mg/dL)
    - systolic_bp            (mmHg)
    - diastolic_bp           (mmHg)
    """

    metrics: Dict[str, float] = {}

    # Normalize spaces
    clean_text = " ".join(text.split())

    # ---------- Glucose ----------
    # Fasting glucose
    val = _extract_number(r"fasting\s+glucose[^\d]{0,15}(\d{2,3})", clean_text)
    if val is None:
        # fallback to generic glucose / FBS
        val = _extract_number(r"(?:glucose|fbs)[^\d]{0,15}(\d{2,3})", clean_text)
    if val is not None:
        metrics["glucose_fasting"] = val

    # Postprandial / PP glucose
    val = _extract_number(
        r"(?:postprandial\s+glucose|pp)[^\d]{0,15}(\d{2,3})", clean_text
    )
    if val is not None:
        metrics["glucose_pp"] = val

    # HbA1c (can be decimal)
    val = _extract_number(r"hba1c[^\d]{0,10}(\d+(?:\.\d+)?)", clean_text)
    if val is not None:
        metrics["hba1c"] = val

    # ---------- Lipids ----------
    val = _extract_number(
        r"total\s+cholesterol[^\d]{0,15}(\d{2,3})", clean_text
    )
    if val is not None:
        metrics["total_cholesterol"] = val

    val = _extract_number(
        r"ldl\s+cholesterol[^\d]{0,15}(\d{2,3})", clean_text
    )
    if val is not None:
        metrics["ldl_cholesterol"] = val

    val = _extract_number(
        r"hdl\s+cholesterol[^\d]{0,15}(\d{2,3})", clean_text
    )
    if val is not None:
        metrics["hdl_cholesterol"] = val

    val = _extract_number(
        r"triglycerides?[^\d]{0,15}(\d{2,3})", clean_text
    )
    if val is not None:
        metrics["triglycerides"] = val

    # ---------- Blood Pressure ----------
    # Systolic/Diastolic written separately
    sys_val = _extract_number(
        r"systolic\s*(?:bp|blood\s*pressure)?[^\d]{0,10}(\d{2,3})", clean_text
    )
    dia_val = _extract_number(
        r"diastolic\s*(?:bp|blood\s*pressure)?[^\d]{0,10}(\d{2,3})", clean_text
    )

    # Fallback to "BP 145/92" style
    if sys_val is None or dia_val is None:
        match_bp = re.search(
            r"(?:blood\s*pressure|bp)[^\d]{0,10}(\d{2,3})\s*/\s*(\d{2,3})",
            clean_text,
            flags=re.IGNORECASE,
        )
        if match_bp:
            try:
                sys_val = float(match_bp.group(1))
                dia_val = float(match_bp.group(2))
            except ValueError:
                pass

    if sys_val is not None:
        metrics["systolic_bp"] = sys_val
    if dia_val is not None:
        metrics["diastolic_bp"] = dia_val

    return metrics


def extract_metrics_from_report(file_bytes: bytes, file_name: str) -> Tuple[Dict[str, float], str]:
    """
    High-level helper:
    - Runs OCR on the file
    - Extracts metrics
    - Returns (metrics_dict, ocr_text)
    """
    text = ocr_text_from_bytes(file_bytes, file_name)
    metrics = extract_metrics_from_text(text)
    return metrics, text


if __name__ == "__main__":
    print("Use extract_metrics_from_report() from app.py")
