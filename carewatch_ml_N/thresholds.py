# carewatch_ml_N/thresholds.py
"""
Simple, explainable medical thresholds for:
- fasting glucose
- systolic blood pressure
- total cholesterol

These are NOT for clinical use, just for a demo and to drive the scoring logic.
"""


def classify_glucose(glucose_mg_dl: float | None) -> str:
    """
    Fasting glucose (simplified):
      < 100  → normal
      100–125 → borderline (impaired fasting)
      >= 126 → high (possible diabetes)
    """
    if glucose_mg_dl is None:
        return "unknown"

    if glucose_mg_dl >= 126:
        return "high"
    elif glucose_mg_dl >= 100:
        return "borderline"
    else:
        return "normal"


def classify_bp_systolic(sbp_mm_hg: float | None) -> str:
    """
    Systolic blood pressure (simplified):
      < 120 → normal
      120–129 → elevated
      130–139 → Stage 1
      >= 140 → Stage 2 (we call it 'high')
    """
    if sbp_mm_hg is None:
        return "unknown"

    if sbp_mm_hg >= 140:
        return "high"
    elif sbp_mm_hg >= 130:
        return "elevated"
    else:
        return "normal"


def classify_cholesterol(total_chol_mg_dl: float | None) -> str:
    """
    Total cholesterol (simplified):
      < 200 → normal
      200–239 → borderline
      >= 240 → high
    """
    if total_chol_mg_dl is None:
        return "unknown"

    if total_chol_mg_dl >= 240:
        return "high"
    elif total_chol_mg_dl >= 200:
        return "borderline"
    else:
        return "normal"


def classify_risks(
    glucose_mg_dl: float | None,
    systolic_bp: float | None,
    cholesterol_mg_dl: float | None,
) -> dict:
    """
    Convenience helper to get all three risk labels at once.
    Returns a dict like:
        {
            "glucose": "borderline",
            "bp": "elevated",
            "cholesterol": "high"
        }
    """
    return {
        "glucose": classify_glucose(glucose_mg_dl),
        "bp": classify_bp_systolic(systolic_bp),
        "cholesterol": classify_cholesterol(cholesterol_mg_dl),
    }


if __name__ == "__main__":
    # Quick self-test
    print(classify_risks(95, 118, 180))   # all normal
    print(classify_risks(110, 135, 230))  # borderline glucose, elevated BP, borderline chol
    print(classify_risks(150, 160, 260))  # high everything
