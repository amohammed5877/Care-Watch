import re
from typing import Dict, Tuple, List, Optional

import pandas as pd

# ----------------------------------------------------------------------
# Extra food names (~200 items) we add to the food table
# ----------------------------------------------------------------------
EXTRA_FOOD_NAMES: List[str] = [
    "Chicken Salad", "Chicken Soup", "Chicken Curry", "Chicken Stir Fry", "Chicken Bowl",
    "Chicken Wrap", "Chicken Sandwich", "Chicken Stew", "Chicken Grill", "Chicken Roast",
    "Salmon Salad", "Salmon Soup", "Salmon Curry", "Salmon Stir Fry", "Salmon Bowl",
    "Salmon Wrap", "Salmon Sandwich", "Salmon Stew", "Salmon Grill", "Salmon Roast",
    "Tofu Salad", "Tofu Soup", "Tofu Curry", "Tofu Stir Fry", "Tofu Bowl",
    "Tofu Wrap", "Tofu Sandwich", "Tofu Stew", "Tofu Grill", "Tofu Roast",
    "Lentil Salad", "Lentil Soup", "Lentil Curry", "Lentil Stir Fry", "Lentil Bowl",
    "Lentil Wrap", "Lentil Sandwich", "Lentil Stew", "Lentil Grill", "Lentil Roast",
    "Chickpea Salad", "Chickpea Soup", "Chickpea Curry", "Chickpea Stir Fry", "Chickpea Bowl",
    "Chickpea Wrap", "Chickpea Sandwich", "Chickpea Stew", "Chickpea Grill", "Chickpea Roast",
    "Broccoli Salad", "Broccoli Soup", "Broccoli Curry", "Broccoli Stir Fry", "Broccoli Bowl",
    "Broccoli Wrap", "Broccoli Sandwich", "Broccoli Stew", "Broccoli Grill", "Broccoli Roast",
    "Spinach Salad", "Spinach Soup", "Spinach Curry", "Spinach Stir Fry", "Spinach Bowl",
    "Spinach Wrap", "Spinach Sandwich", "Spinach Stew", "Spinach Grill", "Spinach Roast",
    "Brown Rice Salad", "Brown Rice Soup", "Brown Rice Curry", "Brown Rice Stir Fry", "Brown Rice Bowl",
    "Brown Rice Wrap", "Brown Rice Sandwich", "Brown Rice Stew", "Brown Rice Grill", "Brown Rice Roast",
    "Oats Salad", "Oats Soup", "Oats Curry", "Oats Stir Fry", "Oats Bowl",
    "Oats Wrap", "Oats Sandwich", "Oats Stew", "Oats Grill", "Oats Roast",
    "Yogurt Salad", "Yogurt Soup", "Yogurt Curry", "Yogurt Stir Fry", "Yogurt Bowl",
    "Yogurt Wrap", "Yogurt Sandwich", "Yogurt Stew", "Yogurt Grill", "Yogurt Roast",
    "Egg Salad", "Egg Soup", "Egg Curry", "Egg Stir Fry", "Egg Bowl",
    "Egg Wrap", "Egg Sandwich", "Egg Stew", "Egg Grill", "Egg Roast",
    "Turkey Salad", "Turkey Soup", "Turkey Curry", "Turkey Stir Fry", "Turkey Bowl",
    "Turkey Wrap", "Turkey Sandwich", "Turkey Stew", "Turkey Grill", "Turkey Roast",
    "Quinoa Salad", "Quinoa Soup", "Quinoa Curry", "Quinoa Stir Fry", "Quinoa Bowl",
    "Quinoa Wrap", "Quinoa Sandwich", "Quinoa Stew", "Quinoa Grill", "Quinoa Roast",
    "Paneer Salad", "Paneer Soup", "Paneer Curry", "Paneer Stir Fry", "Paneer Bowl",
    "Paneer Wrap", "Paneer Sandwich", "Paneer Stew", "Paneer Grill", "Paneer Roast",
    "Cauliflower Salad", "Cauliflower Soup", "Cauliflower Curry", "Cauliflower Stir Fry", "Cauliflower Bowl",
    "Cauliflower Wrap", "Cauliflower Sandwich", "Cauliflower Stew", "Cauliflower Grill", "Cauliflower Roast",
    "Pumpkin Salad", "Pumpkin Soup", "Pumpkin Curry", "Pumpkin Stir Fry", "Pumpkin Bowl",
    "Pumpkin Wrap", "Pumpkin Sandwich", "Pumpkin Stew", "Pumpkin Grill", "Pumpkin Roast",
    "Almond Salad", "Almond Soup", "Almond Curry", "Almond Stir Fry", "Almond Bowl",
    "Almond Wrap", "Almond Sandwich", "Almond Stew", "Almond Grill", "Almond Roast",
    "Walnut Salad", "Walnut Soup", "Walnut Curry", "Walnut Stir Fry", "Walnut Bowl",
    "Walnut Wrap", "Walnut Sandwich", "Walnut Stew", "Walnut Grill", "Walnut Roast",
    "Banana Salad", "Banana Soup", "Banana Curry", "Banana Stir Fry", "Banana Bowl",
    "Banana Wrap", "Banana Sandwich", "Banana Stew", "Banana Grill", "Banana Roast",
    "Apple Salad", "Apple Soup", "Apple Curry", "Apple Stir Fry", "Apple Bowl",
    "Apple Wrap", "Apple Sandwich", "Apple Stew", "Apple Grill", "Apple Roast",
]


# ----------------------------------------------------------------------
# Load foods table (real CSV + extra ~200 foods)
# ----------------------------------------------------------------------


def load_foods_table(csv_path_main: str = "datasets/carewatch_master_food_small.csv") -> pd.DataFrame:
    """
    Load the main CareWatch food table (real food names + nutrients)
    and then append ~200 extra synthetic foods.

    If the main CSV is not found, fall back to synthetic foods only.
    """
    try:
        foods_df = pd.read_csv(csv_path_main)
        foods_df["is_extra"] = False  # mark real dataset rows
    except FileNotFoundError:
        # Fallback: only synthetic extras
        rows = []
        for i, name in enumerate(EXTRA_FOOD_NAMES, start=1):
            rows.append(
                {
                    "fdc_id": i,
                    "description": name,
                    "sugar_g": 5.0,
                    "sodium_mg": 80.0,
                    "sat_fat_g": 1.5,
                    "cholesterol_mg": 20.0,
                    "calories_kcal": 180.0,
                    "is_extra": True,
                }
            )
        return pd.DataFrame(rows)

    # Normalise columns we expect
    for col in ["sugar_g", "sodium_mg", "sat_fat_g", "cholesterol_mg", "calories_kcal"]:
        if col not in foods_df.columns:
            foods_df[col] = 0.0
    if "description" not in foods_df.columns and "food_name" in foods_df.columns:
        foods_df["description"] = foods_df["food_name"]

    # Append the ~200 extra food rows
    max_id = foods_df["fdc_id"].max() if "fdc_id" in foods_df.columns else 0
    extra_rows = []
    for i, name in enumerate(EXTRA_FOOD_NAMES, start=int(max_id) + 1):
        extra_rows.append(
            {
                "fdc_id": i,
                "description": name,
                "sugar_g": 5.0,
                "sodium_mg": 80.0,
                "sat_fat_g": 1.5,
                "cholesterol_mg": 20.0,
                "calories_kcal": 180.0,
                "is_extra": True,
            }
        )

    if extra_rows:
        foods_df = pd.concat([foods_df, pd.DataFrame(extra_rows)], ignore_index=True)

    return foods_df


# ----------------------------------------------------------------------
# OCR text parsing – only real OCR text passed in
# ----------------------------------------------------------------------


def _extract_number_from_text(pattern: str, text: str) -> Optional[float]:
    """
    Search pattern in text and return the first numeric value found.
    If nothing is found, return None.
    """
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None

    num_match = re.search(r"(\d+(?:\.\d+)?)", match.group(0))
    if not num_match:
        return None

    try:
        return float(num_match.group(1))
    except ValueError:
        return None


def parse_lab_report_text(ocr_text: str) -> Dict[str, float]:
    """
    Parse real OCR text from a medical report and extract key metrics.

    No fake values are generated here – everything comes from `ocr_text`.
    """
    text = ocr_text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)

    lab_metrics: Dict[str, float] = {}

    # Glucose (mg/dL)
    glucose = _extract_number_from_text(r"(fasting\s+)?glucose[^0-9]{0,12}\d", text)
    if glucose is not None:
        lab_metrics["glucose"] = glucose

    # Total Cholesterol (mg/dL)
    chol = _extract_number_from_text(r"(total\s+)?cholesterol[^0-9]{0,12}\d", text)
    if chol is not None:
        lab_metrics["cholesterol"] = chol

    # BP – "BP 130/85" or "Blood Pressure: 120 / 80"
    bp_match = re.search(
        r"(bp|blood pressure)[^0-9]{0,12}(\d{2,3})\s*/\s*(\d{2,3})",
        text,
        flags=re.IGNORECASE,
    )
    if bp_match:
        try:
            lab_metrics["systolic_bp"] = float(bp_match.group(2))
            lab_metrics["diastolic_bp"] = float(bp_match.group(3))
        except ValueError:
            pass
    else:
        # Systolic / diastolic written separately
        sys_val = _extract_number_from_text(r"systolic[^0-9]{0,12}\d{2,3}", text)
        dia_val = _extract_number_from_text(r"diastolic[^0-9]{0,12}\d{2,3}", text)
        if sys_val is not None:
            lab_metrics["systolic_bp"] = sys_val
        if dia_val is not None:
            lab_metrics["diastolic_bp"] = dia_val

    return lab_metrics


# ----------------------------------------------------------------------
# Risk helpers
# ----------------------------------------------------------------------


def get_risk_flags(lab_metrics: Dict[str, float]) -> Tuple[bool, bool, bool]:
    """
    Returns: (has_diabetes_risk, has_bp_risk, has_chol_risk)
    """
    glucose = lab_metrics.get("glucose", 0.0)
    systolic_bp = lab_metrics.get("systolic_bp", 0.0)
    cholesterol = lab_metrics.get("cholesterol", 0.0)

    has_diabetes_risk = glucose > 125
    has_bp_risk = systolic_bp > 130
    has_chol_risk = cholesterol > 200

    return has_diabetes_risk, has_bp_risk, has_chol_risk


def classify_lab_metrics(lab_metrics: Dict[str, float]) -> List[Dict[str, str]]:
    """
    Convert numeric lab metrics into user-friendly summary messages.
    """
    summary: List[Dict[str, str]] = []

    if "glucose" in lab_metrics:
        g = lab_metrics["glucose"]
        if g < 100:
            status = "Normal"
            detail = "Glucose is in the normal range."
        elif g < 126:
            status = "Borderline"
            detail = "Glucose is slightly above normal. Watch sugars and carbs."
        else:
            status = "High"
            detail = "Glucose is high and may indicate diabetes risk."
        summary.append(
            {
                "metric": "Glucose",
                "value": f"{g:.0f} mg/dL",
                "status": status,
                "detail": detail,
            }
        )

    if "systolic_bp" in lab_metrics:
        s = lab_metrics["systolic_bp"]
        d = lab_metrics.get("diastolic_bp")
        if s < 120:
            status = "Normal"
            detail = "Blood pressure is in the normal range."
        elif s < 130:
            status = "Elevated"
            detail = "Blood pressure is a bit high. Reduce salt and manage stress."
        elif s < 140:
            status = "High (Stage 1)"
            detail = "Blood pressure is high. Lifestyle changes are important."
        else:
            status = "High (Stage 2)"
            detail = "Blood pressure is very high. Follow-up is recommended."

        bp_value = f"{s:.0f} mmHg"
        if d is not None:
            bp_value = f"{s:.0f}/{d:.0f} mmHg"

        summary.append(
            {
                "metric": "Blood Pressure",
                "value": bp_value,
                "status": status,
                "detail": detail,
            }
        )

    if "cholesterol" in lab_metrics:
        c = lab_metrics["cholesterol"]
        if c < 200:
            status = "Normal"
            detail = "Total cholesterol is in the desirable range."
        elif c < 240:
            status = "Borderline"
            detail = "Cholesterol is borderline high. Be careful with fried and fatty foods."
        else:
            status = "High"
            detail = "Cholesterol is high. Low-fat diet and follow-up may be needed."
        summary.append(
            {
                "metric": "Total Cholesterol",
                "value": f"{c:.0f} mg/dL",
                "status": status,
                "detail": detail,
            }
        )

    return summary


def lab_metrics_to_dataframe(lab_metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Convert metrics into a DataFrame for charts.
    """
    rows: List[Dict[str, str]] = []
    for m in classify_lab_metrics(lab_metrics):
        # take first number before "/" so BP becomes systolic value
        value_str = m["value"].split()[0]
        value_str = value_str.split("/")[0]
        try:
            base_value = float(value_str)
        except ValueError:
            continue

        rows.append(
            {
                "metric": m["metric"],
                "value": base_value,
                "status": m["status"],
            }
        )

    if not rows:
        return pd.DataFrame(columns=["metric", "value", "status"])
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Diet recommendation logic
# ----------------------------------------------------------------------


def recommend_foods(lab_metrics: Dict[str, float], foods_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Recommend foods that are a good match for the user's lab profile.
    Lower sugar, sodium, saturated fat and cholesterol get better scores when there is risk.

    Real foods from the CSV (is_extra = False) are preferred; extras are used only
    if we run out of real foods.
    """
    has_diabetes_risk, has_bp_risk, has_chol_risk = get_risk_flags(lab_metrics)

    df = foods_df.copy()
    if "is_extra" not in df.columns:
        df["is_extra"] = False

    df["health_score_internal"] = 0.0

    # Diabetes risk -> penalise sugar
    if has_diabetes_risk and "sugar_g" in df.columns:
        df["health_score_internal"] += df["sugar_g"] * 2.0

    # BP risk -> penalise sodium
    if has_bp_risk and "sodium_mg" in df.columns:
        df["health_score_internal"] += df["sodium_mg"] / 50.0

    # Cholesterol risk -> penalise saturated fat and cholesterol
    if has_chol_risk:
        if "sat_fat_g" in df.columns:
            df["health_score_internal"] += df["sat_fat_g"] * 3.0
        if "cholesterol_mg" in df.columns:
            df["health_score_internal"] += df["cholesterol_mg"] / 20.0

    # Sort: real foods first, then extras, within each by health score (lower = better)
    df = df.sort_values(by=["is_extra", "health_score_internal"], ascending=[True, True])

    return df.head(top_n).drop(columns=["health_score_internal"])


def foods_to_avoid(lab_metrics: Dict[str, float], foods_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Highlight foods that are not ideal for the user based on their lab profile.

    Real foods (is_extra = False) are shown first. Extras are used only if we
    need more items to fill the list. Highest risk_score = worst foods.
    """
    has_diabetes_risk, has_bp_risk, has_chol_risk = get_risk_flags(lab_metrics)

    df = foods_df.copy()
    if "is_extra" not in df.columns:
        df["is_extra"] = False

    df["risk_score_internal"] = 0.0

    if has_diabetes_risk and "sugar_g" in df.columns:
        df["risk_score_internal"] += df["sugar_g"] * 2.0

    if has_bp_risk and "sodium_mg" in df.columns:
        df["risk_score_internal"] += df["sodium_mg"] / 50.0

    if has_chol_risk:
        if "sat_fat_g" in df.columns:
            df["risk_score_internal"] += df["sat_fat_g"] * 3.0
        if "cholesterol_mg" in df.columns:
            df["risk_score_internal"] += df["cholesterol_mg"] / 20.0

    # Sort: real foods first, then extras, risk score descending (worst first)
    df = df.sort_values(by=["is_extra", "risk_score_internal"], ascending=[True, False])

    return df.head(top_n).drop(columns=["risk_score_internal"])
