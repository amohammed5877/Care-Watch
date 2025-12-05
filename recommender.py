# recommender.py

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from nutrition_model import NutritionSimilarityModel


def classify_health_risk(metrics: Dict[str, float]) -> str:
    """
    Classify health risk based on multiple metrics.

    Possible keys in metrics:
    - glucose_fasting (mg/dL)
    - glucose_pp (mg/dL)
    - hba1c (%)
    - systolic_bp (mmHg)
    - diastolic_bp (mmHg)
    - total_cholesterol (mg/dL)
    - ldl_cholesterol (mg/dL)
    - triglycerides (mg/dL)

    Returns: "normal", "borderline", or "high"
    """
    risk_scores = []

    # ----- Glucose (fasting) -----
    glucose = metrics.get("glucose_fasting")
    if glucose is not None:
        if glucose < 100:
            risk_scores.append(0)
        elif glucose < 126:
            risk_scores.append(1)  # prediabetes
        else:
            risk_scores.append(2)  # diabetes range

    # ----- Postprandial glucose -----
    gpp = metrics.get("glucose_pp")
    if gpp is not None:
        if gpp < 140:
            risk_scores.append(0)
        elif gpp < 200:
            risk_scores.append(1)
        else:
            risk_scores.append(2)

    # ----- HbA1c -----
    hba1c = metrics.get("hba1c")
    if hba1c is not None:
        if hba1c < 5.7:
            risk_scores.append(0)
        elif hba1c < 6.5:
            risk_scores.append(1)
        else:
            risk_scores.append(2)

    # ----- Blood Pressure -----
    systolic = metrics.get("systolic_bp")
    diastolic = metrics.get("diastolic_bp")
    if systolic is not None or diastolic is not None:
        s = systolic if systolic is not None else 0
        d = diastolic if diastolic is not None else 0

        # Simple rule: stage 1 / 2 hypertension
        if s >= 140 or d >= 90:
            risk_scores.append(2)
        elif s >= 130 or d >= 80:
            risk_scores.append(1)
        else:
            risk_scores.append(0)

    # ----- Lipids -----
    total_chol = metrics.get("total_cholesterol")
    if total_chol is not None:
        if total_chol < 200:
            risk_scores.append(0)
        elif total_chol < 240:
            risk_scores.append(1)
        else:
            risk_scores.append(2)

    ldl = metrics.get("ldl_cholesterol")
    if ldl is not None:
        if ldl < 100:
            risk_scores.append(0)
        elif ldl < 160:
            risk_scores.append(1)
        else:
            risk_scores.append(2)

    trig = metrics.get("triglycerides")
    if trig is not None:
        if trig < 150:
            risk_scores.append(0)
        elif trig < 200:
            risk_scores.append(1)
        else:
            risk_scores.append(2)

    # If no metrics at all → assume normal
    if not risk_scores:
        return "normal"

    max_risk = max(risk_scores)
    if max_risk == 0:
        return "normal"
    elif max_risk == 1:
        return "borderline"
    else:
        return "high"


def score_foods_by_risk(df: pd.DataFrame, risk_level: str) -> pd.DataFrame:
    """
    Add a risk-based score and recommendation label to foods.

    - Higher nutrient values -> higher 'risk_score'
    - If user risk is higher, we penalize high-risk foods more.
    - Outputs:
        - risk_score (float)
        - rec_label ("Recommended", "Neutral", "Limit")
    """
    df = df.copy()

    nutrient_cols = ["sugar_g", "sodium_mg", "sat_fat_g", "cholesterol_mg"]
    for col in nutrient_cols:
        if col not in df.columns:
            df[col] = 0

    # Normalize each nutrient to [0,1] based on max in dataset
    for col in nutrient_cols:
        max_val = df[col].max()
        if max_val > 0:
            df[col + "_norm"] = df[col] / max_val
        else:
            df[col + "_norm"] = 0

    # Base risk score: weighted sum
    df["risk_score_base"] = (
        0.4 * df["sugar_g_norm"]
        + 0.3 * df["sodium_mg_norm"]
        + 0.2 * df["sat_fat_g_norm"]
        + 0.1 * df["cholesterol_mg_norm"]
    )

    # Risk multiplier based on user health
    if risk_level == "normal":
        multiplier = 1.0
    elif risk_level == "borderline":
        multiplier = 1.2
    else:  # "high"
        multiplier = 1.5

    df["risk_score"] = df["risk_score_base"] * multiplier

    # Define cutoffs based on percentiles
    low_cut = np.percentile(df["risk_score"], 40)   # 40% lowest risk
    high_cut = np.percentile(df["risk_score"], 70)  # top 30% highest risk

    def label_row(rs: float) -> str:
        if rs <= low_cut:
            return "Recommended"
        elif rs >= high_cut:
            return "Limit"
        else:
            return "Neutral"

    df["rec_label"] = df["risk_score"].apply(label_row)

    return df


def hybrid_recommend(
    food_query: str,
    metrics: Dict[str, float],
    top_n_similar: int = 15,
    top_n_output: int = 5,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Hybrid logic (Option C):
    - Step 1: classify user health risk from metrics
    - Step 2: find similar foods (Option B model)
    - Step 3: apply risk-aware scoring, split into Recommended vs Limit
    """
    # 1) Risk classification from metrics
    risk_level = classify_health_risk(metrics)

    # 2) Similarity model
    model = NutritionSimilarityModel()
    df_similar = model.find_similar_foods(food_query, top_n=top_n_similar)

    # 3) Risk-aware scoring & labels
    df_scored = score_foods_by_risk(df_similar, risk_level=risk_level)

    # Separate recommended vs limit
    df_rec = df_scored[df_scored["rec_label"] == "Recommended"].copy()
    df_lim = df_scored[df_scored["rec_label"] == "Limit"].copy()

    # Sort for nicer display:
    df_rec = df_rec.sort_values(["risk_score", "similarity"], ascending=[True, False])
    df_lim = df_lim.sort_values(["risk_score", "similarity"], ascending=[False, False])

    return risk_level, df_rec.head(top_n_output), df_lim.head(top_n_output)


if __name__ == "__main__":
    # Simple quick test (optional)
    example_metrics = {
        "glucose_fasting": 160,
        "hba1c": 7.4,
        "total_cholesterol": 230,
        "systolic_bp": 145,
        "diastolic_bp": 92,
    }
    risk, _, _ = hybrid_recommend(
        food_query="pizza",
        metrics=example_metrics,
        top_n_similar=10,
        top_n_output=3,
    )
    print("Risk level:", risk)
