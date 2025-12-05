# carewatch_ml_N/recommender.py
"""
Hybrid ML + medical rule-based recommender.

Takes:
- cleaned, clustered food dataset (with columns:
  description, sugar_g, sodium_mg, sat_fat_g, cholesterol_mg, cluster_label)
- risk labels from thresholds.py (glucose / bp / cholesterol)
- fitted scaler

Returns:
- personalized health_score for each food
- top recommended foods
- foods to limit
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_health_score(
    df_clusters: pd.DataFrame,
    risks: dict,
    scaler,
) -> pd.DataFrame:
    """
    Build personalized health_score based on:
      - scaled sugar, sodium, sat_fat, cholesterol
      - medical risks (glucose / bp / cholesterol)

    The recommender uses multipliers to penalize nutrients that affect
    the user's conditions.

    Example:
        If BP is high → sodium weight increases
        If cholesterol is high → sat_fat + cholesterol weights increase
    """

    nutrient_cols = ["sugar_g", "sodium_mg", "sat_fat_g", "cholesterol_mg"]

    X = df_clusters[nutrient_cols].values
    X_scaled = scaler.transform(X)

    # Base weights
    w_sugar = 1.0
    w_sodium = 1.0
    w_satfat = 1.0
    w_chol = 1.0

    # Glucose risk → higher penalty on sugar
    if risks.get("glucose") in ["borderline", "high"]:
        w_sugar *= 2.5

    # BP risk → higher penalty on sodium
    if risks.get("bp") in ["elevated", "high"]:
        w_sodium *= 2.5

    # Cholesterol risk → higher penalty on sat fat + cholesterol
    if risks.get("cholesterol") in ["borderline", "high"]:
        w_satfat *= 2.0
        w_chol *= 2.5

    weights = np.array([w_sugar, w_sodium, w_satfat, w_chol])

    scores = (X_scaled * weights).sum(axis=1)

    df_scored = df_clusters.copy()
    df_scored["health_score"] = scores

    return df_scored


def _filter_junk_for_recommended(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove clearly 'junk' foods from the recommended list
    based on description keywords. These can still appear in
    the 'to limit' list.
    """
    junk_keywords = [
        "candy", "chocolate", "ice cream", "sherbet", "soda", "cola",
        "frosting", "icing", "donut", "doughnut", "cake", "brownie",
        "cookie", "biscuit", "pie", "pastry", "croissant", "chips",
        "fries", "fried", "sugar cookie", "sweet", "milk chocolate",
        "dark chocolate"
    ]

    desc = df["description"].str.lower()
    mask = ~desc.str.contains("|".join(junk_keywords))
    return df[mask]


def get_recommendations(
    df_scored: pd.DataFrame,
    n_top: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        recommended (top N from 'healthy' cluster, lowest scores,
                     with junk filtered out)
        to_limit    (top N from 'risky' cluster, highest scores)
    """

    # Recommended = healthiest cluster + lowest score
    candidate_recommended = (
        df_scored[df_scored["cluster_label"] == "healthy"]
        .sort_values("health_score", ascending=True)
    )

    candidate_recommended = _filter_junk_for_recommended(candidate_recommended)
    recommended = candidate_recommended.head(n_top)

    # To limit = highest score from risky cluster
    to_limit = (
        df_scored[df_scored["cluster_label"] == "risky"]
        .sort_values("health_score", ascending=False)
        .head(n_top)
    )

    return recommended, to_limit


if __name__ == "__main__":
    print("Recommender module ready. Import compute_health_score and get_recommendations in app_ml.py.")
