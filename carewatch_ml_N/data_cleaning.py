# carewatch_ml_N/data_cleaning.py
"""
Load, clean, and cluster the Kaggle nutrition dataset for CareWatch (ML version).

We standardize it so that the rest of the app can use:
    description, sugar_g, sodium_mg, sat_fat_g, cholesterol_mg
and ML clusters: healthy / medium / risky.
"""

from __future__ import annotations

import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

DEFAULT_DATA_PATH = "datasets/nutrition.csv"


# ---------- helpers to parse units ----------

def _parse_grams(value) -> float | None:
    """
    Parse strings like:
        '3.97 g', '0.2g', '71.97 g', '0', 0
    into a float in GRAMS.
    """
    if value is None:
        return None

    # if already numeric
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip().lower()
    if s == "" or s == "nan":
        return None

    # extract first number
    m = re.search(r"([0-9]*\.?[0-9]+)", s)
    if not m:
        return None

    num = float(m.group(1))

    # handle units
    if "mg" in s and "g" not in s.replace("mg", ""):
        # mg → g
        return num / 1000.0
    if "mcg" in s:
        # micrograms → g
        return num / 1_000_000.0

    # default: grams
    return num


def _parse_mg(value) -> float | None:
    """
    Parse strings like:
        '9.00 mg', '842.00 mg', '1mg', '0', 0
    into a float in MILLIGRAMS.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip().lower()
    if s == "" or s == "nan":
        return None

    m = re.search(r"([0-9]*\.?[0-9]+)", s)
    if not m:
        return None

    num = float(m.group(1))

    # if value is in grams
    if "g" in s and "mg" not in s.replace("mg", "") and "mcg" not in s:
        return num * 1000.0
    if "mcg" in s:
        return num / 1000.0

    # default assume mg
    return num
import numpy as np  # add this if it is not already there

# Nutrient columns we care about
NUTRIENT_COLS = ["sugar_g", "sodium_mg", "sat_fat_g", "cholesterol_mg"]

# ---------- main cleaning + clustering ----------

def load_and_prepare_food_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the Kaggle nutrition dataset and standardize it into:

        description, sugar_g, sodium_mg, sat_fat_g, cholesterol_mg

    Any rows where all four of these are missing will be dropped.
    Extreme outliers are clipped at the 99.5th percentile for stability.
    """

    df = pd.read_csv(path)

    # standard name column
    if "name" not in df.columns:
        raise ValueError("Expected a 'name' column in nutrition.csv")
    df["description"] = df["name"].astype(str)

    # parse nutrients
    if "sugars" not in df.columns:
        raise ValueError("Expected a 'sugars' column in nutrition.csv")
    df["sugar_g"] = df["sugars"].apply(_parse_grams)

    if "sodium" not in df.columns:
        raise ValueError("Expected a 'sodium' column in nutrition.csv")
    df["sodium_mg"] = df["sodium"].apply(_parse_mg)

    if "saturated_fat" not in df.columns:
        raise ValueError("Expected a 'saturated_fat' column in nutrition.csv")
    df["sat_fat_g"] = df["saturated_fat"].apply(_parse_grams)

    if "cholesterol" not in df.columns:
        raise ValueError("Expected a 'cholesterol' column in nutrition.csv")
    df["cholesterol_mg"] = df["cholesterol"].apply(_parse_mg)

    # keep only relevant columns
    df_small = df[["description", "sugar_g", "sodium_mg", "sat_fat_g", "cholesterol_mg"]].copy()

    # drop rows where all are missing
    df_small.dropna(subset=["sugar_g", "sodium_mg", "sat_fat_g", "cholesterol_mg"], how="all", inplace=True)

    # fill remaining NaNs with 0 (means "not reported"/very low)
    df_small[["sugar_g", "sodium_mg", "sat_fat_g", "cholesterol_mg"]] = df_small[
        ["sugar_g", "sodium_mg", "sat_fat_g", "cholesterol_mg"]
    ].fillna(0.0)

    # clip extreme outliers to 99.5th percentile to avoid crazy values in charts
    for col in ["sugar_g", "sodium_mg", "sat_fat_g", "cholesterol_mg"]:
        upper = df_small[col].quantile(0.995)
        df_small[col] = df_small[col].clip(lower=0.0, upper=upper)

    return df_small


def build_nutrition_clusters(
    df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
):
    """
    Build a K-Means model on [sugar_g, sodium_mg, sat_fat_g, cholesterol_mg]
    and assign cluster labels:

        'healthy', 'medium', 'risky'

    based on cluster-mean nutrient totals (lower total → "healthier").
    """

    nutrient_cols = ["sugar_g", "sodium_mg", "sat_fat_g", "cholesterol_mg"]
    X = df[nutrient_cols].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_clusters = df.copy()
    df_clusters["cluster"] = clusters

    # Cluster "risk score" = sum of mean nutrients per cluster
    cluster_risk = (
        df_clusters.groupby("cluster")[nutrient_cols]
        .mean()
        .sum(axis=1)
    )

    # Lower score = "healthier" cluster
    sorted_clusters = cluster_risk.sort_values().index.tolist()

    cluster_label_map = {
        sorted_clusters[0]: "healthy",
        sorted_clusters[1]: "medium",
        sorted_clusters[2]: "risky",
    }

    df_clusters["cluster_label"] = df_clusters["cluster"].map(cluster_label_map)

    return df_clusters, scaler, kmeans


# small self-test
if __name__ == "__main__":
    print("Loading and clustering Kaggle nutrition data...")
    df_food = load_and_prepare_food_data()
    print("Sample rows after cleaning:")
    print(df_food.head())

    df_clusters, scaler, kmeans = build_nutrition_clusters(df_food)
    print("\nCluster label counts:")
    print(df_clusters["cluster_label"].value_counts())
