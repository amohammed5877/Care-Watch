# data_cleaning.py

import os
import numpy as np
import pandas as pd

RAW_PATH = os.path.join("datasets", "carewatch_master_food_small.csv")
CLEAN_PATH = os.path.join("datasets", "carewatch_master_food_clean.csv")


def clean_food_dataset(input_path: str = RAW_PATH, output_path: str = CLEAN_PATH) -> pd.DataFrame:
    """
    Load the original CareWatch food dataset, clean it, and save a new CSV.

    Steps:
    - Ensure required columns exist
    - Convert nutrient columns to numeric
    - Fill NaNs with safe defaults (0 for nutrients)
    - Clip extreme outliers (top 0.1%)
    - Fill missing text fields
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Expected columns (based on your old dataset)
    expected_cols = [
        "description",
        "calories_kcal",
        "sugar_g",
        "sodium_mg",
        "cholesterol_mg",
        "sat_fat_g",
        "food_category_id",
        "fdc_id",
    ]

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {missing_cols}")

    # --- 1) Clean text columns ---
    # If description/category is missing, fill with 'Unknown'
    df["description"] = df["description"].fillna("Unknown")
    if "food_category_id" in df.columns:
        df["food_category_id"] = df["food_category_id"].fillna("Unknown")

    # --- 2) Clean numeric nutrient columns ---
    nutrient_cols = ["calories_kcal", "sugar_g", "sodium_mg", "cholesterol_mg", "sat_fat_g"]

    # Coerce to numeric (in case some weird strings are there)
    for col in nutrient_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace NaNs with 0 for nutrients
    df[nutrient_cols] = df[nutrient_cols].fillna(0)

    # --- 3) Clip extreme outliers (top 0.1%) ---
    # This fixes values like 85938 mg cholesterol
    for col in nutrient_cols:
        # If column is all zeros, skip
        if (df[col] == 0).all():
            continue

        upper = df[col].quantile(0.999)  # 99.9th percentile
        df[col] = np.clip(df[col], 0, upper)

    # --- 4) Save cleaned dataset ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Cleaned dataset saved to: {output_path}")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    return df


if __name__ == "__main__":
    clean_food_dataset()
