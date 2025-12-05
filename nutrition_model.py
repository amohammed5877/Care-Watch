# nutrition_model.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

CLEAN_PATH = os.path.join("datasets", "carewatch_master_food_clean.csv")


class NutritionSimilarityModel:
    """
    ML-based nutrient similarity model.
    - Loads cleaned food dataset
    - Represents each food as a nutrient vector
    - Uses cosine similarity to find similar foods
    """

    def __init__(self, csv_path: str = CLEAN_PATH):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Clean dataset not found at {csv_path}. "
                f"Run data_cleaning.py first to generate it."
            )

        self.df = pd.read_csv(csv_path)

        self.feature_cols = [
            "calories_kcal",
            "sugar_g",
            "sodium_mg",
            "cholesterol_mg",
            "sat_fat_g",
        ]

        # Ensure all feature columns exist
        missing = [c for c in self.feature_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in dataset: {missing}")

        # Replace any remaining NaNs with 0 just in case
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)

        # Standardize features (mean 0, std 1)
        self.scaler = StandardScaler()
        self.feature_matrix = self.scaler.fit_transform(self.df[self.feature_cols])

    def _find_food_index(self, food_query: str) -> int:
        """
        Find the index of the best-matching food by description.
        Uses case-insensitive 'contains' search.
        """
        mask = self.df["description"].str.contains(food_query, case=False, na=False)
        matches = self.df[mask]

        if matches.empty:
            raise ValueError(f"No foods found matching: '{food_query}'")

        # Take the first match
        return matches.index[0]

    def find_similar_foods(self, food_query: str, top_n: int = 10) -> pd.DataFrame:
        """
        Given a food name string, return the top N most similar foods
        based on nutrient vectors (cosine similarity).
        """
        idx = self._find_food_index(food_query)
        query_vec = self.feature_matrix[idx : idx + 1]

        # Compute cosine similarity with all foods
        sims = cosine_similarity(query_vec, self.feature_matrix)[0]

        # Attach similarity scores
        df_sim = self.df.copy()
        df_sim["similarity"] = sims

        # Exclude the exact same item (idx) from results
        df_sim = df_sim[df_sim.index != idx]

        # Sort by similarity descending
        df_sim = df_sim.sort_values("similarity", ascending=False)

        # Select columns to show
        cols_to_show = [
            "description",
            "food_category_id",
            "calories_kcal",
            "sugar_g",
            "sodium_mg",
            "cholesterol_mg",
            "sat_fat_g",
            "similarity",
            "fdc_id",
        ]

        cols_to_show = [c for c in cols_to_show if c in df_sim.columns]

        return df_sim[cols_to_show].head(top_n)


if __name__ == "__main__":
    # Simple quick test
    model = NutritionSimilarityModel()
    query = "pizza"  # change to any food that exists in your dataset
    try:
        result = model.find_similar_foods(query, top_n=5)
        print(f"Top 5 foods similar to '{query}':")
        print(result)
    except ValueError as e:
        print(e)
