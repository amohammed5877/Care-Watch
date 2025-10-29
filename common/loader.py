import pandas as pd
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[1] / "datasets"

def load_foundation_food(): return pd.read_csv(DATA_DIR / "foundation_food.csv")
def load_food_nutrient():   return pd.read_csv(DATA_DIR / "food_nutrient.csv")
def load_nutrient():        return pd.read_csv(DATA_DIR / "nutrient.csv")
