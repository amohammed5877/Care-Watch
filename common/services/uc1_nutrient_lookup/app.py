from flask import Flask, request, jsonify
from common.loader import load_foundation_food, load_food_nutrient, load_nutrient
app = Flask(__name__)

FF = load_foundation_food()
FN = load_food_nutrient()
N  = load_nutrient()

def find_food_fdc_ids(food):
    return FF.loc[FF["description"].str.contains(food, case=False, na=False), "fdc_id"].tolist()

def energy_nutrient_ids():
    return N.loc[N["name"].str.contains("Energy", case=False, na=False), "id"].tolist()

@app.get("/v1/nutrients/lookup")
def lookup():
    food = request.args.get("food", "")
    portion_g = float(request.args.get("portion_g", "100"))
    if not food:
        return jsonify({"error": "param 'food' required"}), 400
    ids = find_food_fdc_ids(food)
    if not ids:
        return jsonify({"food": food, "kcal": None, "note": "no match found"})
    eids = energy_nutrient_ids()
    sub = FN[(FN["fdc_id"].isin(ids)) & (FN["nutrient_id"].isin(eids))]
    if sub.empty:
        return jsonify({"food": food, "kcal": None, "note": "no energy value found"})
    amount_per_100g = float(sub.iloc[0]["amount"])
    kcal = amount_per_100g * (portion_g / 100.0)
    return jsonify({"food": food, "portion_g": portion_g, "kcal": round(kcal, 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8003, debug=True)
