# Libraries
from flask import Flask, request, jsonify

app = Flask(__name__)
FOOD_VOCAB = {"chicken","rice","biryani","salad","apple","milk","bread","egg"}

@app.post("/v1/food/extract")
def extract_food():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").lower()
    tokens = [t.strip(",.!?") for t in text.split()]
    found = sorted({t for t in tokens if t in FOOD_VOCAB})
    unknown = [t for t in tokens if t not in FOOD_VOCAB]
    return jsonify({"foods": found, "unknown_tokens": unknown})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)
