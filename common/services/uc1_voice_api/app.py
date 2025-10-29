from flask import Flask, request, jsonify
app = Flask(__name__)

@app.post("/v1/voice/transcribe")
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "audio file missing"}), 400
    return jsonify({"text": "chicken biryani with salad", "confidence": 0.92})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
