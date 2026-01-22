from flask import Flask, render_template, request, jsonify
import time

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    return jsonify({
        "source": "upload",
        "faces": 1,
        "dominant": "happy",
        "confidence": 0.71,
        "time": time.strftime("%H:%M:%S")
    })

@app.route("/webcam", methods=["POST"])
def webcam_frame():
    return jsonify({
        "source": "webcam",
        "faces": 1,
        "dominant": "neutral",
        "confidence": 0.55,
        "time": time.strftime("%H:%M:%S")
    })

@app.route("/llm", methods=["POST"])
def llm():
    data = request.json
    emotion = data.get("emotion", "unknown")
    confidence = data.get("confidence", 0)

    text = f"Το πρόσωπο φαίνεται {emotion} με εμπιστοσύνη {int(confidence*100)}%."
    return jsonify({"result": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
