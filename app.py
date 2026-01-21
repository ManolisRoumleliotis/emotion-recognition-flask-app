import os
import time
import base64
from datetime import datetime
from collections import Counter

import cv2
import numpy as np
import requests
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fer import FER  # pip install fer

app = Flask(__name__)

# -----------------------------
# Paths / config
# -----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CHART_FOLDER = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")

def dataurl_to_bgr(data_url: str):
    try:
        _, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None

# -----------------------------
# Emotion model (CPU)
# -----------------------------
emotion_detector = FER(mtcnn=False)

# -----------------------------
# In-memory session stats
# -----------------------------
emotion_counter = Counter()
dominant_history = []
upload_timestamps = []

# Last detection for LLM (this is what gets sent LIVE)
last_detection = {
    "context": None,          # "upload" or "webcam"
    "timestamp": None,
    "faces_detected": 0,
    "dominant": None,
    "faces": []               # list of {bbox, label, confidence}
}

# -----------------------------
# Charts
# -----------------------------
def _save_fig(fig, filename: str) -> str:
    path = os.path.join(CHART_FOLDER, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return f"charts/{filename}"  # relative to /static

def plot_bar_counts(counter: Counter):
    if not counter:
        return None
    labels = list(counter.keys())
    values = [counter[k] for k in labels]
    fig = plt.figure()
    plt.title("Emotion Distribution (Bar)")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.bar(labels, values)
    return _save_fig(fig, f"bar_{int(time.time()*1000)}.png")

def plot_line_over_time(history, timestamps):
    if not history or not timestamps or len(history) != len(timestamps):
        return None

    emotions = sorted(set(history))
    running = {e: [] for e in emotions}
    counts = {e: 0 for e in emotions}

    for lbl in history:
        counts[lbl] += 1
        for e in emotions:
            running[e].append(counts[e])

    fig = plt.figure()
    plt.title("Emotion Over Time (Line)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative count")
    for e in emotions:
        plt.plot(timestamps, running[e], marker="o", label=e)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    return _save_fig(fig, f"line_{int(time.time()*1000)}.png")

def plot_pie_dominant(counter: Counter):
    if not counter:
        return None
    labels = list(counter.keys())
    values = [counter[k] for k in labels]
    fig = plt.figure()
    plt.title("Dominant Emotions (Pie)")
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    return _save_fig(fig, f"pie_{int(time.time()*1000)}.png")

def generate_charts():
    return (
        plot_bar_counts(emotion_counter),
        plot_line_over_time(dominant_history, upload_timestamps),
        plot_pie_dominant(emotion_counter),
    )

# -----------------------------
# CV inference
# -----------------------------
def analyze_image(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = emotion_detector.detect_emotions(rgb)

    faces = []
    votes = Counter()

    for d in detections:
        x, y, w, h = d["box"]
        emotions = d["emotions"]
        label = max(emotions, key=emotions.get)
        conf = float(emotions[label])

        votes[label] += 1

        # draw bounding + label
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"{label} ({conf:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        faces.append({
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "label": label,
            "confidence": conf
        })

    dominant = votes.most_common(1)[0][0] if faces else None
    return img_bgr, faces, dominant

def update_memory(context: str, faces: list, dominant: str | None):
    last_detection["context"] = context
    last_detection["timestamp"] = stamp()
    last_detection["faces_detected"] = int(len(faces))
    last_detection["dominant"] = dominant
    last_detection["faces"] = faces

    if dominant:
        emotion_counter[dominant] += 1
        dominant_history.append(dominant)
        upload_timestamps.append(last_detection["timestamp"])

# -----------------------------
# LLM (Ollama local)
# -----------------------------
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

def build_llm_prompt(user_prompt: str) -> str:
    # Strict: only use detection data (no hallucinations)
    return f"""
Είσαι βοηθός που γράφει σύντομες περιγραφές στα ελληνικά ΜΟΝΟ από δομημένα δεδομένα ανίχνευσης.
Μην μαντεύεις το περιβάλλον ή αντικείμενα. Μην προσθέτεις πληροφορίες που δεν υπάρχουν.

ΔΕΔΟΜΕΝΑ ΑΝΙΧΝΕΥΣΗΣ:
- Πηγή: {last_detection.get("context")}
- Χρόνος: {last_detection.get("timestamp")}
- Πρόσωπα: {last_detection.get("faces_detected")}
- Κυρίαρχο συναίσθημα: {last_detection.get("dominant")}
- Αναλυτικά πρόσωπα: {last_detection.get("faces")}

PROMPT ΧΡΗΣΤΗ:
{user_prompt}

ΑΠΑΝΤΗΣΗ:
Γράψε 2–4 προτάσεις στα ελληνικά.
""".strip()

def call_ollama(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return render_template("index.html", last=last_detection)

@app.get("/upload")
def upload_page():
    return render_template("index.html", last=last_detection)

@app.post("/upload")
def upload():
    start = time.time()

    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("index.html", result="Δεν επιλέχθηκε αρχείο.", last=last_detection)

    if not allowed_file(file.filename):
        return render_template("index.html", result="Μη έγκυρος τύπος αρχείου.", last=last_detection)

    original_name = secure_filename(file.filename)
    name = f"{int(time.time()*1000)}_{original_name}"
    path = os.path.join(UPLOAD_FOLDER, name)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        return render_template("index.html", result="Αδυναμία ανάγνωσης εικόνας.", last=last_detection)

    annotated, faces, dominant = analyze_image(img)
    update_memory("upload", faces, dominant)

    boxed = f"boxed_{name}"
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, boxed), annotated)

    elapsed_ms = (time.time() - start) * 1000.0
    bar, line, pie = generate_charts()

    if not faces:
        result = f"Faces: 0 | Emotion: not detected | Inference: {elapsed_ms:.1f} ms"
    else:
        result = f"Faces: {len(faces)} | Dominant: {dominant} | Inference: {elapsed_ms:.1f} ms"

    return render_template(
        "index.html",
        filename=boxed,
        result=result,
        bar_chart=bar,
        line_chart=line,
        pie_chart=pie,
        last=last_detection
    )

@app.post("/predict_frame")
def predict_frame():
    start = time.time()
    data = request.get_json(silent=True) or {}
    frame_data = data.get("image")
    if not frame_data:
        return jsonify({"ok": False, "error": "Missing image"}), 400

    img = dataurl_to_bgr(frame_data)
    if img is None:
        return jsonify({"ok": False, "error": "Decode failed"}), 400

    annotated, faces, dominant = analyze_image(img)
    update_memory("webcam", faces, dominant)

    name = f"webcam_{int(time.time()*1000)}.jpg"
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, name), annotated)

    elapsed_ms = (time.time() - start) * 1000.0
    bar, line, pie = generate_charts()

    return jsonify({
        "ok": True,
        "faces_detected": int(len(faces)),
        "dominant": dominant,
        "elapsed_ms": elapsed_ms,
        "faces": faces,
        "annotated_url": f"/uploads/{name}",
        "charts": {
            "bar": f"/static/{bar}" if bar else None,
            "line": f"/static/{line}" if line else None,
            "pie": f"/static/{pie}" if pie else None
        },
        "last": last_detection
    })

@app.post("/llm_generate")
def llm_generate():
    data = request.get_json(silent=True) or {}
    user_prompt = (data.get("prompt") or "").strip()

    if not user_prompt:
        return jsonify({"ok": False, "error": "Γράψε ένα prompt πρώτα."}), 400

    if not last_detection.get("timestamp"):
        return jsonify({"ok": False, "error": "Κάνε πρώτα upload εικόνας ή webcam capture για να υπάρχουν δεδομένα."}), 400

    try:
        full_prompt = build_llm_prompt(user_prompt)
        text = call_ollama(full_prompt)
        return jsonify({"ok": True, "text": text})
    except requests.exceptions.ConnectionError:
        return jsonify({"ok": False, "error": "Δεν τρέχει το Ollama. Άνοιξέ το και ξαναδοκίμασε."}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": f"LLM error: {str(e)}"}), 500

@app.get("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
