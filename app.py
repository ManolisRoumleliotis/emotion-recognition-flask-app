import os
import time
import json
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import requests
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
)

# ------------------------------------------------------------------------------
# Flask setup
# ------------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")  # set in Render env

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where uploaded images are stored (ephemeral on Render free tier; ok for demo)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# (A) Face detection (OpenCV Haar cascade - lightweight, no heavy ML deps)
# ------------------------------------------------------------------------------
# NOTE: Haar cascade ships with opencv; use its default path
HAAR_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

# ------------------------------------------------------------------------------
# (B) Emotion classification
# ------------------------------------------------------------------------------
# You had TensorFlow/Torch before; on Render free tier we keep it lightweight.
# This is a stub you can replace with your own lightweight classifier.
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def predict_emotion_stub(face_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Lightweight placeholder that returns a deterministic-ish label & confidence.
    Replace with your model inference if you have a cloud-friendly approach.
    """
    # Simple heuristic: mean intensity -> map to a label just for demo
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    m = float(np.mean(gray))
    idx = int(m) % len(EMOTION_LABELS)
    # Confidence as a bounded value
    conf = 0.55 + (float((int(m) % 45)) / 100.0)  # 0.55..0.99
    conf = max(0.50, min(0.99, conf))
    return EMOTION_LABELS[idx], conf

def detect_faces_and_emotions(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    results: List[Dict[str, Any]] = []
    for (x, y, w, h) in faces:
        face_roi = image_bgr[y:y+h, x:x+w]
        label, conf = predict_emotion_stub(face_roi)
        results.append(
            {
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "label": label,
                "confidence": float(round(conf, 2)),
            }
        )
    return results

def draw_detections(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    out = image_bgr.copy()
    for det in detections:
        x = det["bbox"]["x"]
        y = det["bbox"]["y"]
        w = det["bbox"]["w"]
        h = det["bbox"]["h"]
        label = det["label"]
        conf = det["confidence"]

        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{label} ({conf})",
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out

# ------------------------------------------------------------------------------
# (C) Optional plots (matplotlib) - safe import (does not crash if missing)
# ------------------------------------------------------------------------------
def build_emotion_distribution_plot(detections: List[Dict[str, Any]]) -> Optional[str]:
    """
    Returns a relative path to a generated PNG under static/, or None if
    matplotlib is not installed / plot cannot be generated.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend for servers
        import matplotlib.pyplot as plt
    except Exception:
        return None

    counts: Dict[str, int] = {k: 0 for k in EMOTION_LABELS}
    for det in detections:
        lbl = det.get("label")
        if lbl in counts:
            counts[lbl] += 1

    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)
    ax.bar(labels, values)
    ax.set_title("Emotion Distribution (Bar)")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plot_name = f"emotion_dist_{int(time.time())}.png"
    plot_path = os.path.join("static", "uploads", plot_name)
    abs_plot_path = os.path.join(BASE_DIR, plot_path)
    fig.savefig(abs_plot_path, dpi=150)
    plt.close(fig)

    return plot_path  # relative path usable in <img src="/static/...">

# ------------------------------------------------------------------------------
# (D) Hugging Face LLM (Inference API) - optional
# ------------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_MODEL = os.environ.get("HF_MODEL", "google/flan-t5-base").strip()  # safer default
HF_API_URL = f"https://api-inference.huggingface.com/models/{HF_MODEL}"

def call_hf_llm(prompt: str, timeout_s: int = 30) -> str:
    """
    Calls HF Inference API. Returns text response or raises RuntimeError.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set. Add it in Render Environment variables.")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"HF error {r.status_code}: {r.text}")

    data = r.json()

    # HF responses vary by model/provider
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return str(data[0]["generated_text"])
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"])
    # Fallback
    return json.dumps(data)[:2000]

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
# If you already have templates, keep them; otherwise this route still works
# but will error if templates are missing. You can switch to jsonify-only if needed.

@app.get("/")
def index():
    """
    Expects templates/index.html. If you don't have it, create it or change to jsonify.
    """
    return render_template("index.html")

@app.post("/upload")
def upload():
    """
    Upload an image, detect faces/emotions, draw boxes, and render a results page.
    Expects templates/result.html (optional). If missing, returns JSON.
    """
    if "image" not in request.files:
        flash("No file part 'image' found.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if not file or file.filename.strip() == "":
        flash("No selected file.")
        return redirect(url_for("index"))

    # Read image into OpenCV
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        flash("Invalid image format.")
        return redirect(url_for("index"))

    detections = detect_faces_and_emotions(image_bgr)
    annotated = draw_detections(image_bgr, detections)

    out_name = f"annotated_{int(time.time())}.jpg"
    out_rel = os.path.join("static", "uploads", out_name)
    out_abs = os.path.join(BASE_DIR, out_rel)
    cv2.imwrite(out_abs, annotated)

    plot_rel = build_emotion_distribution_plot(detections)

    # If you have templates, render them. Otherwise return JSON.
    try:
        return render_template(
            "result.html",
            image_path="/" + out_rel.replace("\\", "/"),
            detections=detections,
            plot_path=("/" + plot_rel.replace("\\", "/")) if plot_rel else None,
        )
    except Exception:
        return jsonify(
            {
                "image_path": "/" + out_rel.replace("\\", "/"),
                "detections": detections,
                "plot_path": ("/" + plot_rel.replace("\\", "/")) if plot_rel else None,
                "note": "Templates missing; returning JSON.",
            }
        )

@app.post("/api/detect")
def api_detect():
    """
    JSON API variant: send an image file; returns detections only.
    """
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Invalid image"}), 400

    detections = detect_faces_and_emotions(image_bgr)
    return jsonify({"detections": detections})

@app.post("/api/llm")
def api_llm():
    """
    LLM endpoint: given last detection summary + user prompt, returns generated text.
    """
    body = request.get_json(silent=True) or {}
    user_prompt = str(body.get("prompt", "")).strip()
    last_detection = body.get("last_detection", None)

    if not user_prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    # Build a disciplined prompt (works better for instruction models)
    context = ""
    if last_detection:
        context = f"Emotion detection result (JSON): {json.dumps(last_detection, ensure_ascii=False)}\n"

    final_prompt = (
        "You are an assistant that writes a short, neutral description of what is visible.\n"
        "If an emotion label is provided, mention it cautiously (e.g., 'appears' or 'seems').\n"
        "Keep it 1-2 sentences.\n\n"
        f"{context}"
        f"User request: {user_prompt}\n"
        "Answer:"
    )

    try:
        text = call_hf_llm(final_prompt)
        return jsonify({"text": text, "model": HF_MODEL})
    except Exception as e:
        return jsonify({"error": str(e), "model": HF_MODEL}), 500

@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})

# ------------------------------------------------------------------------------
# Local development only (Gunicorn on Render will NOT use this block)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Local debug server
    app.run(host="127.0.0.1", port=5001, debug=True)
import os
import time
import json
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import requests
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
)

# ------------------------------------------------------------------------------
# Flask setup
# ------------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")  # set in Render env

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where uploaded images are stored (ephemeral on Render free tier; ok for demo)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# (A) Face detection (OpenCV Haar cascade - lightweight, no heavy ML deps)
# ------------------------------------------------------------------------------
# NOTE: Haar cascade ships with opencv; use its default path
HAAR_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

# ------------------------------------------------------------------------------
# (B) Emotion classification
# ------------------------------------------------------------------------------
# You had TensorFlow/Torch before; on Render free tier we keep it lightweight.
# This is a stub you can replace with your own lightweight classifier.
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def predict_emotion_stub(face_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Lightweight placeholder that returns a deterministic-ish label & confidence.
    Replace with your model inference if you have a cloud-friendly approach.
    """
    # Simple heuristic: mean intensity -> map to a label just for demo
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    m = float(np.mean(gray))
    idx = int(m) % len(EMOTION_LABELS)
    # Confidence as a bounded value
    conf = 0.55 + (float((int(m) % 45)) / 100.0)  # 0.55..0.99
    conf = max(0.50, min(0.99, conf))
    return EMOTION_LABELS[idx], conf

def detect_faces_and_emotions(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    results: List[Dict[str, Any]] = []
    for (x, y, w, h) in faces:
        face_roi = image_bgr[y:y+h, x:x+w]
        label, conf = predict_emotion_stub(face_roi)
        results.append(
            {
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "label": label,
                "confidence": float(round(conf, 2)),
            }
        )
    return results

def draw_detections(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    out = image_bgr.copy()
    for det in detections:
        x = det["bbox"]["x"]
        y = det["bbox"]["y"]
        w = det["bbox"]["w"]
        h = det["bbox"]["h"]
        label = det["label"]
        conf = det["confidence"]

        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{label} ({conf})",
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out

# ------------------------------------------------------------------------------
# (C) Optional plots (matplotlib) - safe import (does not crash if missing)
# ------------------------------------------------------------------------------
def build_emotion_distribution_plot(detections: List[Dict[str, Any]]) -> Optional[str]:
    """
    Returns a relative path to a generated PNG under static/, or None if
    matplotlib is not installed / plot cannot be generated.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend for servers
        import matplotlib.pyplot as plt
    except Exception:
        return None

    counts: Dict[str, int] = {k: 0 for k in EMOTION_LABELS}
    for det in detections:
        lbl = det.get("label")
        if lbl in counts:
            counts[lbl] += 1

    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)
    ax.bar(labels, values)
    ax.set_title("Emotion Distribution (Bar)")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plot_name = f"emotion_dist_{int(time.time())}.png"
    plot_path = os.path.join("static", "uploads", plot_name)
    abs_plot_path = os.path.join(BASE_DIR, plot_path)
    fig.savefig(abs_plot_path, dpi=150)
    plt.close(fig)

    return plot_path  # relative path usable in <img src="/static/...">

# ------------------------------------------------------------------------------
# (D) Hugging Face LLM (Inference API) - optional
# ------------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_MODEL = os.environ.get("HF_MODEL", "google/flan-t5-base").strip()  # safer default
HF_API_URL = f"https://api-inference.huggingface.com/models/{HF_MODEL}"

def call_hf_llm(prompt: str, timeout_s: int = 30) -> str:
    """
    Calls HF Inference API. Returns text response or raises RuntimeError.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set. Add it in Render Environment variables.")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"HF error {r.status_code}: {r.text}")

    data = r.json()

    # HF responses vary by model/provider
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return str(data[0]["generated_text"])
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"])
    # Fallback
    return json.dumps(data)[:2000]

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
# If you already have templates, keep them; otherwise this route still works
# but will error if templates are missing. You can switch to jsonify-only if needed.

@app.get("/")
def index():
    """
    Expects templates/index.html. If you don't have it, create it or change to jsonify.
    """
    return render_template("index.html")

@app.post("/upload")
def upload():
    """
    Upload an image, detect faces/emotions, draw boxes, and render a results page.
    Expects templates/result.html (optional). If missing, returns JSON.
    """
    if "image" not in request.files:
        flash("No file part 'image' found.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if not file or file.filename.strip() == "":
        flash("No selected file.")
        return redirect(url_for("index"))

    # Read image into OpenCV
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        flash("Invalid image format.")
        return redirect(url_for("index"))

    detections = detect_faces_and_emotions(image_bgr)
    annotated = draw_detections(image_bgr, detections)

    out_name = f"annotated_{int(time.time())}.jpg"
    out_rel = os.path.join("static", "uploads", out_name)
    out_abs = os.path.join(BASE_DIR, out_rel)
    cv2.imwrite(out_abs, annotated)

    plot_rel = build_emotion_distribution_plot(detections)

    # If you have templates, render them. Otherwise return JSON.
    try:
        return render_template(
            "result.html",
            image_path="/" + out_rel.replace("\\", "/"),
            detections=detections,
            plot_path=("/" + plot_rel.replace("\\", "/")) if plot_rel else None,
        )
    except Exception:
        return jsonify(
            {
                "image_path": "/" + out_rel.replace("\\", "/"),
                "detections": detections,
                "plot_path": ("/" + plot_rel.replace("\\", "/")) if plot_rel else None,
                "note": "Templates missing; returning JSON.",
            }
        )

@app.post("/api/detect")
def api_detect():
    """
    JSON API variant: send an image file; returns detections only.
    """
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Invalid image"}), 400

    detections = detect_faces_and_emotions(image_bgr)
    return jsonify({"detections": detections})

@app.post("/api/llm")
def api_llm():
    """
    LLM endpoint: given last detection summary + user prompt, returns generated text.
    """
    body = request.get_json(silent=True) or {}
    user_prompt = str(body.get("prompt", "")).strip()
    last_detection = body.get("last_detection", None)

    if not user_prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    # Build a disciplined prompt (works better for instruction models)
    context = ""
    if last_detection:
        context = f"Emotion detection result (JSON): {json.dumps(last_detection, ensure_ascii=False)}\n"

    final_prompt = (
        "You are an assistant that writes a short, neutral description of what is visible.\n"
        "If an emotion label is provided, mention it cautiously (e.g., 'appears' or 'seems').\n"
        "Keep it 1-2 sentences.\n\n"
        f"{context}"
        f"User request: {user_prompt}\n"
        "Answer:"
    )

    try:
        text = call_hf_llm(final_prompt)
        return jsonify({"text": text, "model": HF_MODEL})
    except Exception as e:
        return jsonify({"error": str(e), "model": HF_MODEL}), 500

@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})

# ------------------------------------------------------------------------------
# Local development only (Gunicorn on Render will NOT use this block)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Local debug server
    app.run(host="127.0.0.1", port=5001, debug=True)
