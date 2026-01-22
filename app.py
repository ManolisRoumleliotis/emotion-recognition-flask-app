import os
import io
import time
import base64
from typing import Dict, Any, List, Optional, Tuple

import requests
import numpy as np
import cv2

from flask import Flask, request, jsonify, render_template, redirect, url_for

app = Flask(__name__)

# -----------------------------
# Config
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "nateraw/fer")  # facial emotion recognition
HF_TEXT_MODEL = os.getenv("HF_TEXT_MODEL", "")  # προαιρετικό (text generation / chat)
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"

# Κρατάμε το τελευταίο detection για να το χρησιμοποιεί το LLM panel
LAST_DETECTION: Dict[str, Any] = {
    "source": None,      # "upload" or "webcam"
    "faces": 0,
    "dominant": None,
    "confidence": None,
    "ts": None,
    "details": None,     # raw scores
}

# Ιστορικό για charts (αν το UI σου το χρησιμοποιεί)
HISTORY: List[Dict[str, Any]] = []


# -----------------------------
# Helpers
# -----------------------------
def _hf_headers() -> Dict[str, str]:
    if not HF_TOKEN:
        return {}
    return {"Authorization": f"Bearer {HF_TOKEN}"}


def decode_data_url(data_url: str) -> bytes:
    """
    data:image/jpeg;base64,.... -> raw bytes
    """
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    _, b64 = data_url.split(",", 1)
    return base64.b64decode(b64)


def read_uploaded_file_to_bgr(file_storage) -> np.ndarray:
    data = file_storage.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Δεν μπόρεσα να διαβάσω την εικόνα.")
    return img


def bgr_to_png_data_url(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def detect_faces_opencv(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of (x, y, w, h)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def hf_emotion_scores(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Hugging Face image classification: returns list of {label, score}
    """
    if not HF_TOKEN:
        raise RuntimeError("Λείπει το HF_TOKEN (Hugging Face access token).")

    r = requests.post(
        HF_API_URL,
        headers=_hf_headers(),
        data=image_bytes,
        timeout=60,
    )
    if r.status_code != 200:
        # Πολλές φορές το HF επιστρέφει JSON με error
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Hugging Face error ({r.status_code}): {err}")

    out = r.json()
    # expected: list of dicts
    if isinstance(out, dict) and "error" in out:
        raise RuntimeError(f"Hugging Face error: {out['error']}")
    if not isinstance(out, list):
        raise RuntimeError(f"Unexpected HF response: {out}")

    return out


def pick_dominant(scores: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float]]:
    if not scores:
        return None, None
    best = max(scores, key=lambda x: x.get("score", 0))
    return best.get("label"), float(best.get("score", 0.0))


def update_last_detection(source: str, faces: int, dominant: Optional[str], confidence: Optional[float], details: Any):
    global LAST_DETECTION, HISTORY
    LAST_DETECTION = {
        "source": source,
        "faces": faces,
        "dominant": dominant,
        "confidence": confidence,
        "ts": time.strftime("%H:%M:%S"),
        "details": details,
    }
    HISTORY.append({
        "t": time.time(),
        "dominant": dominant,
        "confidence": confidence,
        "faces": faces,
        "source": source,
    })
    # κρατάμε τελευταίες 200 εγγραφές
    HISTORY = HISTORY[-200:]


def annotate_faces(img_bgr: np.ndarray, faces: List[Tuple[int, int, int, int]], label: Optional[str], conf: Optional[float]) -> np.ndarray:
    out = img_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if label is not None and conf is not None:
            cv2.putText(
                out,
                f"{label} ({conf:.2f})",
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    return out


def simple_llm_response(prompt: str) -> str:
    """
    Χωρίς εξωτερικό LLM: δημιουργεί περιγραφή με βάση το τελευταίο detection.
    """
    d = LAST_DETECTION
    if not d.get("dominant"):
        return "Δεν υπάρχει διαθέσιμη τελευταία ανίχνευση. Κάνε πρώτα Upload ή Capture & Analyze."

    emo = d["dominant"]
    conf = d["confidence"]
    src = d["source"]
    faces = d["faces"]

    base = (
        f"Τελευταία ανίχνευση από {src}. Εντοπίστηκαν {faces} πρόσωπο(α). "
        f"Κυρίαρχο συναίσθημα: {emo} με εμπιστοσύνη {conf:.0%}."
    )

    if prompt and prompt.strip():
        return base + " " + f"Με βάση το prompt σου: {prompt.strip()}"
    return base


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    # Αν έχεις templates/index.html θα το αποδώσει.
    # Αν δεν υπάρχει template, τουλάχιστον δεν θα σκάσει.
    try:
        return render_template("index.html", last=LAST_DETECTION)
    except Exception:
        return "OK - Flask is running on Render", 200


@app.route("/upload", methods=["POST"])
def upload():
    """
    Supports:
    - normal form submit (returns page)
    - fetch/AJAX (returns JSON)
    """
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest" or \
              "application/json" in (request.headers.get("Accept") or "")

    if "file" not in request.files:
        msg = "Δεν επιλέχθηκε αρχείο."
        if is_ajax:
            return jsonify({"ok": False, "error": msg}), 400
        return redirect(url_for("index"))

    f = request.files["file"]
    if not f or f.filename == "":
        msg = "Δεν επιλέχθηκε αρχείο."
        if is_ajax:
            return jsonify({"ok": False, "error": msg}), 400
        return redirect(url_for("index"))

    try:
        # Διαβάζουμε την εικόνα
        img_bgr = read_uploaded_file_to_bgr(f)

        # Face detect
        faces = detect_faces_opencv(img_bgr)

        # Emotion scores via HF
        # Ξανα-παίρνουμε bytes για HF (γιατί το file stream έχει ήδη διαβαστεί)
        _, buf = cv2.imencode(".jpg", img_bgr)
        scores = hf_emotion_scores(buf.tobytes())
        dominant, conf = pick_dominant(scores)

        # Annotated image
        annotated = annotate_faces(img_bgr, faces, dominant, conf)
        annotated_url = bgr_to_png_data_url(annotated)

        update_last_detection("upload", len(faces), dominant, conf, scores)

        payload = {
            "ok": True,
            "faces": len(faces),
            "dominant": dominant,
            "confidence": conf,
            "details": scores,
            "image": annotated_url,
            "ts": LAST_DETECTION["ts"],
        }

        if is_ajax:
            return jsonify(payload)

        # fallback: render page with result
        try:
            return render_template("index.html", last=LAST_DETECTION, upload_result=payload)
        except Exception:
            return jsonify(payload)

    except Exception as e:
        msg = str(e)
        if is_ajax:
            return jsonify({"ok": False, "error": msg}), 500
        return f"Internal Server Error: {msg}", 500


@app.route("/webcam/analyze", methods=["POST"])
def webcam_analyze():
    """
    Expects JSON:
    { "image": "data:image/jpeg;base64,...." }
    Returns JSON for UI
    """
    try:
        data = request.get_json(force=True, silent=False)
        data_url = data.get("image")
        if not data_url:
            return jsonify({"ok": False, "error": "Λείπει το image (data URL)."}), 400

        img_bytes = decode_data_url(data_url)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"ok": False, "error": "Δεν μπόρεσα να διαβάσω το frame."}), 400

        t0 = time.time()

        faces = detect_faces_opencv(img_bgr)

        # Emotion scores via HF
        scores = hf_emotion_scores(img_bytes)
        dominant, conf = pick_dominant(scores)

        annotated = annotate_faces(img_bgr, faces, dominant, conf)
        annotated_url = bgr_to_png_data_url(annotated)

        ms = (time.time() - t0) * 1000.0

        update_last_detection("webcam", len(faces), dominant, conf, scores)

        return jsonify({
            "ok": True,
            "faces": len(faces),
            "dominant": dominant,
            "confidence": conf,
            "details": scores,
            "image": annotated_url,
            "inference_ms": ms,
            "ts": LAST_DETECTION["ts"],
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/llm", methods=["POST"])
def llm():
    """
    Expects JSON: { "prompt": "..." }
    Returns JSON: { ok, text }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        text = simple_llm_response(prompt)
        return jsonify({"ok": True, "text": text, "last": LAST_DETECTION})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# Render uses gunicorn -> app:app
# Το app.run ΔΕΝ χρειάζεται σε Render, αλλά δεν ενοχλεί τοπικά.
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
