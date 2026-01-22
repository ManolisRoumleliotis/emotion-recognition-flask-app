import os
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Απλό in-memory state (για demo)
LAST = {
    "source": None,
    "dominant": None,
    "confidence": None,
    "faces": 0,
    "ts": None,
    "raw": None,
}

@app.get("/")
def index():
    return render_template("index.html", last=LAST)

def fake_analyze():
    # TODO: εδώ μετά θα καλέσουμε το πραγματικό emotion model
    return {
        "faces": 1,
        "dominant": "happy",
        "confidence": 0.71,
        "detections": [
            {"bbox": {"x": 152, "y": 182, "w": 157, "h": 157}, "label": "happy", "confidence": 0.71}
        ],
        "inference_ms": 50.0,
    }

def update_last(source: str, result: dict):
    LAST["source"] = source
    LAST["faces"] = result.get("faces", 0)
    LAST["dominant"] = result.get("dominant")
    LAST["confidence"] = result.get("confidence")
    LAST["raw"] = result
    LAST["ts"] = datetime.now().strftime("%H:%M:%S")

@app.post("/upload")
def upload():
    f = request.files.get("image")
    if not f:
        return jsonify({"ok": False, "error": "Δεν επιλέχθηκε αρχείο."}), 400

    # (Προσωρινά δεν αναλύουμε το αρχείο. Απλώς δείχνουμε ότι το endpoint δουλεύει)
    result = fake_analyze()
    update_last("upload", result)
    return jsonify({"ok": True, "result": result})

@app.post("/webcam_frame")
def webcam_frame():
    data = request.get_json(silent=True) or {}
    b64 = data.get("image_base64")
    if not b64:
        return jsonify({"ok": False, "error": "Λείπει image_base64"}), 400

    # Προσωρινά δεν κάνουμε decode/processing. Μόνο επιβεβαίωση ροής.
    result = fake_analyze()
    update_last("webcam", result)
    return jsonify({"ok": True, "result": result})

@app.post("/llm")
def llm():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "Γράψε prompt."}), 400

    # Προσωρινό LLM text με βάση το LAST
    if not LAST["dominant"]:
        return jsonify({"ok": True, "text": "Δεν υπάρχει τελευταία ανίχνευση (upload ή webcam)."})


    text = (
        f"Η εικόνα δείχνει συναίσθημα: {LAST['dominant']} "
        f"με εμπιστοσύνη {int((LAST['confidence'] or 0)*100)}%."
    )
    return jsonify({"ok": True, "text": text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
