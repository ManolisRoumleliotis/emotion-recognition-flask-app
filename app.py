import os
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename


app = Flask(__name__)

# --- Upload settings ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Προαιρετικό όριο (10MB)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    # Αρχική φόρτωση χωρίς αποτέλεσμα
    return render_template(
        "index.html",
        upload_result=None,
        upload_image_url=None,
        upload_error=None,
    )


@app.get("/upload")
def upload_get():
    # Αν κάποιος ανοίξει /upload απευθείας, τον πάμε πίσω στην αρχική
    return redirect(url_for("index"))


@app.post("/upload")
def upload_post():
    # 1) Έλεγχος ότι ήρθε αρχείο
    if "file" not in request.files:
        return render_template(
            "index.html",
            upload_result=None,
            upload_image_url=None,
            upload_error="Δεν βρέθηκε πεδίο αρχείου (file).",
        ), 400

    file = request.files["file"]

    # 2) Έλεγχος ότι επιλέχθηκε αρχείο
    if not file or file.filename == "":
        return render_template(
            "index.html",
            upload_result=None,
            upload_image_url=None,
            upload_error="Δεν επιλέχθηκε αρχείο.",
        ), 400

    # 3) Έλεγχος τύπου αρχείου
    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            upload_result=None,
            upload_image_url=None,
            upload_error="Μη υποστηριζόμενος τύπος αρχείου. Χρησιμοποίησε png/jpg/jpeg/webp.",
        ), 400

    # 4) Ασφαλές όνομα + μοναδικοποίηση
    original = secure_filename(file.filename)
    ext = original.rsplit(".", 1)[1].lower()
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    saved_name = f"upload_{stamp}.{ext}"

    save_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(save_path)

    # 5) Επιστρέφουμε στη σελίδα με αποτέλεσμα (χωρίς ML προς το παρόν)
    image_url = url_for("static", filename=f"uploads/{saved_name}")

    dummy_result = (
        "Upload OK. (Προσωρινό αποτέλεσμα χωρίς AI) "
        f"Αρχείο: {original}"
    )

    return render_template(
        "index.html",
        upload_result=dummy_result,
        upload_image_url=image_url,
        upload_error=None,
    )


if __name__ == "__main__":
    # Το Render/Gunicorn ΔΕΝ χρησιμοποιεί αυτό το block.
    # Χρήσιμο μόνο για local.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
