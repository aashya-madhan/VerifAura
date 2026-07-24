"""
Flask app — Document Forgery Detection (ELA + CNN)

Routes:
  GET  /           — upload page
  POST /predict    — run inference, render result page
  GET  /static/... — serve uploaded images & CSS
"""

import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ela import compute_ela
from predict import predict_image

# Resolve paths relative to the project root (one level above src/)
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR   = os.path.join(BASE_DIR, "static")
UPLOAD_DIR   = os.path.join(STATIC_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTS = {"png", "jpg", "jpeg", "bmp", "tiff"}

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = "docguard-dev-secret"          # needed for flash()
app.config["UPLOAD_FOLDER"]      = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    # --- validate upload ---
    if "file" not in request.files:
        flash("No file was included in the request.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(f"Unsupported file type. Please upload: {', '.join(ALLOWED_EXTS)}", "error")
        return redirect(url_for("index"))

    # --- save original ---
    filename    = secure_filename(file.filename)
    filepath    = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # --- run prediction ---
    try:
        result = predict_image(filepath)
    except FileNotFoundError:
        flash("No trained model found. Please train the model first.", "error")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Prediction failed: {e}", "error")
        return redirect(url_for("index"))

    # --- save ELA visualization ---
    ela_filename = f"ela_{os.path.splitext(filename)[0]}.png"
    ela_path     = os.path.join(app.config["UPLOAD_FOLDER"], ela_filename)
    try:
        compute_ela(filepath).save(ela_path)
    except Exception:
        ela_filename = None   # ELA save failed — result still shown

    return render_template(
        "result.html",
        result=result,
        original_url=f"/static/uploads/{filename}",
        ela_url=f"/static/uploads/{ela_filename}" if ela_filename else None,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
