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
from predict import predict_image, predict_document
from document_processor import is_document, is_image, DOCUMENT_EXTENSIONS, IMAGE_EXTENSIONS

# Resolve paths relative to the project root (one level above src/)
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR   = os.path.join(BASE_DIR, "static")
UPLOAD_DIR   = os.path.join(STATIC_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# All accepted extensions: images + documents
ALLOWED_EXTS = (
    {ext.lstrip(".") for ext in IMAGE_EXTENSIONS}
    | {ext.lstrip(".") for ext in DOCUMENT_EXTENSIONS}
)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = "docguard-dev-secret"
app.config["UPLOAD_FOLDER"]      = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB (documents can be larger)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    # ── validate upload ──────────────────────────────────────────────────────
    if "file" not in request.files:
        flash("No file was included in the request.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        supported = "PNG, JPG, JPEG, BMP, TIFF, PDF, DOCX, PPTX"
        flash(f"Unsupported file type. Please upload: {supported}", "error")
        return redirect(url_for("index"))

    # ── save original ────────────────────────────────────────────────────────
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ── route to document or image analysis ──────────────────────────────────
    try:
        if is_document(filename):
            result = predict_document(filepath, app.config["UPLOAD_FOLDER"])
            # Build URL-friendly paths for templates
            for page in result["pages"]:
                page["image_url"] = "/static/uploads/" + os.path.basename(page["image_path"])
                page["ela_url"]   = (
                    "/static/uploads/" + os.path.basename(page["ela_path"])
                    if page["ela_path"] else None
                )
            return render_template(
                "result.html",
                result=result,
                original_url=f"/static/uploads/{filename}",
                ela_url=None,  # not used for documents (per-page ELA is in result["pages"])
                is_document=True,
            )
        else:
            # Plain image — existing single-page flow
            result = predict_image(filepath)
            ela_filename = f"ela_{os.path.splitext(filename)[0]}.png"
            ela_path     = os.path.join(app.config["UPLOAD_FOLDER"], ela_filename)
            try:
                compute_ela(filepath).save(ela_path)
            except Exception:
                ela_filename = None

            return render_template(
                "result.html",
                result=result,
                original_url=f"/static/uploads/{filename}",
                ela_url=f"/static/uploads/{ela_filename}" if ela_filename else None,
                is_document=False,
            )

    except FileNotFoundError:
        flash("No trained model found. Please train the model first.", "error")
        return redirect(url_for("index"))
    except ImportError as e:
        flash(f"Missing library for this file type: {e}", "error")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Analysis failed: {e}", "error")
        return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(debug=debug, host="0.0.0.0", port=port)
