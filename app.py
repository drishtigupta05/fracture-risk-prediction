"""
Bone Density Classification - Flask Web Application
=====================================================
Refactored with clean separation of concerns:
    - model_service.py  : Model loading, inference, clinical vectors
    - preprocess_service.py : Preprocessing visualization
    - dxa_validator.py  : DXA scan validation (in parent dir)

All API responses follow a consistent format:
    Success: {"success": true, "prediction": ..., "confidence": ..., ...}
    Error:   {"success": false, "error": "..."}

Endpoints:
    GET  /                                   - Main page
    POST /predict                            - Run prediction
    GET  /preprocessing/<subdir>/<filename>  - Serve preprocessing images
"""

import os
import sys
import uuid
import traceback
from flask import Flask, request, jsonify, render_template, send_from_directory

# ---------------------------------------------------------------------------
# Path setup — add parent directory for project module imports
# ---------------------------------------------------------------------------
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PARENT_DIR)

# ---------------------------------------------------------------------------
# Service imports (separated for clean architecture)
# ---------------------------------------------------------------------------
from model_service import load_model, run_inference
from preprocess_service import run_preprocessing, get_preprocessing_dir
from dxa_validator import validate_dxa_scan
from trabecular_features import extract_trabecular_features


# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Load model at startup
print("[APP] Starting up...")
load_model()
print("[APP] Ready to serve requests.")


# =============================================================================
# HELPERS
# =============================================================================

def allowed_file(filename):
    """Check if the file has an allowed image extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def error_response(message, status_code=400):
    """Build a standardized error JSON response."""
    return jsonify({
        "success": False,
        "error": message
    }), status_code


def success_response(data):
    """Build a standardized success JSON response."""
    data["success"] = True
    return jsonify(data)


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction request.

    Expects multipart form data:
        - file: DXA scan image (PNG/JPG)
        - age: Patient age in years (number)
        - height: Patient height in cm (number)
        - is_menopausal: "1" or "0"

    Returns standardized JSON:
        Success: {success, prediction, confidence, probabilities, preprocessing, clinical_inputs}
        Error:   {success: false, error: "..."}
    """

    # ------------------------------------------------------------------
    # 1. Validate file upload
    # ------------------------------------------------------------------
    if "file" not in request.files:
        return error_response("No file uploaded. Please select a DXA scan image.")

    file = request.files["file"]
    if file.filename == "":
        return error_response("No file selected. Please choose a DXA scan image.")

    if not allowed_file(file.filename):
        return error_response("Invalid file type. Please upload a PNG or JPEG image.")

    # ------------------------------------------------------------------
    # 2. Validate clinical inputs
    # ------------------------------------------------------------------
    try:
        height = float(request.form.get("height", 160))
        age = float(request.form.get("age", 60))
        is_menopausal = int(request.form.get("is_menopausal", 0))
    except (ValueError, TypeError):
        return error_response("Invalid clinical data. Please enter valid numbers.")

    if not (50 <= height <= 250):
        return error_response("Height must be between 50 and 250 cm.")
    if not (1 <= age <= 120):
        return error_response("Age must be between 1 and 120 years.")

    # ------------------------------------------------------------------
    # 3. Save uploaded file with unique name
    # ------------------------------------------------------------------
    unique_id = uuid.uuid4().hex[:8]
    ext = os.path.splitext(file.filename)[1].lower()
    safe_filename = f"upload_{unique_id}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, safe_filename)

    # Read bytes BEFORE saving so CNN extraction can use them
    image_bytes = file.read()
    with open(filepath, "wb") as fout:
        fout.write(image_bytes)

    print(f"[PREDICT] New request: {file.filename} -> {safe_filename}")
    print(f"[PREDICT] Clinical: age={age}, height={height}, menopausal={is_menopausal}")

    # ------------------------------------------------------------------
    # 4. Validate DXA scan
    # ------------------------------------------------------------------
    is_valid, validation_msg = validate_dxa_scan(filepath)
    if not is_valid:
        try:
            os.remove(filepath)
        except OSError:
            pass
        return error_response(
            f"Invalid input: Please upload a valid DXA scan image. {validation_msg}",
            400
        )

    # ------------------------------------------------------------------
    # 5. Run preprocessing visualization
    # ------------------------------------------------------------------
    try:
        preprocess_data = run_preprocessing(filepath)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return error_response(f"Preprocessing failed: {str(e)}", 500)

    # ------------------------------------------------------------------
    # 6. Extract TBT features
    # ------------------------------------------------------------------
    try:
        img_features = extract_trabecular_features(filepath)
        print(f"[PREDICT] TBT features: {img_features.shape}")
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return error_response(f"Feature extraction failed: {str(e)}", 500)

    # ------------------------------------------------------------------
    # 7. Run inference
    # ------------------------------------------------------------------
    try:
        result = run_inference(img_features, height, age, is_menopausal,
                               image_bytes=image_bytes)
    except ValueError as e:
        print(f"[ERROR] Inference error: {e}")
        return error_response(str(e), 500)
    except Exception as e:
        print(f"[ERROR] Unexpected inference error: {e}")
        traceback.print_exc()
        return error_response("Prediction failed due to an internal error.", 500)

    print(f"[PREDICT] Result: {result['prediction']} (confidence: {result['confidence']})")

    # ------------------------------------------------------------------
    # 8. Return standardized success response
    # ------------------------------------------------------------------
    return success_response({
        "prediction": result["prediction"],
        "prediction_index": result["prediction_index"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "fracture_risk": result["fracture_risk"],
        "risk_color": result["risk_color"],
        "risk_score": result["risk_score"],
        "preprocessing_images": preprocess_data,
        "clinical_inputs": {
            "height": height,
            "age": age,
            "is_menopausal": bool(is_menopausal),
        }
    })


@app.route("/preprocessing/<subdir>/<filename>")
def serve_preprocessing(subdir, filename):
    """Serve preprocessing output images."""
    directory = get_preprocessing_dir(subdir)
    return send_from_directory(directory, filename)


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return error_response("File too large. Maximum upload size is 16 MB.", 413)


@app.errorhandler(500)
def internal_error(e):
    """Handle unexpected server errors."""
    return error_response("An internal server error occurred.", 500)


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    print("")
    print("=" * 60)
    print("  BoneScan AI - Bone Density Classification")
    print("  Model: Honest (CNN+TBT, no BMD leakage) — 93.75% test accuracy")
    print("=" * 60)
    print("")

    app.run(debug=True, host="0.0.0.0", port=5000)
