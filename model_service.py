"""
Model Service Module
=====================
Encapsulates all model-related operations:
    - Model and scaler loading (honest model: CNN+TBT, no BMD leakage)
    - Clinical feature vector construction (non-BMD only)
    - CNN feature extraction via ResNet50
    - Feature combination and inference

Separated from routes for clean architecture.
"""

import os
import json
import numpy as np
import joblib
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

PARENT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR    = os.path.join(PARENT_DIR, "improved_model")

# Honest leakage-free model (CNN+TBT+non-BMD clinical)
MODEL_PATH   = os.path.join(MODEL_DIR, "honest_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "honest_scaler.pkl")
CONFIG_PATH  = os.path.join(MODEL_DIR, "honest_config.json")

# Fallback: legacy TBT-only model (kept untouched)
LEGACY_MODEL_PATH  = os.path.join(PARENT_DIR, "multimodal_model_tbt.pkl")
LEGACY_SCALER_PATH = os.path.join(PARENT_DIR, "scaler_tbt.pkl")

# Class labels (match training order)
LABELS = ["Normal", "Osteopenia", "Osteoporosis"]

# Honest clinical features: HEIGHT + AGE + IS_MENOPAUSE + BIRTH_YEAR +
# MENOPAUSE_YEAR_CLEAN + 6 scan-date fields  (NO BMD, NO T-SCORES)
CLINICAL_FEATURE_NAMES = [
    "HEIGHT", "AGE",
    "MENOPAUSE_YEAR_CLEAN", "IS_MENOPAUSE", "BIRTH_YEAR",
    "SPINE_SCANDATE_YEAR",  "SPINE_SCANDATE_MONTH",
    "HIP_SCANDATE_YEAR",    "HIP_SCANDATE_MONTH",
    "HIPNECK_SCANDATE_YEAR","HIPNECK_SCANDATE_MONTH",
]


# =============================================================================
# MODEL LOADER (singleton pattern)
# =============================================================================

_model = None
_scaler = None
_schema = None


def load_model():
    """Load model, scaler, and config. Called once at startup."""
    global _model, _scaler, _schema

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    _model  = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            _schema = json.load(f)
    else:
        _schema = {}

    print(f"[MODEL] Honest model loaded. Expects {_scaler.n_features_in_} features "
          f"(CNN=2048 + TBT=37 + Clinical=11, no BMD).")
    return _model, _scaler, _schema


def get_model():
    """Get the loaded model (lazy-load if needed)."""
    if _model is None:
        load_model()
    return _model


def get_scaler():
    """Get the loaded scaler (lazy-load if needed)."""
    if _scaler is None:
        load_model()
    return _scaler


def get_expected_features():
    """Return the number of features the scaler expects."""
    return get_scaler().n_features_in_


# =============================================================================
# CLINICAL VECTOR BUILDER  (no BMD — honest model)
# =============================================================================

def build_clinical_vector(height, age, is_menopausal):
    """
    Build the 11-element clinical feature vector from 3 user inputs.
    BMD/T-score features are intentionally excluded (they define the label).

    Order must match the training pipeline in improved_model/train_honest.py:
        HEIGHT, AGE, MENOPAUSE_YEAR_CLEAN, IS_MENOPAUSE, BIRTH_YEAR,
        SPINE_SCANDATE_YEAR, SPINE_SCANDATE_MONTH,
        HIP_SCANDATE_YEAR,   HIP_SCANDATE_MONTH,
        HIPNECK_SCANDATE_YEAR, HIPNECK_SCANDATE_MONTH

    Args:
        height (float): Patient height in cm.
        age (float): Patient age in years.
        is_menopausal (int): 1 if menopausal, 0 otherwise.

    Returns:
        np.ndarray: Shape (1, 11)
    """
    now = datetime.now()
    current_year  = now.year
    current_month = now.month
    birth_year    = current_year - int(age)
    menopause_year = (birth_year + 50) if is_menopausal else 0

    return np.array([[
        float(height),          # HEIGHT
        float(age),             # AGE
        float(menopause_year),  # MENOPAUSE_YEAR_CLEAN
        float(is_menopausal),   # IS_MENOPAUSE
        float(birth_year),      # BIRTH_YEAR
        float(current_year),    # SPINE_SCANDATE_YEAR
        float(current_month),   # SPINE_SCANDATE_MONTH
        float(current_year),    # HIP_SCANDATE_YEAR
        float(current_month),   # HIP_SCANDATE_MONTH
        float(current_year),    # HIPNECK_SCANDATE_YEAR
        float(current_month),   # HIPNECK_SCANDATE_MONTH
    ]])


# =============================================================================
# FRACTURE RISK ASSESSMENT
# =============================================================================

# Mapping: prediction -> fracture risk level
RISK_MAP = {
    "Normal":       {"level": "Low Risk",      "color": "green",  "weight": 0.0},
    "Osteopenia":   {"level": "Moderate Risk",  "color": "amber",  "weight": 0.5},
    "Osteoporosis": {"level": "High Risk",      "color": "red",    "weight": 1.0},
}

# Weights for computing continuous risk score from class probabilities
# Normal contributes 0, Osteopenia contributes 0.5, Osteoporosis contributes 1.0
RISK_WEIGHTS = [0.0, 0.5, 1.0]


def compute_fracture_risk(prediction, probabilities):
    """
    Compute fracture risk assessment from model output.

    Args:
        prediction (str): Class label ("Normal", "Osteopenia", "Osteoporosis").
        probabilities (dict): {label: probability, ...}

    Returns:
        dict: {
            "fracture_risk": "Low Risk" | "Moderate Risk" | "High Risk",
            "risk_color": "green" | "amber" | "red",
            "risk_score": float (0.0 to 1.0, continuous risk estimate)
        }
    """
    risk_info = RISK_MAP.get(prediction, RISK_MAP["Normal"])

    # Compute weighted risk score from all class probabilities
    # risk_score = P(Normal)*0 + P(Osteopenia)*0.5 + P(Osteoporosis)*1.0
    risk_score = sum(
        probabilities.get(label, 0.0) * weight
        for label, weight in zip(LABELS, RISK_WEIGHTS)
    )

    return {
        "fracture_risk": risk_info["level"],
        "risk_color": risk_info["color"],
        "risk_score": round(risk_score, 4),
    }


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(tbt_features, height, age, is_menopausal, image_bytes=None):
    """
    Run the full inference pipeline (honest model: CNN+TBT+clinical, no BMD).

    Args:
        tbt_features (np.ndarray): TBT feature vector, shape (37,) or (1, 37).
        height (float): Patient height in cm.
        age (float): Patient age in years.
        is_menopausal (int): 1 if menopausal, 0 if not.
        image_bytes (bytes): Raw image bytes for CNN feature extraction.

    Returns:
        dict with prediction, confidence, probabilities, fracture_risk,
              risk_color, risk_score.
    """
    model  = get_model()
    scaler = get_scaler()

    # Step 1: TBT → 2D
    if tbt_features.ndim == 1:
        tbt_features = tbt_features.reshape(1, -1)

    # Step 2: Extract CNN features (2048-d)
    if image_bytes is not None:
        try:
            from cnn_service import extract_cnn_features
            cnn_vec = extract_cnn_features(image_bytes=image_bytes).reshape(1, -1)
        except Exception as cnn_err:
            print(f"[WARN] CNN extraction failed ({cnn_err}). Using zero vector.")
            cnn_vec = np.zeros((1, 2048))
    else:
        cnn_vec = np.zeros((1, 2048))

    # Step 3: Non-BMD clinical vector (11-d)
    clinical = build_clinical_vector(height, age, is_menopausal)

    # Step 4: Concatenate CNN(2048) + TBT(37) + Clinical(11) = 2096
    X = np.hstack((cnn_vec, tbt_features, clinical))

    # Safety check
    expected = scaler.n_features_in_
    if X.shape[1] != expected:
        raise ValueError(
            f"Feature mismatch: got {X.shape[1]}, expected {expected}. "
            f"CNN={cnn_vec.shape[1]}, TBT={tbt_features.shape[1]}, "
            f"Clinical={clinical.shape[1]}"
        )

    # Step 5: Scale and predict
    X_scaled       = scaler.transform(X)
    prediction_idx = int(model.predict(X_scaled)[0])
    probabilities  = model.predict_proba(X_scaled)[0]
    confidence     = float(probabilities[prediction_idx])

    prob_dict = {
        label: round(float(prob), 4)
        for label, prob in zip(LABELS, probabilities)
    }

    risk = compute_fracture_risk(LABELS[prediction_idx], prob_dict)

    return {
        "prediction":       LABELS[prediction_idx],
        "prediction_index": prediction_idx,
        "confidence":       round(confidence, 4),
        "probabilities":    prob_dict,
        "fracture_risk":    risk["fracture_risk"],
        "risk_color":       risk["risk_color"],
        "risk_score":       risk["risk_score"],
    }

