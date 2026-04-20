"""
Test / Inference Script
========================
Loads a saved model and scaler, extracts features from a single
test image, combines with clinical data, and predicts.

Supports both TBT and CNN feature modes.

IMPORTANT: The clinical input must match EXACTLY the feature schema
used during training. See CLINICAL_FEATURE_NAMES below.
"""

import numpy as np
import cv2
import joblib
import os

# =========================
# ⚙️ FEATURE MODE FLAG
# =========================
USE_TBT_FEATURES = True  # Set to False to use CNN features

# =========================
# 🏥 CLINICAL FEATURE SCHEMA
# =========================
# These are the 14 clinical features used during training
# in fusion_multimodal.py (after dropping IDENTIFIER_1, LABEL,
# LABEL_NAME, SPINE_TSCORE, HIP_TSCORE, HIPNECK_TSCORE).
# ORDER MUST MATCH the training pipeline exactly.

CLINICAL_FEATURE_NAMES = [
    "SPINE_BMD",
    "HIP_BMD",
    "HIPNECK_BMD",
    "HEIGHT",
    "AGE",
    "MENOPAUSE_YEAR_CLEAN",
    "IS_MENOPAUSE",
    "BIRTH_YEAR",
    "SPINE_SCANDATE_YEAR",
    "SPINE_SCANDATE_MONTH",
    "HIP_SCANDATE_YEAR",
    "HIP_SCANDATE_MONTH",
    "HIPNECK_SCANDATE_YEAR",
    "HIPNECK_SCANDATE_MONTH",
]

# =========================
# 🚀 DEVICE (GPU/CPU)
# =========================

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD MODEL + SCALER
# =========================

model_suffix = "_tbt" if USE_TBT_FEATURES else "_cnn"

model_path = f"multimodal_model{model_suffix}.pkl"
scaler_path = f"scaler{model_suffix}.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found: {model_path}. Train first with fusion_multimodal.py")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"❌ Scaler not found: {scaler_path}. Train first with fusion_multimodal.py")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

mode_label = "TBT" if USE_TBT_FEATURES else "CNN"
print(f"✅ Model and scaler loaded ({mode_label} mode)")
print(f"[DEBUG] Scaler expects {scaler.n_features_in_} features")

# =========================
# 🖼️ EXTRACT IMAGE FEATURES
# =========================

img_path = "test_image.jpg"

if USE_TBT_FEATURES:
    # --- TBT Features ---
    from trabecular_features import extract_trabecular_features, get_feature_names

    print(f"\n[INFO] Extracting TBT features from: {img_path}")
    img_features = extract_trabecular_features(img_path).reshape(1, -1)
    print(f"✅ TBT feature shape: {img_features.shape}")

    # Print individual feature values
    names = get_feature_names()
    print(f"\n📊 TBT Feature Breakdown ({len(names)} features):")
    for name, val in zip(names, img_features[0]):
        print(f"  {name:30s} = {val:.6f}")
    print()

else:
    # --- CNN Features (ResNet50) ---
    import torchvision.models as models
    import torchvision.transforms as transforms

    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device)
    resnet.eval()
    print("✅ ResNet loaded")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("❌ Image not found")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = resnet(img)
    img_features = features.cpu().numpy().reshape(1, -1)
    print(f"✅ CNN feature shape: {img_features.shape}")

# =========================
# 🏥 CLINICAL INPUT
# =========================
# Provide all 14 clinical features in the exact order listed above.
# Example values for a test patient:

clinical_values = {
    "SPINE_BMD":              0.93,
    "HIP_BMD":                1.15,
    "HIPNECK_BMD":            0.86,
    "HEIGHT":                 160,
    "AGE":                    65,
    "MENOPAUSE_YEAR_CLEAN":   2000,
    "IS_MENOPAUSE":           1,
    "BIRTH_YEAR":             1960,
    "SPINE_SCANDATE_YEAR":    2020,
    "SPINE_SCANDATE_MONTH":   5,
    "HIP_SCANDATE_YEAR":      2020,
    "HIP_SCANDATE_MONTH":     5,
    "HIPNECK_SCANDATE_YEAR":  2020,
    "HIPNECK_SCANDATE_MONTH": 5,
}

# Build clinical feature vector in the correct order
clinical_input = np.array([[clinical_values[name] for name in CLINICAL_FEATURE_NAMES]])

print(f"\n🏥 Clinical features: {clinical_input.shape[1]} values")
for name, val in zip(CLINICAL_FEATURE_NAMES, clinical_input[0]):
    print(f"  {name:30s} = {val}")

# =========================
# 🔗 COMBINE & VALIDATE
# =========================

# Order: [image_features, clinical_features] — same as training
X = np.hstack((img_features, clinical_input))

print(f"\n[DEBUG] Image features:    {img_features.shape[1]}")
print(f"[DEBUG] Clinical features: {clinical_input.shape[1]}")
print(f"[DEBUG] Combined input:    {X.shape[1]}")
print(f"[DEBUG] Scaler expects:    {scaler.n_features_in_}")

# Safety check before scaling
if X.shape[1] != scaler.n_features_in_:
    raise ValueError(
        f"❌ Feature mismatch! Input has {X.shape[1]} features, "
        f"but scaler expects {scaler.n_features_in_}.\n"
        f"   Image features:    {img_features.shape[1]}\n"
        f"   Clinical features: {clinical_input.shape[1]}\n"
        f"   Check that CLINICAL_FEATURE_NAMES matches the training schema."
    )

X = scaler.transform(X)

print(f"✅ Final input shape: {X.shape}")

# =========================
# 🎯 PREDICT
# =========================

pred = model.predict(X)
prob = model.predict_proba(X)

labels = ["Normal", "Osteopenia", "Osteoporosis"]

print(f"\n🎯 Prediction: {labels[pred[0]]}")
print(f"📊 Probabilities:")
for label, p in zip(labels, prob[0]):
    print(f"  {label:15s} = {p:.4f}")