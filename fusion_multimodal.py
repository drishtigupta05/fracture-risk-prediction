"""
Multimodal Fusion Pipeline
===========================
Combines image features (TBT or CNN) with clinical tabular data,
trains an XGBoost classifier, and evaluates performance.

Controlled by USE_TBT_FEATURES flag.
"""

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc as sk_auc

import joblib

# =========================
# ⚙️ FEATURE MODE FLAG
# =========================
USE_TBT_FEATURES = True  # Set to False to use CNN features

# =========================
# 📂 LOAD DATA
# =========================

df = pd.read_csv("label_outputs/labeled_data.csv")

if USE_TBT_FEATURES:
    features = np.load("tbt_features.npy")
    paths_file = "tbt_image_paths.txt"
    print("[INFO] 🧬 Using TRABECULAR BONE TEXTURE features")
else:
    features = np.load("features.npy")
    paths_file = "image_paths.txt"
    print("[INFO] 🧠 Using CNN (ResNet50) features")

with open(paths_file, "r", encoding="utf-8") as f:
    image_paths = [line.strip() for line in f]

print(f"✅ CSV Loaded: {df.shape}")
print(f"✅ Features Loaded: {features.shape}")
print(f"✅ Paths Loaded: {len(image_paths)}")

# =========================
# 🧠 STEP 2: IMAGE FEATURE MAP
# =========================

print("\n🧠 Creating image feature map...")

image_names = [os.path.basename(p) for p in image_paths]

df_images = pd.DataFrame({
    "image_name": image_names,
    "features": list(features)
})

print("✅ Image feature map created")

# =========================
# 🔄 STEP 3: FIX IMAGE_PATHS FORMAT
# =========================

print("\n🔄 Fixing IMAGE_PATHS format...")

def parse_paths(x):
    if pd.isna(x):
        return []
    return [p.strip() for p in str(x).split("|")]

df["IMAGE_PATHS"] = df["IMAGE_PATHS"].apply(parse_paths)

print("✅ IMAGE_PATHS parsed")

# Debug sample
print("\nSample IMAGE_PATHS:")
print(df["IMAGE_PATHS"].iloc[0])

# =========================
# 📦 STEP 4: EXPLODE
# =========================

print("\n📦 Expanding multi-image rows...")

df_expanded = df.explode("IMAGE_PATHS")

df_expanded["image_name"] = df_expanded["IMAGE_PATHS"].apply(
    lambda x: os.path.basename(x) if pd.notna(x) else None
)

print(f"✅ Expanded shape: {df_expanded.shape}")

# =========================
# 🔗 STEP 5: MERGE FEATURES
# =========================

print("\n🔗 Merging with image features...")

df_merged = pd.merge(df_expanded, df_images, on="image_name")

print(f"✅ Merged shape: {df_merged.shape}")

# Check missing matches
missing = set(df_expanded["image_name"]) - set(df_images["image_name"])
print(f"⚠️ Missing images (not matched): {len(missing)}")

# =========================
# 🧬 STEP 6: AGGREGATE PER PATIENT
# =========================

print("\n🧬 Aggregating per patient...")

target_column = "LABEL"

non_feature_cols = [
    "IDENTIFIER_1",
    "IMAGE_PATHS",
    "image_name",
    "features",
    "LABEL_NAME"
]

clinical_cols = [col for col in df.columns if col not in non_feature_cols]

agg_dict = {col: "first" for col in clinical_cols}
agg_dict["features"] = lambda x: np.mean(np.vstack(x), axis=0)

df_grouped = df_merged.groupby("IDENTIFIER_1").agg(agg_dict).reset_index()

print(f"✅ Patient-level shape: {df_grouped.shape}")

# =========================
# 🎯 STEP 7: PREPARE FEATURES
# =========================

print("\n🎯 Preparing final feature matrix...")

y = df_grouped[target_column].values

drop_cols = ["IDENTIFIER_1", "LABEL", "features", "LABEL_NAME","SPINE_TSCORE","HIP_TSCORE","HIPNECK_TSCORE"]

existing_cols = [col for col in drop_cols if col in df_grouped.columns]

X_clinical = df_grouped.drop(columns=existing_cols)
X_clinical = pd.get_dummies(X_clinical)

# Log clinical feature names for debugging
print(f"\n[DEBUG] Clinical columns used ({len(X_clinical.columns)}):")
for col in X_clinical.columns:
    print(f"   - {col}")

X_image = np.vstack(df_grouped["features"].values)

X = np.hstack([X_image, X_clinical.values])

print(f"\n✅ Final feature shape: {X.shape}")
print(f"   Image features: {X_image.shape[1]}")
print(f"   Clinical features: {X_clinical.shape[1]}")

# Build feature name list for importance plot
if USE_TBT_FEATURES:
    from trabecular_features import get_feature_names
    image_feature_names = get_feature_names()
else:
    image_feature_names = [f"CNN_{i}" for i in range(X_image.shape[1])]

all_feature_names = image_feature_names + list(X_clinical.columns)

# =========================
# ⚖️ STEP 8: SPLIT
# =========================

print("\n⚖️ Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Train/Test split done")

# =========================
# 📏 STEP 9: SCALING
# =========================

print("\n📏 Scaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("✅ Scaling complete")

# =========================
# 🤖 STEP 10: MODEL TRAINING
# =========================

print("\n🤖 Training model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    tree_method="hist",
    device="cuda"
)

model.fit(X_train, y_train)

print("✅ Model training complete")

# =========================
# 📊 STEP 11: EVALUATION
# =========================

print("\n📊 Evaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))

print("\n📋 Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Osteopenia", "Osteoporosis"]
))

auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
print("\n🎯 AUC:", auc)

print("\n✅ PIPELINE COMPLETE 🚀")

# =========================
# 💾 SAVE MODEL + SCALER
# =========================

model_suffix = "_tbt" if USE_TBT_FEATURES else "_cnn"

model_path = f"multimodal_model{model_suffix}.pkl"
scaler_path = f"scaler{model_suffix}.pkl"

joblib.dump(model, model_path)
print(f"✅ Model saved as {model_path}")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved as {scaler_path}")

# Save feature schema for reproducibility
import json
schema_path = f"feature_schema{model_suffix}.json"
schema = {
    "total_features": len(all_feature_names),
    "image_features": image_feature_names,
    "clinical_features": list(X_clinical.columns),
    "all_features": all_feature_names
}
with open(schema_path, "w") as f:
    json.dump(schema, f, indent=2)
print(f"✅ Feature schema saved as {schema_path}")

# =========================
# 📁 CREATE OUTPUT FOLDER
# =========================

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

print(f"\n📁 Saving plots to: {output_dir}/")

# =========================
# 📊 CONFUSION MATRIX
# =========================

print("📊 Saving Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Normal", "Osteopenia", "Osteoporosis"],
            yticklabels=["Normal", "Osteopenia", "Osteoporosis"],
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
mode_label = "TBT" if USE_TBT_FEATURES else "CNN"
plt.title(f"Confusion Matrix ({mode_label} Features)")

cm_path = os.path.join(output_dir, f"confusion_matrix_{mode_label.lower()}.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ Saved: {cm_path}")

# =========================
# 📈 ROC CURVE
# =========================

print("📈 Saving ROC Curve...")

y_test_bin = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = sk_auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))

labels = ["Normal", "Osteopenia", "Osteoporosis"]

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"{labels[i]} (AUC = {roc_auc[i]:.2f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve — Multiclass ({mode_label} Features)")
plt.legend()

roc_path = os.path.join(output_dir, f"roc_curve_{mode_label.lower()}.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ Saved: {roc_path}")

# =========================
# 🌳 FEATURE IMPORTANCE (Top 20)
# =========================

print("🌳 Saving Feature Importance...")

importances = model.feature_importances_

# Pair names with importances
if len(all_feature_names) == len(importances):
    feat_imp = sorted(
        zip(all_feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
    top_n = min(20, len(feat_imp))
    top_names, top_vals = zip(*feat_imp[:top_n])

    plt.figure(figsize=(10, 7))
    plt.barh(range(top_n), top_vals[::-1], color="steelblue")
    plt.yticks(range(top_n), top_names[::-1], fontsize=9)
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances ({mode_label} Features)")
    plt.tight_layout()
else:
    # Fallback: numeric index plot
    plt.figure(figsize=(10, 5))
    plt.plot(importances)
    plt.title(f"Feature Importance ({mode_label} Features)")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")

fi_path = os.path.join(output_dir, f"feature_importance_{mode_label.lower()}.png")
plt.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ Saved: {fi_path}")

print("\n🎉 All plots saved successfully!")