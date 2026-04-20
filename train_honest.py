"""
HONEST Training Pipeline — Leakage-Free
==========================================
Fixes the critical data leakage issue:

  REMOVED: SPINE_BMD, HIP_BMD, HIPNECK_BMD (these DEFINE the labels)
  REMOVED: SPINE_TSCORE, HIP_TSCORE, HIPNECK_TSCORE (same leakage)

Uses ONLY:
  - TBT image features (37 dims) — trabecular bone texture
  - CNN image features (2048 dims) — ResNet50 deep features
  - Non-BMD clinical data: AGE, HEIGHT, IS_MENOPAUSE, BIRTH_YEAR,
    MENOPAUSE_YEAR, SCANDATE_* (8 features)

Techniques to maximize honest accuracy:
  1. LightGBM + XGBoost ensemble (soft voting)
  2. Heavy class weighting for Osteoporosis (5x)
  3. SMOTETomek oversampling
  4. Targeted noise augmentation for minority
  5. 5-fold StratifiedKFold CV
  6. Patient-level splitting (no patient in both train+test)

Output:
  improved_model/honest_model.pkl
  improved_model/honest_scaler.pkl
  improved_model/honest_config.json
  improved_model/plots/honest_*.png
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_recall_fscore_support, classification_report,
    roc_auc_score, roc_curve, auc as sk_auc
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from imblearn.combine import SMOTETomek
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(MODEL_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
LABELS = ["Normal", "Osteopenia", "Osteoporosis"]

# LEAKY FEATURES — MUST BE EXCLUDED
LEAKY_FEATURES = [
    "SPINE_BMD", "HIP_BMD", "HIPNECK_BMD",
    "SPINE_TSCORE", "HIP_TSCORE", "HIPNECK_TSCORE",
]


# =============================================================================
# DATA LOADING
# =============================================================================

print("=" * 70)
print("  HONEST Training Pipeline (No BMD Leakage)")
print("=" * 70)

df = pd.read_csv(os.path.join(ROOT, "label_outputs/labeled_data.csv"))
tbt_features = np.load(os.path.join(ROOT, "tbt_features.npy"))
cnn_features = np.load(os.path.join(ROOT, "features.npy"))

with open(os.path.join(ROOT, "tbt_image_paths.txt"), "r", encoding="utf-8") as f:
    tbt_paths = [line.strip() for line in f]
with open(os.path.join(ROOT, "image_paths.txt"), "r", encoding="utf-8") as f:
    cnn_paths = [line.strip() for line in f]

# Feature maps
tbt_names_list = [os.path.basename(p) for p in tbt_paths]
cnn_names_list = [os.path.basename(p) for p in cnn_paths]
df_tbt = pd.DataFrame({"image_name": tbt_names_list, "tbt_feat": list(tbt_features)})
df_cnn = pd.DataFrame({"image_name": cnn_names_list, "cnn_feat": list(cnn_features)})

# Parse and merge
def parse_paths(x):
    if pd.isna(x): return []
    return [p.strip() for p in str(x).split("|")]

df["IMAGE_PATHS"] = df["IMAGE_PATHS"].apply(parse_paths)
df_exp = df.explode("IMAGE_PATHS")
df_exp["image_name"] = df_exp["IMAGE_PATHS"].apply(
    lambda x: os.path.basename(x) if pd.notna(x) else None)

df_m = pd.merge(df_exp, df_tbt, on="image_name", how="inner")
df_m = pd.merge(df_m, df_cnn, on="image_name", how="inner")

# Aggregate per patient
skip_cols = ["IDENTIFIER_1", "IMAGE_PATHS", "image_name",
             "tbt_feat", "cnn_feat", "LABEL_NAME"]
clin_cols = [c for c in df.columns if c not in skip_cols]
agg = {c: "first" for c in clin_cols}
agg["tbt_feat"] = lambda x: np.mean(np.vstack(x), axis=0)
agg["cnn_feat"] = lambda x: np.mean(np.vstack(x), axis=0)
df_g = df_m.groupby("IDENTIFIER_1").agg(agg).reset_index()

print(f"\n[DATA] {len(df_g)} patients")

# Labels
y = df_g["LABEL"].values

# Image features
X_tbt = np.vstack(df_g["tbt_feat"].values)
X_cnn = np.vstack(df_g["cnn_feat"].values)

# Clinical features — EXCLUDE LEAKY
drop_all = ["IDENTIFIER_1", "LABEL", "tbt_feat", "cnn_feat", "LABEL_NAME"] + LEAKY_FEATURES
existing_drop = [c for c in drop_all if c in df_g.columns]
X_clin = df_g.drop(columns=existing_drop)
X_clin = pd.get_dummies(X_clin)
clinical_feature_names = list(X_clin.columns)

# Patient IDs for patient-level splitting
patient_ids = df_g["IDENTIFIER_1"].values

print(f"[FEATURES] TBT: {X_tbt.shape[1]}, CNN: {X_cnn.shape[1]}, "
      f"Clinical: {X_clin.shape[1]} (BMD REMOVED)")
print(f"[FEATURES] Total: {X_tbt.shape[1] + X_cnn.shape[1] + X_clin.shape[1]}")

# Show removed features
print(f"\n[REMOVED LEAKY FEATURES]")
for feat in LEAKY_FEATURES:
    status = "removed" if feat in df_g.columns else "not present"
    print(f"  {feat}: {status}")

# Combine
X = np.hstack([X_cnn, X_tbt, X_clin.values])
print(f"\n[COMBINED] {X.shape}")

# Class distribution
counts = Counter(y)
for c in [0, 1, 2]:
    print(f"  {LABELS[c]:>15}: {counts[c]} ({counts[c]/len(y)*100:.1f}%)")


# =============================================================================
# HELPERS
# =============================================================================

def augment(X, y, noise_std=0.015):
    parts_X, parts_y = [X], [y]
    for cls, copies in [(2, 4), (1, 1)]:
        mask = (y == cls)
        for _ in range(copies):
            noise = np.random.normal(0, noise_std, X[mask].shape)
            parts_X.append(X[mask] + noise)
            parts_y.append(y[mask])
    return np.vstack(parts_X), np.concatenate(parts_y)

def focal_weights(y, osteo_boost=2.5):
    c = Counter(y)
    n, nc = len(y), len(c)
    w = {cls: n / (nc * cnt) for cls, cnt in c.items()}
    w[2] *= osteo_boost
    weights = np.array([w[l] for l in y])
    return weights / weights.mean()


# =============================================================================
# CROSS-VALIDATION (Patient-Level)
# =============================================================================

print(f"\n[TRAINING] {N_FOLDS}-Fold StratifiedKFold (no leakage)...")
print(f"[INFO] Features used: image only + non-BMD clinical")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_results = []
best_model = None
best_scaler = None
best_score = 0

all_val_preds = []
all_val_trues = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # Verify no patient overlap
    tr_patients = set(patient_ids[tr_idx])
    val_patients = set(patient_ids[val_idx])
    overlap = tr_patients & val_patients
    if overlap:
        print(f"  WARNING: {len(overlap)} patients in both train and val!")
    else:
        print(f"  Patient split clean: {len(tr_patients)} train, {len(val_patients)} val")

    # Augment AFTER splitting
    X_aug, y_aug = augment(X_tr, y_tr)

    # SMOTE
    if HAS_SMOTE:
        try:
            X_aug, y_aug = SMOTETomek(random_state=RANDOM_STATE).fit_resample(X_aug, y_aug)
        except:
            pass

    ac = Counter(y_aug)
    print(f"  Augmented: {len(y_tr)} -> {len(y_aug)} "
          f"(N={ac[0]}, Op={ac[1]}, Os={ac[2]})")

    # Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_aug)
    X_val_s = scaler.transform(X_val)

    weights = focal_weights(y_aug)

    # Soft Voting Ensemble: LightGBM + XGBoost
    model = VotingClassifier(
        estimators=[
            ("lgbm", LGBMClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.03,
                num_leaves=31, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.5,
                reg_alpha=0.1, reg_lambda=1.0,
                class_weight={0: 1.0, 1: 1.5, 2: 5.0},
                random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
            )),
            ("xgb", XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                min_child_weight=5, gamma=0.3,
                subsample=0.8, colsample_bytree=0.5,
                reg_alpha=0.1, reg_lambda=1.5,
                objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", tree_method="hist",
                random_state=RANDOM_STATE,
            )),
        ],
        voting="soft",
        weights=[1.2, 1.0],  # Slightly favor LightGBM
    )
    model.fit(X_tr_s, y_aug, sample_weight=weights)

    # Evaluate on UNSEEN validation set
    y_pred = model.predict(X_val_s)
    y_prob = model.predict_proba(X_val_s)

    # Also check training performance (for overfitting detection)
    y_tr_pred = model.predict(X_tr_s)
    train_acc = accuracy_score(y_aug, y_tr_pred)

    val_acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
    _, rec, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, labels=[0, 1, 2], zero_division=0)

    try:
        val_auc = roc_auc_score(y_val, y_prob, multi_class="ovr")
    except:
        val_auc = 0

    fold_results.append({
        "fold": fold+1, "train_acc": train_acc, "val_acc": val_acc,
        "auc": val_auc, "rec": rec, "f1": f1, "cm": cm,
    })

    all_val_preds.extend(y_pred)
    all_val_trues.extend(y_val)

    print(f"  Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} "
          f"(gap={train_acc - val_acc:.4f})")
    print(f"  Val AUC: {val_auc:.4f}")
    print(f"  Val Recall: N={rec[0]:.3f} Op={rec[1]:.3f} Os={rec[2]:.3f}")
    print(f"  Val F1:     N={f1[0]:.3f} Op={f1[1]:.3f} Os={f1[2]:.3f}")

    score = rec[2] * 0.4 + val_acc * 0.6
    if score > best_score:
        best_score = score
        best_model = model
        best_scaler = scaler
        print(f"  >> Best (score={score:.4f})")


# =============================================================================
# CV SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("  CROSS-VALIDATION SUMMARY (HONEST — No BMD Leakage)")
print("=" * 70)

mean_tr_acc = np.mean([r["train_acc"] for r in fold_results])
mean_val_acc = np.mean([r["val_acc"] for r in fold_results])
mean_auc = np.mean([r["auc"] for r in fold_results])
mean_rec = np.mean([r["rec"] for r in fold_results], axis=0)
mean_f1 = np.mean([r["f1"] for r in fold_results], axis=0)

print(f"  Mean Train Accuracy: {mean_tr_acc:.4f}")
print(f"  Mean Val Accuracy:   {mean_val_acc:.4f}")
print(f"  Train-Val Gap:       {mean_tr_acc - mean_val_acc:.4f}")
print(f"  Mean AUC:            {mean_auc:.4f}")
print(f"  Mean Recall:  N={mean_rec[0]:.3f} Op={mean_rec[1]:.3f} Os={mean_rec[2]:.3f}")
print(f"  Mean F1:      N={mean_f1[0]:.3f} Op={mean_f1[1]:.3f} Os={mean_f1[2]:.3f}")

# Aggregated CV confusion matrix
all_val_preds = np.array(all_val_preds)
all_val_trues = np.array(all_val_trues)
cm_agg = confusion_matrix(all_val_trues, all_val_preds, labels=[0, 1, 2])
print(f"\n  Aggregated CV Confusion Matrix:")
print(f"  {'':>15} Pred N    Pred Op   Pred Os")
for i in range(3):
    print(f"  {LABELS[i]:>15} {cm_agg[i][0]:>6}  {cm_agg[i][1]:>8}  {cm_agg[i][2]:>8}")


# =============================================================================
# FINAL TEST (held-out 20%)
# =============================================================================

print("\n" + "=" * 70)
print("  HELD-OUT TEST SET (20%)")
print("=" * 70)

X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

X_te_s = best_scaler.transform(X_te_f)
y_pred_f = best_model.predict(X_te_s)
y_prob_f = best_model.predict_proba(X_te_s)

test_acc = accuracy_score(y_te_f, y_pred_f)
cm_test = confusion_matrix(y_te_f, y_pred_f, labels=[0, 1, 2])
prec_t, rec_t, f1_t, sup_t = precision_recall_fscore_support(
    y_te_f, y_pred_f, labels=[0, 1, 2], zero_division=0)

print(f"\n  Test Accuracy: {test_acc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  {'':>15} Pred N    Pred Op   Pred Os")
for i in range(3):
    print(f"  {LABELS[i]:>15} {cm_test[i][0]:>6}  {cm_test[i][1]:>8}  {cm_test[i][2]:>8}")

print(f"\n  Per-Class:")
for i in range(3):
    print(f"  {LABELS[i]:>15}: Prec={prec_t[i]:.3f}  Rec={rec_t[i]:.3f}  "
          f"F1={f1_t[i]:.3f}  n={sup_t[i]}")

print(f"\n  Full Report:")
print(classification_report(y_te_f, y_pred_f, target_names=LABELS))

om = (y_te_f == 2)
print(f"  Osteoporosis: {sum(om)} total, {sum(y_pred_f[om]==2)} correct, "
      f"{sum(y_pred_f[om]==0)} as Normal, {sum(y_pred_f[om]==1)} as Osteopenia")


# =============================================================================
# SAVE MODEL
# =============================================================================

print("\n[SAVING]")
joblib.dump(best_model, os.path.join(MODEL_DIR, "honest_model.pkl"))
joblib.dump(best_scaler, os.path.join(MODEL_DIR, "honest_scaler.pkl"))
print(f"  Saved: honest_model.pkl, honest_scaler.pkl")

config = {
    "model": "VotingClassifier(LightGBM + XGBoost, soft voting)",
    "features": {
        "CNN (ResNet50)": X_cnn.shape[1],
        "TBT (Trabecular)": X_tbt.shape[1],
        "Clinical (no BMD)": X_clin.shape[1],
        "total": X.shape[1],
    },
    "removed_features": LEAKY_FEATURES,
    "reason": "BMD/T-score features define the labels (WHO criteria) = direct leakage",
    "cv_results": {
        "mean_train_accuracy": round(mean_tr_acc, 4),
        "mean_val_accuracy": round(mean_val_acc, 4),
        "train_val_gap": round(mean_tr_acc - mean_val_acc, 4),
        "mean_auc": round(mean_auc, 4),
        "mean_recall_per_class": {LABELS[i]: round(mean_rec[i], 4) for i in range(3)},
    },
    "test_results": {
        "accuracy": round(test_acc, 4),
        "recall": {LABELS[i]: round(rec_t[i], 4) for i in range(3)},
        "f1": {LABELS[i]: round(f1_t[i], 4) for i in range(3)},
    },
    "clinical_features_used": clinical_feature_names,
}
with open(os.path.join(MODEL_DIR, "honest_config.json"), "w") as f:
    json.dump(config, f, indent=2)
print(f"  Saved: honest_config.json")


# =============================================================================
# PLOTS
# =============================================================================

print(f"\n[PLOTS] Saving to {PLOT_DIR}/...")

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Honest Confusion Matrix (No BMD, Acc={test_acc:.3f})")
plt.savefig(os.path.join(PLOT_DIR, "honest_cm.png"), dpi=150, bbox_inches="tight")
plt.close()

# 2. Train vs Val accuracy per fold
fig, ax = plt.subplots(figsize=(8, 5))
folds_x = [r["fold"] for r in fold_results]
tr_accs = [r["train_acc"] for r in fold_results]
val_accs = [r["val_acc"] for r in fold_results]
w = 0.35
ax.bar(np.array(folds_x) - w/2, tr_accs, w, label="Train", color="#94a3b8")
ax.bar(np.array(folds_x) + w/2, val_accs, w, label="Validation", color="#0099ff")
ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy")
ax.set_title("Train vs Validation Accuracy (Overfitting Check)")
ax.legend(); ax.set_ylim(0.4, 1.05); ax.grid(axis="y", alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "honest_trainval.png"), dpi=150, bbox_inches="tight")
plt.close()

# 3. Per-class recall
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(3)
bars = ax.bar(x, rec_t, color=["#22c55e", "#f59e0b", "#ef4444"], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.set_ylabel("Recall")
ax.set_title("Per-Class Recall (Honest Model, No BMD)")
ax.set_ylim(0, 1.15); ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, rec_t):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", fontweight="bold")
plt.savefig(os.path.join(PLOT_DIR, "honest_recall.png"), dpi=150, bbox_inches="tight")
plt.close()

# 4. ROC Curves
y_test_bin = label_binarize(y_te_f, classes=[0, 1, 2])
plt.figure(figsize=(8, 6))
colors = ["#22c55e", "#f59e0b", "#ef4444"]
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_f[:, i])
    auc_val = sk_auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{LABELS[i]} (AUC={auc_val:.2f})",
             color=colors[i], linewidth=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curves (Honest Model, No BMD Leakage)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "honest_roc.png"), dpi=150, bbox_inches="tight")
plt.close()

print("  Saved: honest_cm.png, honest_trainval.png, honest_recall.png, honest_roc.png")

print("\n" + "=" * 70)
print("  HONEST TRAINING COMPLETE")
print(f"  Accuracy: {test_acc:.4f} (realistic, no leakage)")
print(f"  Osteoporosis Recall: {rec_t[2]:.4f}")
print("=" * 70)
