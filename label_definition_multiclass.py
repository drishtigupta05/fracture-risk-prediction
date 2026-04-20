import pandas as pd
import numpy as np
import os
import time

# -------- CONFIG --------
INPUT_PATH = "integrated_data.pkl"

OUTPUT_DIR = "outputs"
PKL_OUTPUT = os.path.join(OUTPUT_DIR, "labeled_data.pkl")
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "labeled_data.csv")

# -------- LABEL FUNCTION --------
def classify_tscore(t):
    if t <= -2.5:
        return 2   # Osteoporosis
    elif t < -1:
        return 1   # Osteopenia
    else:
        return 0   # Normal

# -------- MAIN --------
def define_labels():
    print("🚀 Starting label definition...")

    # Check input
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] File not found: {INPUT_PATH}")
        return

    # Load data
    df = pd.read_pickle(INPUT_PATH)
    print(f"[INFO] Loaded data: {df.shape}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check column
    if 'HIP_TSCORE' not in df.columns:
        print("[ERROR] HIP_TSCORE column missing!")
        return

    # Handle missing values
    print("[INFO] Cleaning missing T-scores...")
    df = df.dropna(subset=['HIP_TSCORE'])
    print(f"[INFO] Remaining rows: {len(df)}")

    # -------- CREATE LABELS --------
    print("[INFO] Creating multiclass labels...")
    df['LABEL'] = df['HIP_TSCORE'].apply(classify_tscore)

    label_map = {
        0: "Normal",
        1: "Osteopenia",
        2: "Osteoporosis"
    }
    df['LABEL_NAME'] = df['LABEL'].map(label_map)

    # -------- DEBUG --------
    print("\n📊 Label Distribution:")
    print(df['LABEL_NAME'].value_counts())

    print("\n📉 T-score Summary:")
    print(df['HIP_TSCORE'].describe())

    # -------- SAVE PKL --------
    print("\n💾 Saving PKL...")
    df.to_pickle(PKL_OUTPUT)
    print(f"✅ PKL saved at: {PKL_OUTPUT}")

    # -------- PREPARE CSV --------
    df_csv = df.copy()
    df_csv['IMAGE_PATHS'] = df_csv['IMAGE_PATHS'].apply(
        lambda x: " | ".join(x) if isinstance(x, list) else str(x)
    )

    # -------- SAFE CSV SAVE --------
    print("💾 Saving CSV... (close file if open)")

    for attempt in range(3):
        try:
            df_csv.to_csv(CSV_OUTPUT, index=False)
            print(f"✅ CSV saved at: {CSV_OUTPUT}")
            break
        except PermissionError:
            print(f"⚠️ CSV file is open. Attempt {attempt+1}/3...")
            time.sleep(2)
    else:
        print("❌ Could not save CSV. Please close Excel/VS Code preview and rerun.")

    print("\n🎉 Labeling completed successfully!")

# -------- RUN --------
if __name__ == "__main__":
    define_labels()