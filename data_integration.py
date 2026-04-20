import os
import pandas as pd
import re

# -------- CONFIG --------
IMAGE_FOLDER = "FinallDATA/images"
CSV_PATH = "cleaned_clinical_data.csv"

PKL_OUTPUT = "integrated_data.pkl"
CSV_OUTPUT = "integrated_data.csv"

# -------- CLEAN ID FUNCTION --------
def clean_id(x):
    x = str(x).strip()
    x = x.replace(".0", "")        # remove float artifact
    x = x.lstrip('0')              # remove leading zeros
    return x

# -------- EXTRACT PATIENT ID FROM IMAGE --------
def get_patient_id(filename):
    # Extract first number sequence (robust)
    match = re.search(r'\d+', filename)
    if match:
        pid = match.group()
        pid = clean_id(pid)
        print(f"[INFO] Extracted ID {pid} from {filename}")
        return pid
    else:
        print(f"[WARNING] No ID found in filename: {filename}")
        return None

# -------- BUILD IMAGE DICTIONARY --------
def build_image_dict(image_folder):
    print("\n[INFO] Building image dictionary...")

    if not os.path.exists(image_folder):
        print(f"[ERROR] Image folder not found: {image_folder}")
        return {}

    files = os.listdir(image_folder)
    print(f"[INFO] Total files found: {len(files)}")

    image_dict = {}

    for img in files:
        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
            pid = get_patient_id(img)

            if pid is None:
                continue

            full_path = os.path.join(image_folder, img)
            image_dict.setdefault(pid, []).append(full_path)

    print(f"[INFO] Total patients with images: {len(image_dict)}")
    return image_dict

# -------- MAIN INTEGRATION --------
def integrate_data():
    print("\n[INFO] Starting data integration...")

    # Check CSV
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV not found: {CSV_PATH}")
        return

    # Load CSV
    print("[INFO] Loading clinical data...")
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Clinical data shape: {df.shape}")

    if 'IDENTIFIER_1' not in df.columns:
        print("[ERROR] IDENTIFIER_1 column missing!")
        return

    # Clean CSV IDs
    print("[INFO] Cleaning patient IDs in CSV...")
    df['IDENTIFIER_1'] = df['IDENTIFIER_1'].apply(clean_id)

    # Build image mapping
    image_dict = build_image_dict(IMAGE_FOLDER)

    if len(image_dict) == 0:
        print("[ERROR] No images found or mapping failed!")
        return

    # DEBUG: Compare ID sets
    csv_ids = set(df['IDENTIFIER_1'])
    image_ids = set(image_dict.keys())

    print("\n[DEBUG INFO]")
    print("Patients in CSV:", len(csv_ids))
    print("Patients in Images:", len(image_ids))
    print("Common patients:", len(csv_ids & image_ids))
    print("Only in CSV:", len(csv_ids - image_ids))
    print("Only in Images:", len(image_ids - csv_ids))

    print("\nSample CSV IDs:", list(csv_ids)[:5])
    print("Sample Image IDs:", list(image_ids)[:5])

    # Map images
    print("\n[INFO] Mapping images to dataframe...")
    df['IMAGE_PATHS'] = df['IDENTIFIER_1'].map(image_dict)

    # Filter
    before = len(df)
    df = df.dropna(subset=['IMAGE_PATHS'])
    after = len(df)

    print(f"[INFO] Rows before filtering: {before}")
    print(f"[INFO] Rows after filtering: {after}")

    if after == 0:
        print("[ERROR] No matching patients found! Check ID formatting.")
        return

    # -------- SAVE FILES --------
    print("\n[INFO] Saving files...")

    # Save PKL (for pipeline)
    df.to_pickle(PKL_OUTPUT)

    # Save CSV (for human viewing)
    df_copy = df.copy()
    df_copy['IMAGE_PATHS'] = df_copy['IMAGE_PATHS'].apply(lambda x: str(x))
    df_copy.to_csv(CSV_OUTPUT, index=False)

    # Confirm
    if os.path.exists(PKL_OUTPUT) and os.path.exists(CSV_OUTPUT):
        print("[SUCCESS] Both PKL and CSV files saved successfully!")
    else:
        print("[ERROR] File saving failed!")

    print("\n[INFO] Data integration completed successfully!")


# -------- RUN --------
if __name__ == "__main__":
    print("[INFO] Script started...")
    integrate_data()
    print("[INFO] Script finished.")