"""
Preprocessing Visualizer
=========================
Saves intermediate images at each preprocessing step for
visual inspection and debugging.

Steps visualized:
    1. Original (as-loaded)
    2. Grayscale conversion
    3. Resized (256×256)
    4. Center-cropped ROI (60%)
    5. Gaussian denoised
    6. CLAHE enhanced (final)

Output structure:
    preprocessing_outputs/
        original/
        grayscale/
        resized/
        cropped/
        denoised/
        final/

Usage:
    from preprocessing_visualizer import visualize_preprocessing
    steps = visualize_preprocessing("path/to/image.png")
"""

import os
import cv2
import numpy as np

# Reuse the same parameters from trabecular_features.py
TARGET_SIZE = (256, 256)
ROI_FRACTION = 0.6
GAUSSIAN_SIGMA = 1.0
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# Output directory
OUTPUT_BASE = "preprocessing_outputs"

# Step subdirectories
STEP_DIRS = ["original", "grayscale", "resized", "cropped", "denoised", "final"]


def _ensure_dirs(base_dir):
    """Create all step subdirectories if they don't exist."""
    for step in STEP_DIRS:
        os.makedirs(os.path.join(base_dir, step), exist_ok=True)


def visualize_preprocessing(image_path, output_dir=None):
    """
    Run the full preprocessing pipeline on a single image,
    saving the result of each step to disk.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Base output directory. Defaults to OUTPUT_BASE.

    Returns:
        dict: Mapping of step name → saved file path.
        np.ndarray: The final preprocessed image (uint8 grayscale).
    """
    if output_dir is None:
        output_dir = OUTPUT_BASE

    _ensure_dirs(output_dir)

    # Derive filename stem for traceability
    stem = os.path.splitext(os.path.basename(image_path))[0]
    saved_paths = {}

    # ------------------------------------------------------------------
    # Step 1: Load original (color)
    # ------------------------------------------------------------------
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    path = os.path.join(output_dir, "original", f"{stem}_original.png")
    cv2.imwrite(path, img_color)
    saved_paths["original"] = path

    # ------------------------------------------------------------------
    # Step 2: Grayscale conversion
    # ------------------------------------------------------------------
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    path = os.path.join(output_dir, "grayscale", f"{stem}_grayscale.png")
    cv2.imwrite(path, img_gray)
    saved_paths["grayscale"] = path

    # ------------------------------------------------------------------
    # Step 3: Resize to TARGET_SIZE
    # ------------------------------------------------------------------
    img_resized = cv2.resize(img_gray, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    path = os.path.join(output_dir, "resized", f"{stem}_resized.png")
    cv2.imwrite(path, img_resized)
    saved_paths["resized"] = path

    # ------------------------------------------------------------------
    # Step 4: Center-crop ROI
    # ------------------------------------------------------------------
    h, w = img_resized.shape
    crop_h, crop_w = int(h * ROI_FRACTION), int(w * ROI_FRACTION)
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    img_cropped = img_resized[start_y:start_y + crop_h, start_x:start_x + crop_w]

    path = os.path.join(output_dir, "cropped", f"{stem}_cropped.png")
    cv2.imwrite(path, img_cropped)
    saved_paths["cropped"] = path

    # ------------------------------------------------------------------
    # Step 5: Gaussian blur (denoising)
    # ------------------------------------------------------------------
    ksize = int(GAUSSIAN_SIGMA * 4) | 1  # ensure odd kernel size
    img_denoised = cv2.GaussianBlur(img_cropped, (ksize, ksize), GAUSSIAN_SIGMA)

    path = os.path.join(output_dir, "denoised", f"{stem}_denoised.png")
    cv2.imwrite(path, img_denoised)
    saved_paths["denoised"] = path

    # ------------------------------------------------------------------
    # Step 6: CLAHE contrast enhancement (final)
    # ------------------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    img_final = clahe.apply(img_denoised)

    path = os.path.join(output_dir, "final", f"{stem}_final.png")
    cv2.imwrite(path, img_final)
    saved_paths["final"] = path

    print(f"[INFO] Preprocessing visualization saved for: {stem}")
    for step, fpath in saved_paths.items():
        print(f"  {step:12s} -> {fpath}")

    return saved_paths, img_final


# =============================================================================
# ▶️ STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = "test_image.jpg"

    print(f"[INFO] Visualizing preprocessing for: {test_path}")
    paths, final = visualize_preprocessing(test_path)
    print(f"\n[INFO] Final image shape: {final.shape}")
    print("[INFO] Done!")
