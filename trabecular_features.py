"""
Trabecular Bone Texture (TBT) Feature Extraction Module
=========================================================
Extracts medically meaningful bone microarchitecture features from
DXA / X-ray images using classical computer-vision descriptors.

Feature Vector (37-d):
  - GLCM  : 16 features (4 properties × 4 angles)
  - LBP   : 10 features (uniform histogram bins)
  - Fractal:  1 feature  (box-counting dimension)
  - Gabor : 10 features (mean + std × 5 orientations)

Usage:
    from trabecular_features import extract_trabecular_features
    features = extract_trabecular_features("path/to/image.png")
"""

import os
import numpy as np
import cv2
from tqdm import tqdm

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# =============================================================================
# 🔧 CONFIGURATION
# =============================================================================

# Image preprocessing
TARGET_SIZE = (256, 256)        # Resize target before ROI crop
ROI_FRACTION = 0.6              # Center-crop fraction (60% of image)
GAUSSIAN_SIGMA = 1.0            # Gaussian blur sigma for denoising
CLAHE_CLIP = 2.0                # CLAHE clip limit
CLAHE_TILE = (8, 8)             # CLAHE tile grid size

# GLCM parameters
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0°, 45°, 90°, 135°
GLCM_PROPERTIES = ['contrast', 'correlation', 'energy', 'homogeneity']

# LBP parameters
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS  # 24
LBP_METHOD = 'uniform'
LBP_N_BINS = LBP_N_POINTS + 2  # 26 uniform patterns; we keep top 10

# Gabor filter parameters
GABOR_FREQUENCY = 0.1
GABOR_ORIENTATIONS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

# Output paths
TBT_FEATURES_PATH = "tbt_features.npy"
TBT_PATHS_FILE = "tbt_image_paths.txt"


# =============================================================================
# 🖼️ PREPROCESSING
# =============================================================================

def preprocess_image(image_path):
    """
    Load and preprocess a DXA image for texture analysis.

    Steps:
        1. Load as grayscale
        2. Resize to TARGET_SIZE
        3. Center-crop ROI (middle ROI_FRACTION)
        4. Gaussian blur for denoising
        5. CLAHE contrast enhancement

    Returns:
        np.ndarray: preprocessed grayscale image (uint8)
    """
    # Load as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Resize
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    # Center-crop ROI
    h, w = img.shape
    crop_h, crop_w = int(h * ROI_FRACTION), int(w * ROI_FRACTION)
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    img = img[start_y:start_y + crop_h, start_x:start_x + crop_w]

    # Gaussian blur for noise reduction
    ksize = int(GAUSSIAN_SIGMA * 4) | 1  # ensure odd kernel size
    img = cv2.GaussianBlur(img, (ksize, ksize), GAUSSIAN_SIGMA)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    img = clahe.apply(img)

    return img


# =============================================================================
# 📊 FEATURE EXTRACTORS
# =============================================================================

def compute_glcm_features(img):
    """
    Compute Gray Level Co-occurrence Matrix features.

    Returns 16 features: 4 properties × 4 angles.
    """
    glcm = graycomatrix(
        img,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=True
    )

    features = []
    for prop in GLCM_PROPERTIES:
        vals = graycoprops(glcm, prop)[0]  # shape: (n_angles,)
        features.extend(vals.tolist())

    return np.array(features, dtype=np.float64)


def compute_lbp_features(img):
    """
    Compute Local Binary Pattern histogram features.

    Uses uniform patterns with P=24, R=3.
    Returns top 10 histogram bins (normalized).
    """
    lbp = local_binary_pattern(img, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)

    # Compute normalized histogram
    n_bins = LBP_N_BINS
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    # Keep top 10 bins by magnitude for fixed-length output
    top_indices = np.argsort(hist)[::-1][:10]
    top_indices = np.sort(top_indices)  # maintain consistent ordering
    features = hist[top_indices]

    return features.astype(np.float64)


def compute_fractal_dimension(img):
    """
    Estimate fractal dimension using the box-counting method.

    Binarizes the image with Otsu's threshold, then counts
    how many boxes of decreasing size are needed to cover
    the foreground structure.

    Returns a single float value.
    """
    # Binarize
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = binary > 0  # boolean

    # Pad to nearest power of 2
    max_dim = max(binary.shape)
    p = int(np.ceil(np.log2(max_dim)))
    size = 2 ** p
    padded = np.zeros((size, size), dtype=bool)
    padded[:binary.shape[0], :binary.shape[1]] = binary

    # Box counting
    sizes = 2 ** np.arange(1, p + 1)
    counts = []

    for s in sizes:
        # Reshape into boxes of size s×s and check if any pixel is set
        n_boxes_h = size // s
        n_boxes_w = size // s
        reshaped = padded[:n_boxes_h * s, :n_boxes_w * s].reshape(
            n_boxes_h, s, n_boxes_w, s
        )
        counts.append(np.any(reshaped, axis=(1, 3)).sum())

    # Linear fit of log(count) vs log(1/size)
    coeffs = np.polyfit(np.log(1.0 / sizes), np.log(np.array(counts, dtype=float) + 1), 1)
    fractal_dim = coeffs[0]

    return np.array([fractal_dim], dtype=np.float64)


def compute_gabor_features(img):
    """
    Compute Gabor filter response statistics.

    Applies Gabor filters at 5 orientations with fixed frequency.
    Returns 10 features: mean + std of response per orientation.
    """
    features = []
    img_float = img.astype(np.float64) / 255.0

    for theta in GABOR_ORIENTATIONS:
        kernel = cv2.getGaborKernel(
            ksize=(31, 31),
            sigma=4.0,
            theta=theta,
            lambd=1.0 / GABOR_FREQUENCY,
            gamma=0.5,
            psi=0
        )
        filtered = cv2.filter2D(img_float, cv2.CV_64F, kernel)
        features.append(filtered.mean())
        features.append(filtered.std())

    return np.array(features, dtype=np.float64)


# =============================================================================
# 🧬 MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_trabecular_features(image_path):
    """
    Extract the full 37-d trabecular bone texture feature vector
    from a single image.

    Args:
        image_path (str): Path to the DXA image file.

    Returns:
        np.ndarray: Feature vector of shape (37,).
    """
    img = preprocess_image(image_path)

    glcm_feats = compute_glcm_features(img)       # 16
    lbp_feats = compute_lbp_features(img)          # 10
    fractal_feats = compute_fractal_dimension(img)  #  1
    gabor_feats = compute_gabor_features(img)      # 10

    feature_vector = np.concatenate([
        glcm_feats,
        lbp_feats,
        fractal_feats,
        gabor_feats
    ])

    return feature_vector


def get_feature_names():
    """
    Return human-readable names for each of the 37 features.
    Useful for feature importance visualization.
    """
    names = []

    # GLCM names (4 properties × 4 angles)
    angle_labels = ['0°', '45°', '90°', '135°']
    for prop in GLCM_PROPERTIES:
        for angle in angle_labels:
            names.append(f"GLCM_{prop}_{angle}")

    # LBP names
    for i in range(10):
        names.append(f"LBP_bin_{i}")

    # Fractal
    names.append("Fractal_dim")

    # Gabor names
    for i, theta in enumerate(GABOR_ORIENTATIONS):
        angle_deg = int(np.degrees(theta))
        names.append(f"Gabor_mean_{angle_deg}°")
        names.append(f"Gabor_std_{angle_deg}°")

    return names


# =============================================================================
# 📦 BATCH EXTRACTION
# =============================================================================

def extract_all_features(image_folder, save=True):
    """
    Extract TBT features from all images in a folder.

    Args:
        image_folder (str): Path to folder containing DXA images.
        save (bool): If True, save features to tbt_features.npy
                     and paths to tbt_image_paths.txt.

    Returns:
        features (np.ndarray): Shape (N, 37).
        image_paths (list[str]): Corresponding file paths.
    """
    print("[INFO] Scanning image folder...")

    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_paths = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(valid_extensions)
    ])

    print(f"[INFO] Found {len(image_paths)} images")
    print("[INFO] Extracting trabecular bone texture features...")

    all_features = []
    failed = []

    for path in tqdm(image_paths, desc="TBT Extraction"):
        try:
            feat = extract_trabecular_features(path)
            all_features.append(feat)
        except Exception as e:
            print(f"[WARNING] Failed on {path}: {e}")
            failed.append(path)

    # Remove failed paths from the list
    successful_paths = [p for p in image_paths if p not in failed]

    features = np.vstack(all_features)

    print(f"[INFO] Extraction complete: {features.shape}")
    if failed:
        print(f"[WARNING] {len(failed)} images failed extraction")

    if save:
        np.save(TBT_FEATURES_PATH, features)
        print(f"[INFO] Features saved to {TBT_FEATURES_PATH}")

        with open(TBT_PATHS_FILE, "w", encoding="utf-8") as f:
            for p in successful_paths:
                f.write(p + "\n")
        print(f"[INFO] Image paths saved to {TBT_PATHS_FILE}")

    return features, successful_paths


# =============================================================================
# ▶️ STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    # Quick single-image test mode
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"[INFO] Testing single image: {test_path}")
        feats = extract_trabecular_features(test_path)
        names = get_feature_names()

        print(f"[INFO] Feature vector shape: {feats.shape}")
        print(f"[INFO] Feature names ({len(names)}):")
        for name, val in zip(names, feats):
            print(f"  {name:30s} = {val:.6f}")
    else:
        # Full batch extraction
        IMAGE_FOLDER = "FinallDATA/images"
        print(f"[INFO] Starting batch extraction from: {IMAGE_FOLDER}")
        features, paths = extract_all_features(IMAGE_FOLDER, save=True)
        print(f"[INFO] Done. Final shape: {features.shape}")
