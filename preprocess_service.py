"""
Preprocessing Service Module (Base64 Mode)
============================================
Runs the preprocessing pipeline and returns Base64-encoded images
directly in the API response. This eliminates all file-serving
path issues that plague URL-based approaches on Windows.
"""

import os
import sys
import base64
import cv2

# Add parent directory for imports
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PARENT_DIR)

from preprocessing_visualizer import visualize_preprocessing

# Output directory for preprocessed images (still saved to disk for traceability)
PREPROCESS_OUTPUT = os.path.join(PARENT_DIR, "preprocessing_outputs")

# Step metadata
STEP_ORDER = ["original", "grayscale", "resized", "cropped", "denoised", "final"]
STEP_LABELS = {
    "original": "Original",
    "grayscale": "Grayscale",
    "resized": "Resized (256x256)",
    "cropped": "ROI Crop (60%)",
    "denoised": "Gaussian Denoised",
    "final": "CLAHE Enhanced",
}


def _file_to_base64(filepath):
    """Read an image file and return a Base64 data URI string."""
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"[PREPROCESS] Warning: could not encode {filepath}: {e}")
        return None


def run_preprocessing(filepath):
    """
    Run the full preprocessing pipeline, save images to disk,
    and return Base64-encoded versions for direct embedding.

    Args:
        filepath (str): Path to the uploaded image.

    Returns:
        dict: {
            "original": "data:image/png;base64,...",
            "grayscale": "data:image/png;base64,...",
            "resized": "data:image/png;base64,...",
            "cropped": "data:image/png;base64,...",
            "denoised": "data:image/png;base64,...",
            "final": "data:image/png;base64,...",
            "steps": [
                {"name": "original", "label": "Original", "image": "data:image/png;base64,..."},
                ...
            ]
        }
    """
    print(f"[PREPROCESS] Running pipeline on: {os.path.basename(filepath)}")

    # Run the visualizer (saves to disk)
    preprocess_paths, _ = visualize_preprocessing(filepath, PREPROCESS_OUTPUT)

    # Build both flat dict and steps array
    flat_images = {}
    steps = []

    for step_name in STEP_ORDER:
        if step_name in preprocess_paths:
            fpath = preprocess_paths[step_name]
            b64_data = _file_to_base64(fpath)

            if b64_data:
                flat_images[step_name] = b64_data
                steps.append({
                    "name": step_name,
                    "label": STEP_LABELS.get(step_name, step_name.title()),
                    "image": b64_data,
                })
                print(f"[PREPROCESS]   {step_name}: OK ({os.path.getsize(fpath)} bytes)")
            else:
                print(f"[PREPROCESS]   {step_name}: FAILED to encode")
        else:
            print(f"[PREPROCESS]   {step_name}: not found in pipeline output")

    print(f"[PREPROCESS] Encoded {len(steps)}/{len(STEP_ORDER)} steps successfully")

    result = dict(flat_images)
    result["steps"] = steps
    return result


def get_preprocessing_dir(subdir):
    """Get the absolute path of a preprocessing subdirectory."""
    return os.path.join(PREPROCESS_OUTPUT, subdir)
