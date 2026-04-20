"""
CNN Feature Extraction Service (for inference)
================================================
Extracts ResNet50 features from a single image at inference time.
Uses the same transform pipeline as feature_extraction.py.

Singleton pattern — model loaded once, reused for all requests.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io


# =============================================================================
# SINGLETON ResNet50 FEATURE EXTRACTOR
# =============================================================================

_cnn_model = None
_device = None
_transform = None


def _load_cnn():
    """Load ResNet50 (minus final FC layer) once."""
    global _cnn_model, _device, _transform

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ResNet50 without the final classification layer -> 2048-d output
    base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    _cnn_model = nn.Sequential(*list(base.children())[:-1])
    _cnn_model = _cnn_model.to(_device)
    _cnn_model.eval()

    # Same transform as in feature_extraction.py
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print(f"[CNN] ResNet50 loaded on {_device}")


def extract_cnn_features(image_path=None, image_bytes=None):
    """
    Extract 2048-d CNN features from a single image.

    Args:
        image_path (str): Path to image file. OR
        image_bytes (bytes): Raw image bytes (from upload).

    Returns:
        np.ndarray: Shape (2048,) feature vector.
    """
    global _cnn_model, _device, _transform

    if _cnn_model is None:
        _load_cnn()

    # Load image
    if image_bytes is not None:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    elif image_path is not None:
        img = Image.open(image_path).convert("RGB")
    else:
        raise ValueError("Provide either image_path or image_bytes")

    # Transform and extract
    tensor = _transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        features = _cnn_model(tensor)
        features = features.squeeze().cpu().numpy()

    return features  # Shape: (2048,)
