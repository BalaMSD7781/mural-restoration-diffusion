"""Damage segmentation (U-Net / Attention U-Net) for the mural restoration app.

Works with any Keras model that maps ``(H, W, C)`` -> ``(H, W, 1)`` probabilities where C is
3 (RGB) or 4 (RGB + Canny edge channel, as used by the Attention U-Net training notebook).
Input size and channel count are read from the model, so swapping in the Attention U-Net
needs only a file in ``models/`` (see config.SEGMENTATION_MODELS).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

import config

try:
    import streamlit as st
except ImportError:  # CLI / tests
    st = None

_MODEL_CACHE: Dict[str, Any] = {}


def _load_model_uncached(path: str):
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is not installed. Install requirements.txt.") from exc
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Segmentation model not found: {p}. Place it in models/ or edit config.SEGMENTATION_MODELS."
        )
    return keras.models.load_model(p, compile=False)


if st is not None:
    _load_model_uncached = st.cache_resource(show_spinner=False)(_load_model_uncached)


def load_model(model_key: Optional[str] = None):
    """Load (and cache) a registered segmentation model. Returns (model, entry)."""
    key = model_key or config.DEFAULT_SEGMENTATION_MODEL
    if key not in config.SEGMENTATION_MODELS:
        raise KeyError(f"Unknown segmentation model '{key}'. Known: {list(config.SEGMENTATION_MODELS)}")
    entry = config.SEGMENTATION_MODELS[key]
    return _load_model_uncached(str(entry["path"])), entry


def _prepare_input(image: Image.Image, size: Tuple[int, int], channels: int) -> np.ndarray:
    rgb = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    if channels == 3:
        return arr[None, ...]
    if channels == 4:  # identical to the training pipeline: Canny(100,200) on the resized RGB
        import cv2
        gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, config.CANNY_LOW, config.CANNY_HIGH).astype(np.float32) / 255.0
        return np.concatenate([arr, canny[..., None]], axis=-1)[None, ...]
    raise ValueError(f"Unsupported number of input channels: {channels}")


def predict_probability(image: Image.Image, model_key: Optional[str] = None) -> np.ndarray:
    """Probability map at the *original* image size, float32 in [0, 1]."""
    model, _ = load_model(model_key)
    _, h, w, c = model.input_shape
    batch = _prepare_input(image, (w, h), c)
    pred = np.squeeze(np.asarray(model.predict(batch, verbose=0)))
    if pred.ndim != 2:
        raise ValueError(f"Expected a single-channel mask, got shape {pred.shape}.")
    pred = np.clip(pred, 0.0, 1.0).astype(np.float32)
    prob = Image.fromarray((pred * 255).astype(np.uint8), mode="L").resize(
        image.size, Image.Resampling.BILINEAR
    )
    return np.asarray(prob, dtype=np.float32) / 255.0


def threshold_mask(probability: np.ndarray, threshold: float) -> Image.Image:
    return Image.fromarray(((probability >= threshold).astype(np.uint8)) * 255, mode="L")


def predict_damage_mask(
    image: Image.Image,
    threshold: Optional[float] = None,
    model_key: Optional[str] = None,
    **_: Any,
) -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
    """Predict (binary mask, probability map, stats) for an RGB image."""
    key = model_key or config.DEFAULT_SEGMENTATION_MODEL
    entry = config.SEGMENTATION_MODELS[key]
    thr = float(entry["threshold"] if threshold is None else threshold)
    prob = predict_probability(image, key)
    mask = threshold_mask(prob, thr)
    damage_pixels = int(np.count_nonzero(np.asarray(mask) > 0))
    total = mask.width * mask.height
    return mask, prob, {
        "model_key": key,
        "model": Path(entry["path"]).name,
        "threshold": thr,
        "damage_pixels": damage_pixels,
        "damage_percentage": round(damage_pixels / total * 100, 2),
    }
