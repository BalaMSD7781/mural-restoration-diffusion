"""Adaptive, region-wise heritage post-processing (the project's novelty over MuralDiff).

Everything here is classical (NumPy / OpenCV) so it runs on a CPU-only laptop and is unit-testable.
All corrections are *region-wise*: they are computed per connected damaged region from the intact
mural ring around it and applied only inside a feathered mask, so undamaged pixels are never altered.

Pipeline (see ``postprocess``):
  1. composite      - paste the generated pixels into the original with a feathered mask
                      (diffusion re-encodes the whole image through the VAE; compositing keeps
                      every intact pixel bit-exact)
  2. colour match   - per-region LAB mean/std transfer from the surrounding ring (heritage-aware
                      colour: pigments of the neighbourhood, not a global palette)
  3. texture match  - adds high-frequency detail where the generated region is smoother than its
                      surroundings (plaster grain / brush texture preservation)
  4. edge-aware sharpen - unsharp masking weighted by the local edge map (Canny) so lines get
                      crisper without amplifying flat-area noise
  5. super-resolution - optional x2 upscaling (Real-ESRGAN if installed, else Lanczos + sharpen)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import config


# ------------------------------------------------------------------------ helpers
def _to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _mask_np(mask: Image.Image, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    m = mask.convert("L")
    if size is not None and m.size != size:
        m = m.resize(size, Image.Resampling.NEAREST)
    return (np.asarray(m) > 127)


def feather(mask_bool: np.ndarray, px: int) -> np.ndarray:
    """Soft alpha in [0,1]: 1 inside the mask, linear falloff over ~px pixels outside."""
    m = mask_bool.astype(np.uint8) * 255
    if px <= 0:
        return (m / 255.0).astype(np.float32)
    k = int(px) * 2 + 1
    soft = cv2.GaussianBlur(m, (k, k), sigmaX=max(px / 2.0, 0.5))
    alpha = np.maximum(soft.astype(np.float32) / 255.0, mask_bool.astype(np.float32))
    alpha[alpha < 1e-3] = 0.0                      # exactly zero away from the region -> intact pixels stay bit-exact
    return np.clip(alpha, 0.0, 1.0)


def composite(original: Image.Image, generated: Image.Image, mask: Image.Image,
              feather_px: int = 6) -> Image.Image:
    """Keep original pixels outside the damage; blend generated pixels in with a soft edge."""
    o = _to_np(original).astype(np.float32)
    g = _to_np(generated.resize(original.size, Image.Resampling.LANCZOS)).astype(np.float32)
    alpha = feather(_mask_np(mask, original.size), feather_px)[..., None]
    out = o * (1 - alpha) + g * alpha
    return Image.fromarray(np.rint(np.clip(out, 0, 255)).astype(np.uint8))


def _regions(mask_bool: np.ndarray, min_area: int = 25) -> List[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def _ring(region: np.ndarray, full_mask: np.ndarray, width: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    dil = cv2.dilate(region.astype(np.uint8), k).astype(bool)
    return dil & ~cv2.dilate(full_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


# ------------------------------------------------------------------ colour matching
def match_color(restored: Image.Image, original: Image.Image, mask: Image.Image,
                strength: float = 0.7, ring_px: int = 24, feather_px: int = 6) -> Image.Image:
    """Region-wise LAB mean/std transfer from the intact ring around each damaged region."""
    rest = cv2.cvtColor(_to_np(restored), cv2.COLOR_RGB2LAB).astype(np.float32)
    orig = cv2.cvtColor(_to_np(original), cv2.COLOR_RGB2LAB).astype(np.float32)
    m = _mask_np(mask, original.size)
    out = rest.copy()
    touched = np.zeros(m.shape, dtype=bool)
    for reg in _regions(m):
        ring = _ring(reg, m, ring_px)
        if ring.sum() < 50:
            continue
        for c in range(3):
            tgt_mu, tgt_sd = orig[..., c][ring].mean(), orig[..., c][ring].std() + 1e-6
            src = rest[..., c][reg]
            src_mu, src_sd = src.mean(), src.std() + 1e-6
            # clamp the std gain: a flat patch must not be blown up to the ring's contrast
            gain = float(np.clip(tgt_sd / src_sd, 0.6, 1.6))
            adj = (rest[..., c] - src_mu) * gain + tgt_mu
            alpha = feather(reg, feather_px) * strength
            out[..., c] = out[..., c] * (1 - alpha) + adj * alpha
        touched |= feather(reg, feather_px) > 0
    lab_rgb = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    base_rgb = _to_np(restored)
    res = np.where(touched[..., None], lab_rgb, base_rgb)      # untouched pixels keep exact RGB values
    return Image.fromarray(res.astype(np.uint8))


# ----------------------------------------------------------------- texture matching
def match_texture(restored: Image.Image, original: Image.Image, mask: Image.Image,
                  strength: float = 0.6, ring_px: int = 24, seed: int = 0) -> Image.Image:
    """Add grain so a too-smooth generated region matches the ring's high-frequency energy."""
    rng = np.random.default_rng(seed)
    img = _to_np(restored).astype(np.float32)
    gray_o = cv2.cvtColor(_to_np(original), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_r = cv2.cvtColor(_to_np(restored), cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp_o = gray_o - cv2.GaussianBlur(gray_o, (0, 0), 1.5)
    hp_r = gray_r - cv2.GaussianBlur(gray_r, (0, 0), 1.5)
    m = _mask_np(mask, original.size)
    for reg in _regions(m):
        ring = _ring(reg, m, ring_px)
        if ring.sum() < 50:
            continue
        target = hp_o[ring].std()
        have = hp_r[reg].std()
        deficit = target ** 2 - have ** 2
        if deficit <= 0:
            continue
        noise = rng.normal(0, 1, gray_r.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 0.8)           # grain, not salt-and-pepper
        noise *= np.sqrt(deficit) / (noise.std() + 1e-6)
        alpha = (feather(reg, 4) * strength)[..., None]
        img += noise[..., None] * alpha
    return Image.fromarray(np.rint(np.clip(img, 0, 255)).astype(np.uint8))


# ----------------------------------------------------------------- edge-aware sharpen
def sharpen_edges(restored: Image.Image, mask: Image.Image, amount: float = 0.8,
                  radius: float = 1.2, feather_px: int = 6) -> Image.Image:
    """Unsharp masking weighted by a smoothed Canny edge map, applied inside the mask only."""
    img = _to_np(restored)
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    sharp = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, config.CANNY_LOW, config.CANNY_HIGH)
    edge_w = cv2.GaussianBlur(cv2.dilate(edges, np.ones((3, 3), np.uint8)).astype(np.float32) / 255.0, (0, 0), 2.0)
    edge_w = np.clip(edge_w * 2.0, 0.15, 1.0)                    # keep a little sharpening in flat areas
    alpha = (feather(_mask_np(mask, restored.size), feather_px) * edge_w)[..., None]
    out = img.astype(np.float32) * (1 - alpha) + sharp.astype(np.float32) * alpha
    return Image.fromarray(np.rint(np.clip(out, 0, 255)).astype(np.uint8))


# ------------------------------------------------------------------ super-resolution
_REALESRGAN = None


def _try_realesrgan(scale: int):
    """Optional: Real-ESRGAN x2 if `realesrgan` is installed and weights sit in models/."""
    global _REALESRGAN
    weights = config.MODELS_DIR / "RealESRGAN_x2plus.pth"
    if scale != 2 or not weights.is_file():
        return None
    if _REALESRGAN is None:
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            _REALESRGAN = RealESRGANer(scale=2, model_path=str(weights), model=net, tile=256,
                                       half=torch.cuda.is_available())
        except Exception:
            _REALESRGAN = False
    return _REALESRGAN or None


def super_resolve(img: Image.Image, scale: int = 2) -> Tuple[Image.Image, str]:
    """Upscale by `scale`. Returns (image, method)."""
    up = _try_realesrgan(scale)
    if up is not None:
        try:
            bgr = cv2.cvtColor(_to_np(img), cv2.COLOR_RGB2BGR)
            out, _ = up.enhance(bgr, outscale=scale)
            return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), "Real-ESRGAN x2"
        except Exception:
            pass
    w, h = img.size
    big = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    arr = _to_np(big)
    blur = cv2.GaussianBlur(arr, (0, 0), 1.0)
    sharp = cv2.addWeighted(arr, 1.6, blur, -0.6, 0)
    return Image.fromarray(sharp), f"Lanczos x{scale} + unsharp"


# ------------------------------------------------------------------------ pipeline
def postprocess(original: Image.Image, generated: Image.Image, mask: Image.Image,
                params: Optional[Dict[str, Any]] = None) -> Tuple[Image.Image, Dict[str, Any]]:
    """Run the enabled post-processing stages. Returns (final image, info).

    params keys (all optional): feather_px, color (bool), color_strength, texture (bool),
    sharpen (bool), sharpen_amount, super_res (bool), sr_scale.
    """
    p = {"feather_px": 6, "color": True, "color_strength": 0.7, "texture": True,
         "sharpen": True, "sharpen_amount": 0.8, "super_res": False, "sr_scale": 2}
    p.update(params or {})
    info: Dict[str, Any] = {"stages": ["composite"]}
    out = composite(original, generated, mask, p["feather_px"])
    if p["color"]:
        out = match_color(out, original, mask, strength=p["color_strength"], feather_px=p["feather_px"])
        info["stages"].append("region_color_match")
    if p["texture"]:
        out = match_texture(out, original, mask)
        info["stages"].append("texture_match")
    if p["sharpen"]:
        out = sharpen_edges(out, mask, amount=p["sharpen_amount"], feather_px=p["feather_px"])
        info["stages"].append("edge_aware_sharpen")
    if p["super_res"]:
        out, method = super_resolve(out, int(p["sr_scale"]))
        info["stages"].append(f"super_resolution ({method})")
        info["sr_method"] = method
    return out, info
