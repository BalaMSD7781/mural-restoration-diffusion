"""Image-quality metrics for restoration experiments (PSNR, SSIM, optional LPIPS).

Because real murals have no undamaged ground truth, restoration is evaluated on *synthetic* damage
applied to clean MuralDH crops (tools/synthetic_damage.py): the clean crop is the reference.
Metrics are reported on the whole image and inside the damaged region only (masked), since the
untouched pixels are identical by construction and inflate whole-image scores.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_LPIPS = {"model": None, "failed": False}


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _bbox(mask: np.ndarray, pad: int = 8, min_size: int = 64):
    """Padded bounding box of the mask, grown to at least `min_size` px (AlexNet-LPIPS needs >=~64 px)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h), max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)

    def grow(a, b, n):
        if b - a >= min_size:
            return a, b
        a = max(0, min(a - (min_size - (b - a)) // 2, n - min_size)); return a, min(n, a + min_size)
    y0, y1 = grow(y0, y1, h); x0, x1 = grow(x0, x1, w)
    return y0, y1, x0, x1


def masked_psnr(ref: np.ndarray, out: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    mse = np.mean((ref[mask].astype(np.float64) - out[mask].astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def masked_ssim(ref: np.ndarray, out: np.ndarray, mask: np.ndarray) -> float:
    """Mean SSIM over pixels inside the mask (SSIM map averaged where mask is True)."""
    if mask.sum() == 0:
        return float("nan")
    _, smap = structural_similarity(ref, out, channel_axis=2, full=True, data_range=255)
    return float(smap.mean(axis=2)[mask].mean())


def lpips_distance(ref: Image.Image, out: Image.Image) -> Optional[float]:
    """LPIPS (AlexNet) if the `lpips` + torch packages are installed, else None."""
    try:
        import torch
        if _LPIPS["model"] is None:
            import lpips
            _LPIPS["model"] = lpips.LPIPS(net="alex", verbose=False)
        t = lambda im: torch.from_numpy(_arr(im)).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        with torch.no_grad():
            return float(_LPIPS["model"](t(ref), t(out)).item())
    except ImportError:
        return None                      # lpips not installed: metric unavailable
    except Exception as e:               # a single bad crop must not disable LPIPS for the rest of the run
        print("LPIPS failed for one image:", repr(e))
        return float("nan")


def restoration_metrics(reference: Image.Image, output: Image.Image, mask: Image.Image,
                        with_lpips: bool = True) -> Dict[str, Optional[float]]:
    ref = _arr(reference)
    out = _arr(output.resize(reference.size))
    m = np.asarray(mask.convert("L").resize(reference.size, Image.Resampling.NEAREST)) > 127
    res: Dict[str, Optional[float]] = {
        "psnr": float(peak_signal_noise_ratio(ref, out, data_range=255)),
        "ssim": float(structural_similarity(ref, out, channel_axis=2, data_range=255)),
        "psnr_masked": masked_psnr(ref, out, m),
        "ssim_masked": masked_ssim(ref, out, m),
        "damage_pct": float(m.mean() * 100),
    }
    if with_lpips:
        bb = _bbox(m)
        if bb:
            y0, y1, x0, x1 = bb
            res["lpips_masked"] = lpips_distance(
                Image.fromarray(ref[y0:y1, x0:x1]), Image.fromarray(out[y0:y1, x0:x1]))
        res["lpips"] = lpips_distance(reference, output)
    return res
