"""Damage-aware adaptive restoration (project novelty).

MuralDiff and plain inpainting treat every damaged pixel the same way: one mask, one denoising
strength, one generator. Real mural damage is not uniform - a hairline crack hides almost intact
paint, while a large paint loss has nothing left to keep. This module adapts the restoration to the
damage, region by region:

1. Region analysis   - every connected damaged region is measured (area, thickness from the
                       distance transform, elongation) and its severity is read from the
                       segmentation probability map (90th percentile inside the region).
2. Damage-type routing - each region is classified as `crack`, `flake` or `loss` with simple,
                       explainable shape rules and routed accordingly (cracks skip the edge stage,
                       and can optionally be handled by classical inpainting only).
3. Severity-adaptive strength - denoising strength = type base strength scaled by severity, so
                       cracks are re-touched lightly (most of the underlying paint survives) and
                       large losses are fully regenerated. Strengths are quantised to a few tiers
                       so the cost stays at <= 3 diffusion calls, each running only `strength x steps`.

All holes are first pre-filled with classical (Telea) inpainting; low-strength diffusion then refines
that pre-fill instead of starting from a blank hole. Everything here except the diffusion call is
NumPy/OpenCV, so the analysis runs and is unit-tested on a CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from model.diffusion_model import classical_inpaint, run_restoration
from model.postprocess import composite

# base denoising strength per damage type, and the tiers strengths are snapped to
BASE_STRENGTH = {"crack": 0.45, "flake": 0.75, "loss": 1.0}
TIERS = (0.45, 0.75, 1.0)
TYPE_COLORS = {"crack": (230, 57, 70), "flake": (244, 162, 97), "loss": (42, 157, 143)}   # for the UI overlay


@dataclass
class DamageRegion:
    id: int
    type: str              # crack | flake | loss
    area_px: int
    area_pct: float        # of the whole image
    thickness_px: float    # 2 x max distance to the region border
    elongation: float      # length / thickness (minAreaRect)
    severity: float        # 0..1 from the probability map
    strength: float        # severity-adapted denoising strength
    tier: float            # strength snapped to TIERS
    bbox: Tuple[int, int, int, int]


def _to_bool(mask: Image.Image | np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if isinstance(mask, Image.Image):
        m = mask.convert("L")
        if size is not None and m.size != size:
            m = m.resize(size, Image.Resampling.NEAREST)
        return np.asarray(m) > 127
    return np.asarray(mask) > (127 if np.asarray(mask).dtype == np.uint8 else 0.5)


def _prob_array(prob: Any, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if prob is None:
        return None
    p = np.asarray(prob.convert("L"), dtype=np.float32) / 255.0 if isinstance(prob, Image.Image) else np.asarray(prob, np.float32)
    p = np.squeeze(p)
    if p.shape != shape:
        p = cv2.resize(p, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(p, 0.0, 1.0)


def classify(area_px: int, thickness: float, elongation: float, img_area: int, scale: float) -> str:
    """Explainable shape rules. `scale` = short image side / 512 so thresholds follow resolution."""
    area_pct = 100.0 * area_px / img_area
    if thickness <= 10 * scale and (elongation >= 3.0 or area_px / max(thickness, 1.0) ** 2 >= 8):
        return "crack"
    if area_pct >= 1.5 or thickness >= 40 * scale:
        return "loss"
    return "flake"


def snap(strength: float) -> float:
    return min(TIERS, key=lambda t: abs(t - strength))


def analyze_regions(mask: Image.Image | np.ndarray, prob: Any = None, threshold: float = 0.5,
                    min_area: int = 20, margin_px: int = 0) -> List[DamageRegion]:
    """Measure, classify and assign a strength to every connected damaged region.

    `margin_px` = dilation already applied to the mask (refine step). Shape rules are evaluated on the
    mask *before* that margin, otherwise a 4 px dilation turns every thin crack into a 'flake'.
    """
    m = _to_bool(mask)
    h, w = m.shape
    scale = min(h, w) / 512.0
    p = _prob_array(prob, (h, w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    regions: List[DamageRegion] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        reg = labels == i
        thickness = float(2.0 * dist[reg].max())
        thick_raw = max(thickness - 2.0 * margin_px, 1.0)        # thickness of the damage itself
        ys, xs = np.nonzero(reg)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        length = max(rw, rh, 1.0)
        elong = float(max(length - 2.0 * margin_px, 1.0) / thick_raw)
        raw_area = max(area - (2.0 * margin_px) * max(length - margin_px, 1.0), 1.0) if margin_px else area
        rtype = classify(int(raw_area), thick_raw, elong, h * w, scale)
        if p is None:
            sev = 0.5                           # no probability map: neutral severity -> base strength
        else:                                   # robust: the refined mask is dilated beyond the U-Net support
            sev = float(np.clip((np.percentile(p[reg], 90) - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0))
        strength = float(np.clip(BASE_STRENGTH[rtype] * (0.8 + 0.4 * sev), 0.3, 1.0))
        x, y, bw, bh = (int(v) for v in stats[i, :4])
        regions.append(DamageRegion(i, rtype, area, round(100.0 * area / (h * w), 3), round(thickness, 1),
                                    round(elong, 2), round(sev, 3), round(strength, 3), snap(strength), (x, y, bw, bh)))
    return regions


def region_masks(mask: Image.Image | np.ndarray, regions: List[DamageRegion]) -> Dict[float, np.ndarray]:
    """Union mask per strength tier."""
    m = _to_bool(mask)
    _, labels = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    out: Dict[float, np.ndarray] = {}
    for r in regions:
        out.setdefault(r.tier, np.zeros_like(m))
        out[r.tier] |= labels == r.id
    return out


def type_overlay(image: Image.Image, mask: Image.Image | np.ndarray, regions: List[DamageRegion],
                 alpha: float = 0.55) -> Image.Image:
    """Colour each region by damage type (crack red, flake orange, loss teal) for display."""
    img = np.asarray(image.convert("RGB")).astype(np.float32)
    m = _to_bool(mask, image.size)
    _, labels = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    for r in regions:
        reg = labels == r.id
        img[reg] = img[reg] * (1 - alpha) + np.array(TYPE_COLORS[r.type], np.float32) * alpha
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def summarize(regions: List[DamageRegion]) -> Dict[str, Any]:
    by = {t: [r for r in regions if r.type == t] for t in BASE_STRENGTH}
    return {"regions": len(regions),
            **{f"{t}_regions": len(v) for t, v in by.items()},
            **{f"{t}_area_pct": round(sum(r.area_pct for r in v), 3) for t, v in by.items()},
            "tiers_used": sorted({r.tier for r in regions})}


def restore_damage_aware(image: Image.Image, mask: Image.Image, prob: Any = None, mode: str = "sd",
                         cfg: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Image.Image], Dict[str, Any]]:
    """Tiered restoration. Returns ({'stage1': generated, 'prefill': ..., 'types': overlay}, metrics).

    cfg extras: threshold (for severity), crack_backend ('diffusion'|'classical'), feather_px.
    The returned 'stage1' image is the full generated canvas; restore_mural composites and post-processes it.
    """
    cfg = dict(cfg or {})
    image = image.convert("RGB")
    mask_b = _to_bool(mask, image.size)
    regions = analyze_regions(mask_b, prob, threshold=float(cfg.get("threshold", 0.5)),
                              min_area=int(cfg.get("min_area", 20)), margin_px=int(cfg.get("margin_px", 0)))
    tiers = region_masks(mask_b, regions)
    covered = np.zeros_like(mask_b)
    for t in tiers.values():
        covered |= t
    leftover = mask_b & ~covered            # specks below min_area: give them to the lightest tier
    if leftover.any():
        tiers.setdefault(TIERS[0], np.zeros_like(mask_b))
        tiers[TIERS[0]] |= leftover

    prefill = classical_inpaint(image, Image.fromarray(mask_b.astype(np.uint8) * 255))
    canvas = prefill          # result after Stage 1 + Stage 2 (fed to the next tier)
    canvas_s1 = prefill       # the same regions after Stage 1 only (for display / stage-wise evaluation)
    had_stage2 = False
    feather_px = int(cfg.get("feather_px", 4))
    calls: List[Dict[str, Any]] = []
    last_metrics: Dict[str, Any] = {}
    crack_backend = cfg.get("crack_backend", "diffusion")

    for tier in sorted(tiers):                       # light -> heavy
        tm = Image.fromarray(tiers[tier].astype(np.uint8) * 255)
        if mode == "classical" or (tier == TIERS[0] and crack_backend == "classical"):
            calls.append({"tier": tier, "backend": "classical", "pixels": int(tiers[tier].sum())})
            continue                                  # the Telea pre-fill already covers it
        tier_mode = "sd" if (tier == TIERS[0] and mode in ("sd_edge", "full")) else mode   # thin cracks skip Stage 2
        imgs, met = run_restoration(canvas, tm, tier_mode, {**cfg, "strength": tier})
        out = imgs.get("stage2", imgs["stage1"])
        had_stage2 = had_stage2 or "stage2" in imgs
        canvas_s1 = composite(canvas_s1, imgs["stage1"], tm, feather_px=feather_px)
        canvas = composite(canvas, out, tm, feather_px=feather_px)
        calls.append({"tier": tier, "backend": tier_mode, "pixels": int(tiers[tier].sum()),
                      "seconds": met.get("execution_time_seconds"), "stage2_error": met.get("stage2_error")})
        last_metrics = met

    metrics = {k: v for k, v in last_metrics.items() if k not in ("execution_time_seconds",)}
    metrics.update({
        "backend": (last_metrics.get("backend", "classical (Telea) inpainting") + " | damage-aware tiers"),
        "mode": mode, "damage_aware": True, "damage_summary": summarize(regions), "tier_calls": calls,
        "regions": [asdict(r) for r in regions],
    })
    images = {"stage1": canvas_s1, "prefill": prefill, "types": type_overlay(image, mask_b, regions)}
    if had_stage2:
        images["stage2"] = canvas
    return images, metrics
