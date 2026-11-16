"""Synthetic mural damage generator (cracks, flaking, fading, stains, holes).

Extracted unchanged from the segmentation-training notebook so the *same* damage model is used for
(a) training the segmentation network and (b) building ground-truth pairs for restoration evaluation:
clean MuralDH crop -> (damaged image, binary mask); the clean crop is the PSNR/SSIM/LPIPS reference.
"""

import cv2
import numpy as np
import random

def generate_synthetic_damage(clean_img, seed=None):
    """
    clean_img: RGB uint8 array (H, W, 3)
    Returns: (damaged_img, mask) uint8, mask: 255=damaged
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    h, w = clean_img.shape[:2]
    damaged = clean_img.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    # pick 1-3 damage types per image, weighted toward fade/flake since
    # that's what dominates in the real MuralDH reference set
    pool = ["crack", "flake", "stain", "hole", "fade", "fade", "flake"]
    n_types = random.randint(1, 3)
    chosen = random.sample(pool, min(n_types, len(pool)))

    for dtype in chosen:
        if dtype == "crack": mask = _add_cracks(damaged, mask)
        elif dtype == "flake": mask = _add_flaking(damaged, mask)
        elif dtype == "stain": mask = _add_stains(damaged, mask)
        elif dtype == "hole": mask = _add_holes(damaged, mask)
        elif dtype == "fade": mask = _add_fading(damaged, mask)

    return damaged, mask


def _add_cracks(img, mask, n_cracks=None):
    h, w = img.shape[:2]
    n_cracks = n_cracks or random.randint(1, 3)
    for _ in range(n_cracks):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        length = random.randint(int(0.15 * w), int(0.5 * w))
        angle = random.uniform(0, 2 * np.pi)
        pts = [(x, y)]
        for _ in range(length // 5):
            angle += random.uniform(-0.35, 0.35)
            x = int(np.clip(x + 5 * np.cos(angle), 0, w - 1))
            y = int(np.clip(y + 5 * np.sin(angle), 0, h - 1))
            pts.append((x, y))
        thickness = random.randint(1, 3)  # real cracks were thin, single-stroke
        base = img[y, x].astype(int)
        crack_color = tuple(int(c) for c in np.clip(base - random.randint(30, 70), 0, 255))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], crack_color, thickness, cv2.LINE_AA)
            cv2.line(mask, pts[i], pts[i + 1], 255, thickness + 1, cv2.LINE_AA)
    return mask


def _add_flaking(img, mask, n_patches=None):
    """Large, irregular paint-loss patches — widened per calibration."""
    h, w = img.shape[:2]
    n_patches = n_patches or random.randint(1, 2)
    for _ in range(n_patches):
        cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
        n_pts = random.randint(6, 10)
        base_r = random.randint(int(0.08 * w), int(0.28 * w))  # much wider than before
        pts = []
        for k in range(n_pts):
            ang = 2 * np.pi * k / n_pts
            r = base_r * random.uniform(0.6, 1.3)  # irregular, not a clean circle
            pts.append([int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))])
        pts = np.array(pts, dtype=np.int32)

        patch_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(patch_mask, [pts], 255)
        patch_mask = cv2.GaussianBlur(patch_mask, (21, 21), 0)

        fill_color = np.array([205, 195, 175])  # plaster/base-layer tone — tune vs your murals
        alpha = (patch_mask.astype(np.float32) / 255.0)[..., None]
        img[:] = (img * (1 - alpha) + fill_color * alpha).astype(np.uint8)
        mask[:] = np.maximum(mask, (patch_mask > 50).astype(np.uint8) * 255)
    return mask


def _add_fading(img, mask, n_regions=None):
    """NEW — desaturates/lightens a region instead of overwriting it.
    This is what most of your real MuralDH samples actually look like
    (001241, 001268, 001219): faded color, not a flat replacement fill."""
    h, w = img.shape[:2]
    n_regions = n_regions or random.randint(1, 2)
    for _ in range(n_regions):
        cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
        n_pts = random.randint(5, 9)
        base_r = random.randint(int(0.1 * w), int(0.35 * w))
        pts = []
        for k in range(n_pts):
            ang = 2 * np.pi * k / n_pts
            r = base_r * random.uniform(0.7, 1.25)
            pts.append([int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))])
        pts = np.array(pts, dtype=np.int32)

        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(region_mask, [pts], 255)
        region_mask = cv2.GaussianBlur(region_mask, (35, 35), 0)  # very soft edge — fading has no hard boundary

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        alpha = (region_mask.astype(np.float32) / 255.0)
        desat = random.uniform(0.4, 0.7)   # reduce saturation
        lighten = random.uniform(1.15, 1.35)  # raise brightness/value
        hsv[..., 1] = hsv[..., 1] * (1 - alpha * desat)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1 + alpha * (lighten - 1)), 0, 255)
        img[:] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        mask[:] = np.maximum(mask, (region_mask > 40).astype(np.uint8) * 255)
    return mask


def _add_stains(img, mask, n_stains=None):
    h, w = img.shape[:2]
    n_stains = n_stains or random.randint(1, 2)
    for _ in range(n_stains):
        cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
        r = random.randint(15, 50)
        stain_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(stain_mask, (cx, cy), r, 255, -1)
        stain_mask = cv2.GaussianBlur(stain_mask, (31, 31), 0)
        darken = random.uniform(0.3, 0.6)
        alpha = (stain_mask.astype(np.float32) / 255.0 * darken)[..., None]
        img[:] = (img * (1 - alpha)).astype(np.uint8)
        mask[:] = np.maximum(mask, (stain_mask > 40).astype(np.uint8) * 255)
    return mask


def _add_holes(img, mask, n_holes=None):
    h, w = img.shape[:2]
    n_holes = n_holes or random.randint(0, 2)
    for _ in range(n_holes):
        cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
        r = random.randint(8, 30)
        hole_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(hole_mask, (cx, cy), r, 255, -1)
        fill_color = np.array([60, 50, 45])
        alpha = (hole_mask.astype(np.float32) / 255.0)[..., None]
        img[:] = (img * (1 - alpha) + fill_color * alpha).astype(np.uint8)
        mask[:] = np.maximum(mask, hole_mask)
    return mask


# ---- validation guard: run this before any full-scale generation ----
def validate_pair(clean, damaged, mask):
    diff = np.abs(clean.astype(np.int16) - damaged.astype(np.int16))
    mask_bool = mask > 127
    if mask_bool.sum() == 0:
        return
    diff_in_mask = diff[mask_bool].mean()
    assert diff_in_mask > 3, (
        f"BUG: no real difference inside mask (mean diff={diff_in_mask:.3f})"
    )
