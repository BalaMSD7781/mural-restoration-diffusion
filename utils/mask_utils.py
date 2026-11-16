"""
Mask generation and manipulation utilities.
Handles rendering lasso polygons into binary masks, computing bounding boxes,
combining multiple selections, and rendering overlays.
"""

from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw
import numpy as np


def generate_binary_mask_from_polygons(
    polygons: List[List[Tuple[int, int]]],
    image_size: Tuple[int, int]
) -> Image.Image:
    """
    Generate an 8-bit single-channel binary mask from one or more polygons.
    Background = 0 (black)
    Selected damaged region = 255 (white)

    Args:
        polygons: List of polygon coordinate lists [[(x1, y1), (x2, y2), ...], ...]
                  in original image resolution space.
        image_size: (width, height) matching original image dimensions.

    Returns:
        PIL.Image in mode 'L' (8-bit grayscale) with 0 and 255 values.
    """
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for poly in polygons:
        if len(poly) >= 3:
            # Solid filled polygon
            draw.polygon(poly, fill=255, outline=255)
        elif len(poly) == 2:
            # Line or stroke
            draw.line(poly, fill=255, width=5)
        elif len(poly) == 1:
            x, y = poly[0]
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=255)

    return mask


def extract_bounding_box(mask: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute the bounding box of non-zero (damaged) pixels in the mask.
    Returns:
        (x_min, y_min, x_max, y_max) or None if mask is empty.
    """
    mask_np = np.array(mask)
    non_zero = np.argwhere(mask_np > 0)

    if non_zero.size == 0:
        return None

    # non_zero coordinates are [row (y), col (x)]
    y_min, x_min = non_zero.min(axis=0)
    y_max, x_max = non_zero.max(axis=0)

    return int(x_min), int(y_min), int(x_max), int(y_max)


def is_mask_empty(mask: Image.Image) -> bool:
    """Check if the mask has any non-zero pixels."""
    bbox = extract_bounding_box(mask)
    return bbox is None


def apply_mask_addition(base_mask: Image.Image, added_mask: Image.Image) -> Image.Image:
    """
    Apply additive refinement to damage mask.
    refined_mask = base_mask OR user_added_mask
    Preserves single-channel uint8 format and original dimensions.
    """
    if base_mask.size != added_mask.size:
        added_mask = added_mask.resize(base_mask.size, Image.Resampling.NEAREST)
        
    base_arr = np.array(base_mask.convert("L"), dtype=np.uint8)
    added_arr = np.array(added_mask.convert("L"), dtype=np.uint8)
    
    combined = np.maximum(base_arr, added_arr)
    # Enforce strict binary values 0 and 255
    refined = np.where(combined > 127, 255, 0).astype(np.uint8)
    return Image.fromarray(refined, mode="L")


def apply_mask_removal(base_mask: Image.Image, removed_mask: Image.Image) -> Image.Image:
    """
    Apply subtractive refinement to remove false-positive damage detections.
    refined_mask = base_mask AND NOT user_removed_mask
    Preserves single-channel uint8 format and original dimensions.
    """
    if base_mask.size != removed_mask.size:
        removed_mask = removed_mask.resize(base_mask.size, Image.Resampling.NEAREST)
        
    base_arr = np.array(base_mask.convert("L"), dtype=np.uint8)
    removed_arr = np.array(removed_mask.convert("L"), dtype=np.uint8)
    
    # Zero out any pixel where user indicated removal
    subtracted = np.where(removed_arr > 127, 0, base_arr).astype(np.uint8)
    refined = np.where(subtracted > 127, 255, 0).astype(np.uint8)
    return Image.fromarray(refined, mode="L")


def compute_damage_pixel_stats(mask: Optional[Image.Image]) -> Dict[str, Any]:
    """
    Compute quantitative damage area statistics.
    Returns:
        {
            'damage_pixels': int,
            'total_pixels': int,
            'damage_percentage': float
        }
    """
    if mask is None:
        return {"damage_pixels": 0, "total_pixels": 0, "damage_percentage": 0.0}
        
    mask_np = np.array(mask.convert("L"))
    damage_count = int(np.count_nonzero(mask_np > 127))
    total_count = int(mask_np.size)
    pct = round((damage_count / total_count) * 100.0, 3) if total_count > 0 else 0.0
    
    return {
        "damage_pixels": damage_count,
        "total_pixels": total_count,
        "damage_percentage": pct
    }


def create_mask_overlay(
    image: Image.Image,
    mask: Image.Image,
    overlay_color: Tuple[int, int, int] = (230, 45, 45),  # Crimson damage highlight
    alpha: float = 0.5
) -> Image.Image:
    """
    Create a blended visual overlay of the damage mask on top of the original mural.
    Damaged regions are highlighted with a translucent tinted color.
    """
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    img_rgb = image.convert("RGBA")
    mask_np = np.array(mask) > 128

    # Create solid color layer
    color_layer = Image.new("RGBA", image.size, (*overlay_color, int(alpha * 255)))
    empty_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))

    # Mask for composite
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    tinted_overlay = Image.composite(color_layer, empty_layer, mask_pil)

    # Blend with original
    blended = Image.alpha_composite(img_rgb, tinted_overlay)
    return blended.convert("RGB")



def refine_predicted_mask(
    mask: Image.Image,
    close_px: int = 5,
    min_area: int = 30,
    dilate_px: int = 6,
) -> Image.Image:
    """Clean a raw U-Net mask before diffusion.

    1. morphological close   - joins fragmented crack segments
    2. small-component filter - drops isolated specks (false positives)
    3. dilation              - diffusion needs a margin so damaged edge pixels are re-drawn too
    """
    import cv2
    m = (np.array(mask.convert("L")) > 127).astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 1
        m = keep
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    return Image.fromarray((m * 255).astype(np.uint8), mode="L")


def mask_from_canvas_rgba(canvas_rgba: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    """Convert a drawable-canvas RGBA array (any stroke colour, alpha>0 = painted) to a binary mask."""
    alpha = canvas_rgba[..., 3] if canvas_rgba.shape[-1] == 4 else (canvas_rgba.sum(-1) > 0) * 255
    m = Image.fromarray(((alpha > 10).astype(np.uint8)) * 255, mode="L")
    return m.resize(size, Image.Resampling.NEAREST)
