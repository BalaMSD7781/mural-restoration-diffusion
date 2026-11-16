"""End-to-end restoration pipeline.

Attention U-Net mask -> mask refinement -> damage-aware module -> Stage 1 (SD + mural LoRA)
-> Stage 2 (Canny ControlNet + Canny LoRA) -> final diffusion (blending pass) -> post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time

from PIL import Image

import config
from model.diffusion_model import run_restoration, canny_map, final_diffusion
from model.damage_aware import restore_damage_aware
from model.postprocess import composite, postprocess
from utils.mask_utils import refine_predicted_mask


@dataclass
class MuralRestorationResult:
    restored_image: Image.Image                 # final output (after post-processing)
    raw_restored_image: Image.Image             # generated pixels composited, *before* post-processing
    canny_map: Optional[Image.Image]
    processing_time: float
    mode: str
    model_name: str
    metrics: Dict[str, Any]
    predicted_mask: Optional[Image.Image] = None
    refined_mask: Optional[Image.Image] = None
    stages: Dict[str, Image.Image] = field(default_factory=dict)


def restore_mural(
    image: Image.Image,
    mask: Optional[Image.Image] = None,
    selected_region: Optional[Image.Image] = None,   # kept for API compatibility
    mode: str = "automatic",
    config_dict: Optional[Dict[str, Any]] = None,
    canny_map_in: Optional[Image.Image] = None,
    prob_map: Any = None,
    **kwargs: Any,
) -> MuralRestorationResult:
    """Restore `image` inside `mask`.

    config_dict keys: restore_mode ('full' = proposed pipeline | 'classical' = CPU preview), steps, guidance_scale, seed,
    prompt, negative_prompt, use_lora, lora_scale, stage2_strength, edge_scale,
    refine_mask (bool) + close_px/min_area/dilate_px, post (dict for postprocess()), max_side,
    damage_aware (bool, novelty: per-region type routing + severity-adaptive strength; uses prob_map),
    threshold (for severity), crack_backend ('diffusion'|'classical').
    """
    started = time.time()
    cfg = dict(config_dict or kwargs.get("config") or {})
    restore_mode = cfg.get("restore_mode", "full")
    if mask is None:
        from model.damage_detector import predict_damage_mask
        mask, prob_map, _ = predict_damage_mask(image)
    predicted_mask = mask

    used_mask = mask
    if cfg.get("refine_mask", True):
        used_mask = refine_predicted_mask(
            mask, close_px=cfg.get("close_px", 5), min_area=cfg.get("min_area", 30),
            dilate_px=cfg.get("dilate_px", 6),
        )

    if cfg.get("damage_aware", True):
        margin = int(cfg.get("dilate_px", 6)) if cfg.get("refine_mask", True) else 0
        images, gen_metrics = restore_damage_aware(image, used_mask, prob_map, restore_mode, {**cfg, "margin_px": margin})
    else:
        images, gen_metrics = run_restoration(image, used_mask, restore_mode, cfg)
    gen_stage = images.get("stage2", images["stage1"])
    post_cfg = cfg.get("post")
    if post_cfg is None:
        post_cfg = {"color": True, "texture": True, "sharpen": True, "super_res": False}
    feather_px = post_cfg.get("feather_px", 6) if isinstance(post_cfg, dict) else 6
    raw = composite(image, gen_stage, used_mask, feather_px=feather_px)        # after Stages 1-2

    # ---- final diffusion: low-strength blending pass over the damage + a ring of intact paint
    final_in, final_mask, fd_metrics = gen_stage, used_mask, {}
    if restore_mode != "classical" and cfg.get("final_diffusion", True):
        try:
            fd_img, fd_mask, fd_metrics = final_diffusion(raw, used_mask, cfg)
            final_in, final_mask = fd_img, fd_mask
        except Exception as exc:
            fd_metrics = {"final_diffusion_error": f"{type(exc).__name__}: {exc}"[:300]}
    after_final = composite(image, final_in, final_mask, feather_px=feather_px)

    # ---- post-processing (composite keeps every intact pixel bit-exact outside the ring)
    if post_cfg is False:
        final, post_info = after_final, {"stages": ["composite"]}
    else:
        final, post_info = postprocess(image, final_in, final_mask, post_cfg)
    gen_metrics = {**gen_metrics, **fd_metrics}

    stages = {"stage1": composite(image, images["stage1"], used_mask)}
    if "stage2" in images:
        stages["stage2"] = composite(image, images["stage2"], used_mask)
    for extra in ("prefill", "types"):
        if extra in images:
            stages[extra] = images[extra]
    stages["raw"] = raw
    if fd_metrics and "final_diffusion_error" not in fd_metrics:
        stages["final_diffusion"] = after_final
    stages["final"] = final
    edge = images.get("canny") or canny_map(final if final.size == image.size else raw)

    elapsed = round(time.time() - started, 2)
    return MuralRestorationResult(
        restored_image=final,
        raw_restored_image=raw,
        canny_map=edge,
        processing_time=elapsed,
        mode=restore_mode,
        model_name=gen_metrics.get("backend", restore_mode),
        metrics={**gen_metrics, "post_stages": post_info["stages"], "total_seconds": elapsed,
                 "refined_mask": bool(cfg.get("refine_mask", True)),
                 "damage_aware": bool(cfg.get("damage_aware", True))},
        predicted_mask=predicted_mask,
        refined_mask=used_mask,
        stages=stages,
    )
