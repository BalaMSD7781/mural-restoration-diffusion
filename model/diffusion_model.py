"""Diffusion stages of the restoration pipeline (plus a classical CPU preview).

Stage 1          SD 1.5 inpainting + mural LoRA (adapter "mural") fills the damaged region.
Stage 2          Canny edges of the Stage-1 result condition a Canny ControlNet inpainting pass with the
                 Canny fine-tuned LoRA (adapter "canny"), re-painting the region along continuous outlines.
Final diffusion  A low-strength SD inpainting pass with the mural LoRA over the damaged region plus a ring
                 of intact paint around it, blending the seams before post-processing.
Both LoRAs live on the one shared inpainting UNet; set_adapters() switches between them per stage.

All heavy imports are lazy so the rest of the project (and the tests) work without torch/diffusers.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import config

_PIPE: Dict[str, Any] = {}          # cache: "sd", "cn", "lora_loaded"


# ------------------------------------------------------------------------- device
def resolve_device() -> str:
    """cuda > mps > cpu, honouring MURAL_DEVICE (auto|cpu|cuda|mps)."""
    want = config.DEFAULT_DEVICE
    try:
        import torch
    except ImportError:
        return "cpu"
    if want in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if want in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def diffusion_available() -> bool:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------- helpers
def work_size(size: Tuple[int, int], device: str, max_side: Optional[int] = None) -> Tuple[int, int]:
    """Scale so the longest side <= max_side (512 on CPU, 768 elsewhere), multiples of 8."""
    max_side = max_side or (512 if device == "cpu" else 768)
    w, h = size
    s = min(1.0, max_side / max(w, h))
    nw, nh = max(64, int(w * s)), max(64, int(h * s))
    return nw - nw % 8, nh - nh % 8


def canny_map(image: Image.Image, low: int = config.CANNY_LOW, high: int = config.CANNY_HIGH) -> Image.Image:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return Image.fromarray(cv2.Canny(gray, low, high), mode="L")


# ------------------------------------------------------------------ classical
def classical_inpaint(image: Image.Image, mask: Image.Image, radius: int = 5) -> Image.Image:
    """Telea inpainting - instant, CPU-only; used for live preview and as the weakest baseline."""
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    m = (np.asarray(mask.convert("L").resize(image.size, Image.Resampling.NEAREST)) > 127).astype(np.uint8) * 255
    out = cv2.inpaint(bgr, m, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


# --------------------------------------------------------------- SD pipelines
def _load_adapter(pipe, name: str, path) -> None:
    key = f"lora_{name}"
    if _PIPE.get(key) is not None:
        return
    if not path.is_file():
        _PIPE[key] = False
        _PIPE[f"{key}_error"] = f"LoRA file not found: {path.name}"
        return
    try:
        pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=name)
        _PIPE[key] = True
    except Exception as exc:                               # keep running without this LoRA
        _PIPE[key] = False
        _PIPE[f"{key}_error"] = f"{type(exc).__name__}: {exc}"[:300]


def use_adapter(pipe, name: Optional[str], scale: float = 1.0) -> bool:
    """Activate one LoRA adapter (or none). Returns True if the adapter is active."""
    if name and _PIPE.get(f"lora_{name}"):
        pipe.set_adapters([name], adapter_weights=[scale])
        return True
    try:
        pipe.disable_lora()
    except Exception:
        pass
    return False


def load_sd_pipeline(use_lora: bool = True):
    """Load + cache the SD inpainting pipeline with the mural and Canny LoRAs. LoRA problems never abort loading."""
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    device = resolve_device()
    if "sd" not in _PIPE:
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            config.SD_MODEL_ID, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        _PIPE["sd"] = pipe
    pipe = _PIPE["sd"]
    if use_lora:
        _load_adapter(pipe, "mural", config.LORA_PATH)
        _load_adapter(pipe, "canny", config.CANNY_LORA_PATH)
    return pipe


def load_controlnet_pipeline(use_lora: bool = True):
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline

    pipe = load_sd_pipeline(use_lora)
    if "cn" not in _PIPE:
        dtype = pipe.unet.dtype
        controlnet = ControlNetModel.from_pretrained(config.CONTROLNET_ID, torch_dtype=dtype).to(resolve_device())
        comps = dict(pipe.components)
        comps.pop("image_encoder", None)
        cn_pipe = StableDiffusionControlNetInpaintPipeline(
            **{k: v for k, v in comps.items() if k != "controlnet"},
            controlnet=controlnet, requires_safety_checker=False,
        )
        cn_pipe.enable_attention_slicing()
        _PIPE["cn"] = cn_pipe
    return _PIPE["cn"]


def run_restoration(
    image: Image.Image,
    mask: Image.Image,
    mode: str = "sd",
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Image.Image], Dict[str, Any]]:
    """Generate restored pixels. Returns ({'stage1': img, ['stage2': img]}, metrics).

    Output images are at the *input* resolution and are NOT yet composited with the original;
    model.postprocess.composite does that so intact pixels stay untouched.
    """
    cfg = cfg or {}
    t0 = time.time()
    if mode == "classical":
        out = classical_inpaint(image, mask)
        return {"stage1": out}, {"backend": "classical (Telea) inpainting", "mode": mode,
                                 "device": "cpu", "execution_time_seconds": round(time.time() - t0, 2)}

    if not diffusion_available():
        raise RuntimeError("torch/diffusers not installed - use the 'classical' mode or install requirements.txt.")
    import torch

    device = resolve_device()
    seed = int(cfg.get("seed", config.DEFAULT_SEED))
    steps = int(cfg.get("steps", config.DEFAULT_INFERENCE_STEPS))
    guidance = float(cfg.get("guidance_scale", config.DEFAULT_GUIDANCE_SCALE))
    prompt = str(cfg.get("prompt", config.DEFAULT_PROMPT))
    negative = str(cfg.get("negative_prompt", config.DEFAULT_NEGATIVE_PROMPT))
    use_lora = bool(cfg.get("use_lora", True))
    lora_scale = float(cfg.get("lora_scale", config.LORA_SCALE))

    size = work_size(image.size, device, cfg.get("max_side"))
    img_w = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    msk_w = mask.convert("L").resize(size, Image.Resampling.NEAREST)
    gen = torch.Generator("cpu").manual_seed(seed)

    pipe = load_sd_pipeline(use_lora)
    lora_on = use_lora and use_adapter(pipe, "mural", lora_scale)
    kw = dict(prompt=prompt, negative_prompt=negative, height=size[1], width=size[0],
              num_inference_steps=steps, guidance_scale=guidance, generator=gen)

    strength = float(cfg.get("strength", 1.0))         # < 1: refine the given pixels (damage-aware tiers)
    if strength < 0.999:
        kw["strength"] = strength
    t1 = time.time()
    s1 = pipe(image=img_w, mask_image=msk_w, **kw).images[0]
    stage1_time = time.time() - t1
    images = {"stage1": s1.resize(image.size, Image.Resampling.LANCZOS)}
    metrics: Dict[str, Any] = {
        "backend": "Stage 1: SD 1.5 inpainting + mural LoRA", "mode": mode, "model": config.SD_MODEL_ID,
        "device": device, "diffusion_steps": steps, "guidance_scale": guidance, "seed": seed,
        "work_resolution": f"{size[0]}x{size[1]}",
        "lora_applied": lora_on, "lora_scale": lora_scale if lora_on else None,
        "lora_error": _PIPE.get("lora_mural_error") if use_lora and not lora_on else None,
        "stage1_seconds": round(stage1_time, 2), "stage1_strength": strength,
    }

    if mode in ("sd_edge", "full"):
        try:
            cn_pipe = load_controlnet_pipeline(use_lora)
            canny_on = use_lora and use_adapter(pipe, "canny", float(cfg.get("canny_lora_scale", config.CANNY_LORA_SCALE)))
            edges = canny_map(s1)
            control = Image.merge("RGB", [edges] * 3)
            t2 = time.time()
            kw2 = dict(prompt=prompt, negative_prompt=negative, height=size[1], width=size[0],
                       num_inference_steps=steps, guidance_scale=guidance,
                       generator=torch.Generator("cpu").manual_seed(seed + 1),
                       strength=float(cfg.get("stage2_strength", config.STAGE2_STRENGTH)),
                       controlnet_conditioning_scale=float(cfg.get("edge_scale", config.EDGE_SCALE)))
            s2 = cn_pipe(image=s1, mask_image=msk_w, control_image=control, **kw2).images[0]
            images["stage2"] = s2.resize(image.size, Image.Resampling.LANCZOS)
            images["canny"] = edges.resize(image.size, Image.Resampling.NEAREST)
            metrics["stage2_seconds"] = round(time.time() - t2, 2)
            metrics["canny_lora_applied"] = canny_on
            if use_lora and not canny_on:
                metrics["canny_lora_error"] = _PIPE.get("lora_canny_error")
            metrics["backend"] = "Stage 1 (mural LoRA) + Stage 2 (Canny ControlNet + Canny LoRA)"
        except Exception as exc:  # never lose the Stage-1 result because Stage 2 failed
            metrics["stage2_error"] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            use_adapter(pipe, "mural", lora_scale) if use_lora else None
    metrics["execution_time_seconds"] = round(time.time() - t0, 2)
    return images, metrics


def final_diffusion(image: Image.Image, mask: Image.Image, cfg: Optional[Dict[str, Any]] = None
                    ) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """Low-strength SD inpainting pass (mural LoRA) over the damaged region + a ring of intact paint.

    `image` is the composited result of Stages 1-2. Returns (generated image at input size, ring mask, metrics).
    """
    import torch
    cfg = cfg or {}
    t0 = time.time()
    device = resolve_device()
    ring = int(cfg.get("final_ring_px", config.FINAL_RING_PX))
    m = (np.asarray(mask.convert("L").resize(image.size, Image.Resampling.NEAREST)) > 127).astype(np.uint8)
    if ring > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1)))
    ring_mask = Image.fromarray(m * 255)
    size = work_size(image.size, device, cfg.get("max_side"))
    pipe = load_sd_pipeline(bool(cfg.get("use_lora", True)))
    lora_on = bool(cfg.get("use_lora", True)) and use_adapter(pipe, "mural", float(cfg.get("lora_scale", config.LORA_SCALE)))
    strength = float(cfg.get("final_strength", config.FINAL_STRENGTH))
    out = pipe(prompt=str(cfg.get("prompt", config.DEFAULT_PROMPT)),
               negative_prompt=str(cfg.get("negative_prompt", config.DEFAULT_NEGATIVE_PROMPT)),
               image=image.convert("RGB").resize(size, Image.Resampling.LANCZOS),
               mask_image=ring_mask.resize(size, Image.Resampling.NEAREST),
               height=size[1], width=size[0], strength=strength,
               num_inference_steps=int(cfg.get("steps", config.DEFAULT_INFERENCE_STEPS)),
               guidance_scale=float(cfg.get("guidance_scale", config.DEFAULT_GUIDANCE_SCALE)),
               generator=torch.Generator("cpu").manual_seed(int(cfg.get("seed", config.DEFAULT_SEED)) + 2)).images[0]
    return (out.resize(image.size, Image.Resampling.LANCZOS), ring_mask,
            {"final_strength": strength, "final_ring_px": ring, "final_lora_applied": lora_on,
             "final_seconds": round(time.time() - t0, 2)})


# ------------------------------------------------------- backwards-compatible shim
def run_one_stage_diffusion(image, mask, canny=None, config_dict=None):
    imgs, metrics = run_restoration(image, mask, "sd", config_dict)
    return imgs["stage1"], metrics
