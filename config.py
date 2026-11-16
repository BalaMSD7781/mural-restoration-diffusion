"""
Configuration for the AI Mural Restoration System.

Paths, model registry, defaults and environment-variable overrides.
The segmentation model is *swappable*: add an entry to SEGMENTATION_MODELS (or drop an
``attention_unet.keras`` file into ``models/``) and select it in the app sidebar or with
the SEGMENTATION_MODEL environment variable. Input channels (RGB=3 or RGB+Canny=4) are
detected from the loaded model, so a different architecture needs no code change.
"""

from pathlib import Path
import json
import os

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- paths
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MASK_DIR = DATA_DIR / "masks"
CROP_DIR = DATA_DIR / "crops"
OUTPUT_DIR = DATA_DIR / "outputs"
RAW_DIR = DATA_DIR / "raw"
CANNY_DIR = DATA_DIR / "canny"
METADATA_DIR = DATA_DIR / "metadata"
SAMPLE_DIR = DATA_DIR / "samples"
CACHE_DIR = SAMPLE_DIR / "cache"          # pre-computed demo results (see tools/precompute_demo.py)
RESULTS_DIR = BASE_DIR / "results"        # evaluation JSON produced by the Kaggle notebooks
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"

for _d in (UPLOAD_DIR, MASK_DIR, CROP_DIR, OUTPUT_DIR, RAW_DIR, CANNY_DIR,
           METADATA_DIR, SAMPLE_DIR, CACHE_DIR, RESULTS_DIR, ASSETS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SUPPORTED_FORMATS = ["png", "jpg", "jpeg"]

# ------------------------------------------------------------------ device selection
# "auto" -> CUDA if available, else MPS, else CPU.   Override: MURAL_DEVICE=cpu|cuda|mps
DEFAULT_DEVICE = os.getenv("MURAL_DEVICE", "auto").lower()

# ------------------------------------------------------------------- segmentation
# Damage segmentation uses the team's Attention U-Net. Put the trained file at models/attention_unet.keras.
# Input size and channels (RGB = 3, RGB + Canny = 4) are read from the model, so no code change is needed.
SEGMENTATION_MODELS = {
    "attention_unet": {
        "label": "Attention U-Net",
        "path": MODELS_DIR / "attention_unet.keras",
        "threshold": 0.5,
    },
}
DEFAULT_SEGMENTATION_MODEL = "attention_unet"
if os.getenv("SEGMENTATION_MODEL_PATH"):
    SEGMENTATION_MODELS["attention_unet"]["path"] = Path(os.environ["SEGMENTATION_MODEL_PATH"])

# Tuned thresholds are written by notebooks/01_segmentation_eval.ipynb to
# models/thresholds.json, e.g. {"attention_unet": 0.4}. They override the
# defaults above so the demo always uses the threshold chosen on the validation data.
_threshold_file = MODELS_DIR / "thresholds.json"
if _threshold_file.is_file():
    try:
        for _k, _v in json.loads(_threshold_file.read_text()).items():
            if _k in SEGMENTATION_MODELS:
                SEGMENTATION_MODELS[_k]["threshold"] = float(_v)
    except (ValueError, OSError):
        pass


def available_segmentation_models():
    """Registry entries whose weight file exists on disk."""
    return {k: v for k, v in SEGMENTATION_MODELS.items() if Path(v["path"]).is_file()}


# Backwards-compatible names used by older modules / tools.
DAMAGE_MODEL_PATH = Path(SEGMENTATION_MODELS[DEFAULT_SEGMENTATION_MODEL]["path"])
DAMAGE_THRESHOLD = float(SEGMENTATION_MODELS[DEFAULT_SEGMENTATION_MODEL]["threshold"])
UNET_INPUT_SHAPE = (384, 384)          # fallback only; real size is read from the model
UNET_MODEL_PATH = DAMAGE_MODEL_PATH

# --------------------------------------------------------------------- diffusion
SD_MODEL_ID = os.getenv("SD_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-inpainting")
CONTROLNET_ID = os.getenv("CONTROLNET_ID", "lllyasviel/control_v11p_sd15_canny")
# Stage 1: mural LoRA (notebooks/04, trained on the 5,096 MuralDH murals).
LORA_PATH = Path(os.getenv("MURAL_LORA_PATH", str(MODELS_DIR / "dunhuang_inpaint_lora.safetensors")))
LORA_SCALE = float(os.getenv("MURAL_LORA_SCALE", "1.0"))
# Stage 2: Canny fine-tuned LoRA (notebooks/05, trained with the Canny ControlNet on the Canny maps of the same murals).
CANNY_LORA_PATH = Path(os.getenv("CANNY_LORA_PATH", str(MODELS_DIR / "dunhuang_canny_lora.safetensors")))
CANNY_LORA_SCALE = float(os.getenv("CANNY_LORA_SCALE", "1.0"))
STAGE2_STRENGTH = 0.6          # how much of the damaged region Stage 2 re-paints
EDGE_SCALE = 0.8               # ControlNet conditioning scale
# Final diffusion: a low-strength pass with the mural LoRA over the damaged region plus a ring around it,
# so the seams between restored and original paint blend before post-processing.
FINAL_STRENGTH = 0.3
FINAL_RING_PX = 8
LORA_TRIGGER = "dunhuang_mural_style"

DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SEED = 42
DEFAULT_PROMPT = (
    "dunhuang_mural_style, a Dunhuang Buddhist mural painting, faded mineral pigments, "
    "aged plaster wall, hand-painted fine lines"
)
DEFAULT_NEGATIVE_PROMPT = (
    "modern objects, text, watermark, photo, 3d render, oversaturated, blurry, "
    "cartoon, signature, frame"
)

# Restoration modes. "full" is the proposed pipeline; "classical" is the CPU-only preview.
RESTORE_MODES = {
    "full": "Full pipeline: Stage 1 (mural LoRA) → Stage 2 (Canny LoRA) → final diffusion",
    "classical": "Classical preview (CPU, no diffusion)",
}
PIPELINE_STEPS = [
    "Attention U-Net damage segmentation",
    "Damage-aware module (damage type + severity → per-region strength)",
    "Stage 1 diffusion: SD 1.5 inpainting + mural LoRA",
    "Stage 2 diffusion: Canny ControlNet + Canny fine-tuned LoRA",
    "Final diffusion: low-strength blending pass",
    "Post-processing: colour, texture, sharpening",
]

# Canny thresholds selected in the Second Review.
CANNY_LOW, CANNY_HIGH = 100, 200

# ------------------------------------------------------------------------- UI theme
HERITAGE_PALETTE = {
    "primary": "#8B3A2B",
    "secondary": "#C59B27",
    "dark_accent": "#1F2421",
    "light_bg": "#FBF9F5",
    "card_bg": "#FFFFFF",
    "success": "#2E7D32",
    "warning": "#ED6C02",
    "border": "#E5DEC9",
}
