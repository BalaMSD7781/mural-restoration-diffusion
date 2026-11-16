"""Environment smoke test - run this FIRST on the demo laptop (and on Kaggle).

    python -m tools.smoke_test

Checks torch/CUDA, the Attention U-Net file, both LoRAs, and runs the full diffusion pipeline
(Stage 1 -> Stage 2 -> final diffusion) on a synthetic image with 8 steps; prints timings.
"""
import time

from PIL import Image, ImageDraw

import config
from model.diffusion_model import diffusion_available, resolve_device
from model.inference import restore_mural


def main():
    seg = config.SEGMENTATION_MODELS[config.DEFAULT_SEGMENTATION_MODEL]
    print("device:", resolve_device(), "| diffusers/torch available:", diffusion_available())
    print("Attention U-Net:", seg["path"].name, "exists:", seg["path"].is_file())
    print("Mural LoRA:", config.LORA_PATH.name, "exists:", config.LORA_PATH.is_file())
    print("Canny LoRA:", config.CANNY_LORA_PATH.name, "exists:", config.CANNY_LORA_PATH.is_file())
    if not diffusion_available():
        raise SystemExit("Install requirements.txt (torch, diffusers, transformers, accelerate, peft).")
    img = Image.new("RGB", (512, 512), (170, 130, 90))
    ImageDraw.Draw(img).ellipse([120, 120, 380, 380], outline=(90, 40, 30), width=10)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse([200, 200, 320, 320], fill=255)
    t = time.time()
    res = restore_mural(img, mask, config_dict={"restore_mode": "full", "steps": 8})
    m = res.metrics
    print(f"full pipeline: {time.time() - t:.1f}s | stages: {list(res.stages)}")
    print("  mural LoRA:", m.get("lora_applied"), m.get("lora_error") or "")
    print("  Canny LoRA:", m.get("canny_lora_applied"), m.get("canny_lora_error") or "", m.get("stage2_error") or "")
    print("  final diffusion:", m.get("final_seconds"), "s", m.get("final_diffusion_error") or "")
    res.restored_image.save("smoke_full.png")
    print("OK - open smoke_full.png to eyeball the output.")


if __name__ == "__main__":
    main()
