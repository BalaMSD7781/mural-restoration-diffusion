"""Convert the raw Kaggle LoRA export (bare UNet keys) to the diffusers 'unet.' prefixed format.

The training notebook saved with ``unet.save_attn_procs``; its keys look like
``down_blocks.0.attentions.0...to_q.lora_A.weight`` so ``pipe.load_lora_weights`` reports
"No LoRA keys associated to UNet2DConditionModel found with the prefix='unet'" and silently loads nothing.

Usage:  python -m tools.fix_lora_keys models/dunhuang_mural_lora_step500.safetensors
Writes  models/dunhuang_mural_lora_step500_fixed.safetensors
"""
import sys
from pathlib import Path

from safetensors.numpy import load_file, save_file


def fix(src: Path, dst: Path) -> int:
    tensors = load_file(str(src))
    out = {(k if k.startswith("unet.") else f"unet.{k}"): v for k, v in tensors.items()}
    save_file(out, str(dst), metadata={"format": "pt"})
    return len(out)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_fixed.safetensors")
    print(f"wrote {dst} with {fix(src, dst)} tensors")
