"""Pre-compute demo results for the sample images (run this on a GPU machine or Kaggle).

For every image in data/samples/ it segments the damage, then runs the chosen restoration modes and
stores the final result in data/samples/cache/<name>__<mode>.png. In the app, tick
"Use pre-computed result" and the CPU-only laptop shows these instantly, so the live demo never
depends on a slow CPU diffusion run.

    python -m tools.precompute_demo --steps 30
"""
import argparse
import json
import time

from PIL import Image

import config
from model.damage_detector import predict_damage_mask
from model.inference import restore_mural


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["full"], choices=list(config.RESTORE_MODES))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--model", default=config.DEFAULT_SEGMENTATION_MODEL)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    a = ap.parse_args()

    files = sorted(p for p in config.SAMPLE_DIR.glob("*") if p.suffix.lower().lstrip(".") in config.SUPPORTED_FORMATS)
    if not files:
        raise SystemExit(f"No sample images in {config.SAMPLE_DIR}")
    for path in files:
        image = Image.open(path).convert("RGB")
        mask, prob, stats = predict_damage_mask(image, threshold=a.threshold, model_key=a.model)
        print(f"{path.name}: damage {stats['damage_percentage']}%")
        mask.save(config.CACHE_DIR / f"{path.stem}__mask.png")
        for mode in a.modes:
            t = time.time()
            cfg = {"restore_mode": mode, "steps": a.steps, "seed": a.seed, "damage_aware": True,
                   "threshold": stats.get("threshold", 0.5)}
            res = restore_mural(image, mask, config_dict=cfg, prob_map=prob)
            res.restored_image.save(config.CACHE_DIR / f"{path.stem}__{mode}.png")
            for k, im in res.stages.items():
                if k in ("types", "prefill", "stage1", "stage2", "final_diffusion", "final"):
                    im.save(config.CACHE_DIR / f"{path.stem}__{mode}__{k}.png")
            keep = {k: v for k, v in res.metrics.items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
            (config.CACHE_DIR / f"{path.stem}__{mode}__metrics.json").write_text(json.dumps(keep, default=str, indent=1))
            errs = {k: v for k, v in res.metrics.items() if k.endswith("_error") and v}
            print(f"  {mode}: {time.time() - t:.1f}s  {errs or ''}")


if __name__ == "__main__":
    main()
