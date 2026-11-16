"""Evaluate segmentation thresholds on paired mural images and masks.

This evaluates the current checkpoint; it does not train a model or claim that
the discovered files are an independent research test set. Use ``--split test``
for a deterministic holdout based on file names and report the split alongside
any published metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

from evaluation.segmentation_metrics import binary_segmentation_metrics
from model.damage_detector import predict_damage_mask


def _paired_files(
    uploads_dir: Path,
    masks_dir: Path,
) -> List[Tuple[Path, Path]]:
    pairs = []
    for image_path in sorted(uploads_dir.glob("*_original.*")):
        mask_path = masks_dir / f"{image_path.stem[:-len('_original')]}_mask.png"
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def _split_name(image_path: Path) -> str:
    digest = hashlib.sha256(image_path.stem.encode("utf-8")).digest()
    return "test" if digest[0] % 5 == 0 else "train"


def _thresholds(start: float, stop: float, step: float) -> Iterable[float]:
    value = start
    while value <= stop + 1e-9:
        yield round(value, 4)
        value += step


def evaluate(
    pairs: List[Tuple[Path, Path]],
    thresholds: Iterable[float],
) -> List[dict]:
    rows = []
    for image_path, mask_path in pairs:
        image = Image.open(image_path).convert("RGB")
        target = np.asarray(Image.open(mask_path).convert("L")) > 127
        _, probability, _ = predict_damage_mask(image, threshold=1.0)
        for threshold in thresholds:
            prediction = probability >= threshold
            metrics = binary_segmentation_metrics(prediction, target)
            rows.append(
                {
                    "image": image_path.name,
                    "target_mask": mask_path.name,
                    "threshold": threshold,
                    **metrics,
                    "target_damage_percentage": float(target.mean() * 100),
                    "predicted_damage_percentage": float(prediction.mean() * 100),
                }
            )
    return rows


def _aggregate(rows: List[dict]) -> List[dict]:
    by_threshold = {}
    for row in rows:
        by_threshold.setdefault(row["threshold"], []).append(row)
    return [
        {
            "threshold": threshold,
            "samples": len(items),
            **{
                metric: float(np.mean([item[metric] for item in items]))
                for metric in ("dice", "iou", "precision", "recall")
            },
        }
        for threshold, items in sorted(by_threshold.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--split", choices=("all", "train", "test"), default="test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--threshold-start", type=float, default=0.1)
    parser.add_argument("--threshold-stop", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.1)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    pairs = _paired_files(project_dir / "data" / "uploads", project_dir / "data" / "masks")
    if args.split != "all":
        pairs = [pair for pair in pairs if _split_name(pair[0]) == args.split]
    if not pairs:
        raise SystemExit("No paired *_original image and *_mask.png files were found.")

    rows = evaluate(pairs, list(_thresholds(
        args.threshold_start, args.threshold_stop, args.threshold_step
    )))
    result = {
        "split": args.split,
        "samples": [image_path.name for image_path, _ in pairs],
        "per_image": rows,
        "aggregate": _aggregate(rows),
    }
    output = args.output or project_dir / "data" / "metadata" / "segmentation_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "csv": str(csv_path), "samples": len(pairs)}))


if __name__ == "__main__":
    main()
