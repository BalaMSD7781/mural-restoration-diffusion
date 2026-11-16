"""Validate and manifest paired mural images and binary damage masks.

The validator intentionally does not infer a research split from filenames.
Use ``--split-file`` to provide a CSV with ``image,split`` assignments, or
leave the split empty and resolve the split before publishing metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SPLITS = {"train", "validation", "test"}


def _load_split_file(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    assignments: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image = (row.get("image") or "").strip()
            split = (row.get("split") or "").strip().lower()
            if not image or split not in SPLITS:
                raise ValueError(
                    f"Invalid split row in {path}: expected image and one of "
                    f"{sorted(SPLITS)}, got {row!r}."
                )
            if image in assignments:
                raise ValueError(f"Duplicate image assignment: {image}")
            assignments[image] = split
    return assignments


def _mask_summary(mask_path: Path) -> dict:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    unique = np.unique(mask)
    binary = set(unique.tolist()).issubset({0, 255})
    foreground = int(np.count_nonzero(mask > 127))
    return {
        "mask_width": int(mask.shape[1]),
        "mask_height": int(mask.shape[0]),
        "mask_unique_values": [int(value) for value in unique],
        "mask_is_binary": binary,
        "mask_damage_pixels": foreground,
        "mask_damage_percentage": round(float(mask.size and foreground / mask.size * 100), 4),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    uploads_dir: Path,
    masks_dir: Path,
    split_file: Optional[Path] = None,
) -> dict:
    assignments = _load_split_file(split_file)
    records = []
    errors = []
    seen_assignments = set()

    image_paths = sorted(
        path
        for path in uploads_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        and path.stem.endswith("_original")
    )
    for image_path in image_paths:
        image_id = image_path.stem[: -len("_original")]
        mask_path = masks_dir / f"{image_id}_mask.png"
        record = {
            "image": image_path.name,
            "mask": mask_path.name,
            "image_id": image_id,
            "split": assignments.get(image_path.name),
            "image_sha256": _sha256(image_path),
        }
        try:
            with Image.open(image_path) as image:
                image_size = image.size
                image_mode = image.mode
            record.update(
                {
                    "image_width": image_size[0],
                    "image_height": image_size[1],
                    "image_mode": image_mode,
                }
            )
            if not mask_path.is_file():
                raise ValueError(f"Missing mask: {mask_path.name}")
            record.update(_mask_summary(mask_path))
            record["mask_sha256"] = _sha256(mask_path)
            if (record["image_width"], record["image_height"]) != (
                record["mask_width"],
                record["mask_height"],
            ):
                raise ValueError("Image and mask dimensions differ")
            if not record["mask_is_binary"]:
                raise ValueError("Mask contains values other than 0 and 255")
            if record["split"] is not None:
                seen_assignments.add(image_path.name)
        except (OSError, ValueError) as exc:
            record["error"] = str(exc)
            errors.append({"image": image_path.name, "error": str(exc)})
        records.append(record)

    unknown_assignments = sorted(set(assignments) - seen_assignments)
    errors.extend(
        {"image": image, "error": "Split file references an unknown image"}
        for image in unknown_assignments
    )
    split_counts = {
        split: sum(record.get("split") == split for record in records)
        for split in sorted(SPLITS)
    }
    return {
        "schema_version": 1,
        "uploads_dir": str(uploads_dir),
        "masks_dir": str(masks_dir),
        "split_file": str(split_file) if split_file else None,
        "records": records,
        "summary": {
            "images": len(records),
            "valid_pairs": sum("error" not in record for record in records),
            "errors": len(errors),
            "split_counts": split_counts,
            "unassigned": sum(record.get("split") is None for record in records),
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--uploads-dir", type=Path, default=None)
    parser.add_argument("--masks-dir", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    uploads_dir = (args.uploads_dir or project_dir / "data" / "uploads").resolve()
    masks_dir = (args.masks_dir or project_dir / "data" / "masks").resolve()
    if not uploads_dir.is_dir():
        raise SystemExit(f"Uploads directory does not exist: {uploads_dir}")
    if not masks_dir.is_dir():
        raise SystemExit(f"Masks directory does not exist: {masks_dir}")

    result = build_manifest(uploads_dir, masks_dir, args.split_file)
    output = args.output or project_dir / "data" / "metadata" / "dataset_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **result["summary"]}))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
