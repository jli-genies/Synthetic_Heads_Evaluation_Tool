#!/usr/bin/env python3
"""Run GenieSAM locally on a single image and write ISAT JSON.

Intended to be launched by the evaluation GUI via QProcess with the GenieSAM
venv Python so Torch/SAM3 stay out of the GUI process.

Example:
    python tools/run_geniesam.py \\
      --image renders/asset/front.png \\
      --output-dir renders/asset/segmentation \\
      --geniesam-root C:/Users/auror/Documents/Github/GenieSAM \\
      --checkpoint C:/path/to/sam3.pth \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local GenieSAM inference to ISAT JSON")
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for ISAT JSON (writes <stem>.json)",
    )
    parser.add_argument(
        "--geniesam-root",
        required=True,
        type=Path,
        help="Path to GenieSAM repository root",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SAM3_CHECKPOINT", ""),
        help="Local SAM3 .pth path (or set SAM3_CHECKPOINT)",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Override category list (defaults to GenieSAM config defaults)",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Output JSON basename without extension (default: image stem)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    geniesam_root = args.geniesam_root.resolve()
    basename = args.basename or image_path.stem

    if not image_path.is_file():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 2
    if not geniesam_root.is_dir():
        print(f"ERROR: GenieSAM root not found: {geniesam_root}", file=sys.stderr)
        return 2

    checkpoint = (args.checkpoint or "").strip()
    if not checkpoint:
        print(
            "ERROR: --checkpoint or SAM3_CHECKPOINT env var is required",
            file=sys.stderr,
        )
        return 2
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(geniesam_root))
    try:
        import torch
        from segmentation.segmentation import (
            DEFAULT_CATEGORIES,
            load_image,
            load_sam3_model,
            segment_image_in_json,
        )
    except ImportError as error:
        print(f"ERROR: failed to import GenieSAM: {error}", file=sys.stderr)
        return 2

    categories = args.categories or list(DEFAULT_CATEGORIES)
    # Prefer Core_Face over Face when both appear in defaults.
    if "Core_Face" in categories and "Face" in categories:
        categories = [c for c in categories if c != "Face"]

    print(f"Loading image: {image_path}")
    image_np = load_image(str(image_path))
    if image_np is None:
        print(f"ERROR: could not load image: {image_path}", file=sys.stderr)
        return 2

    print(f"Loading SAM3 on {args.device} from {checkpoint_path}")
    model, image_transform = load_sam3_model(
        device=args.device,
        image_size=args.image_size,
        sam3_ckpt=str(checkpoint_path),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Segmenting categories: {categories}")
    with torch.no_grad():
        if args.device == "cuda" and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_json, _masks = segment_image_in_json(
                    model=model,
                    image_transform=image_transform,
                    image_np=image_np,
                    categories=categories,
                    device=args.device,
                    image_size=args.image_size,
                    save_folder=str(output_dir),
                    image_base_name=basename,
                )
        else:
            output_json, _masks = segment_image_in_json(
                model=model,
                image_transform=image_transform,
                image_np=image_np,
                categories=categories,
                device=args.device,
                image_size=args.image_size,
                save_folder=str(output_dir),
                image_base_name=basename,
            )

    if not output_json:
        print("ERROR: segmentation produced no output", file=sys.stderr)
        return 1

    out_path = output_dir / f"{basename}.json"
    # segment_image_in_json already writes the file; ensure it exists and report.
    if not out_path.is_file():
        out_path.write_text(json.dumps(output_json, indent=4), encoding="utf-8")

    n_objects = len(output_json.get("objects", []))
    print(f"OK: wrote {out_path} ({n_objects} objects)")
    print(json.dumps({"status": "success", "path": str(out_path), "objects": n_objects}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
