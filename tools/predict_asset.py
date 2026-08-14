#!/usr/bin/env python3
"""Score a single, unlabeled asset with the trained good/bad model.

Given just an asset filename (the same naming convention used in
lists/good.json / lists/bad.json), this:
  1. Derives its authored parent from the filename and loads that parent's
     landmarks from landmarks_raw/ (same as tools/build_training_dataset.py).
  2. Resolves the asset's file on disk under {dataset-root}/{variation-folder}/
     (searches every variation_* folder if --variation-folder isn't given).
  3. Runs it through blender/extract_landmarks_variation.py to get its own
     landmark positions.
  4. Computes the same dist_/ratio_/absdiff_/pctdiff_/chaos_ feature row used
     for training (landmarks/feature_row.py) and scores it with the model
     saved by tools/train_classifier.py.

This is inference on the asset's 3D geometry + generation metadata, not on
the rendered PNGs in renders/ -- those aren't a model input.

Example:
    python tools/predict_asset.py auth_asian_f_0022_frame_0004_subdiv_head.glb --variation-folder variation_long_faces_002
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from landmarks.feature_row import compute_feature_row  # noqa: E402
from tools.build_training_dataset import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    find_blender_executable,
    parent_stem_for,
    resolve_variant_path,
)

DEFAULT_MODEL = PROJECT_ROOT / "landmarks" / "model_hist_gb.joblib"
DEFAULT_RAW_DIR = PROJECT_ROOT / "landmarks_raw"
DEFAULT_VARIANT_RAW_DIR = PROJECT_ROOT / "landmarks_raw_variants"

# Only these two features carried any signal in the model trained on
# landmarks/training_dataset.csv (see permutation importance in
# tools/train_classifier.py) -- everything else is along for the ride.
DOMINANT_FEATURES = ("chaos_JawBind", "chaos_MouthBind")


def find_asset(dataset_root: Path, name: str, variation_folder: str | None) -> tuple[Path, str]:
    if variation_folder:
        path = resolve_variant_path(dataset_root, variation_folder, name)
        if path is None:
            raise FileNotFoundError(f"'{name}' not found under {dataset_root / variation_folder}")
        return path, variation_folder

    matches: list[tuple[Path, str]] = []
    for folder in sorted(p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("variation_")):
        found = list(folder.rglob(name))
        matches.extend((f, folder.name) for f in found)
    if not matches:
        raise FileNotFoundError(f"'{name}' not found under any variation_* folder in {dataset_root}")
    if len(matches) > 1:
        folders = ", ".join(sorted({m[1] for m in matches}))
        raise ValueError(f"'{name}' exists in more than one variation folder ({folders}); pass --variation-folder to disambiguate")
    return matches[0]


def extract_landmarks(blender_exe: Path, asset_path: Path, parent_landmarks_path: Path, output_path: Path) -> None:
    script = PROJECT_ROOT / "blender" / "extract_landmarks_variation.py"
    command = [
        str(blender_exe),
        "--background",
        "--factory-startup",
        "--python",
        str(script),
        "--",
        str(asset_path),
        "--parent-landmarks",
        str(parent_landmarks_path),
        "--output",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Blender extraction failed with exit code {result.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="Asset filename, e.g. auth_asian_f_0022_frame_0004_subdiv_head.glb")
    parser.add_argument("--variation-folder", default=None, help="Disambiguate which variation_* batch to look under.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Where authored-head raw landmark JSON lives.")
    parser.add_argument("--variant-raw-dir", type=Path, default=DEFAULT_VARIANT_RAW_DIR, help="Where this asset's extracted landmark JSON is cached.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the joblib model saved by tools/train_classifier.py.")
    parser.add_argument("--blender", type=Path, default=None, help="Override the Blender executable path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    parsed = parent_stem_for(args.name)
    if parsed is None:
        raise SystemExit(f"'{args.name}' doesn't match either known variant naming convention.")
    parent_stem, ethnicity, gender, head_id, frame = parsed

    parent_landmarks_path = args.raw_dir.resolve() / f"{parent_stem}.json"
    if not parent_landmarks_path.is_file():
        raise SystemExit(f"Parent landmarks not found: {parent_landmarks_path}")

    asset_path, variation_folder = find_asset(dataset_root, args.name, args.variation_folder)
    print(f"Resolved asset: {asset_path} (variation_folder={variation_folder})")

    blender_exe = args.blender.resolve() if args.blender else find_blender_executable()
    if blender_exe is None or not blender_exe.is_file():
        raise SystemExit("Could not find blender.exe. Install Blender or pass --blender.")

    output_path = args.variant_raw_dir.resolve() / f"{variation_folder}__{Path(args.name).stem}.json"
    print("Extracting landmarks via Blender ...")
    extract_landmarks(blender_exe, asset_path, parent_landmarks_path, output_path)

    feature_row = compute_feature_row(output_path, parent_landmarks_path)
    feature_row.pop("asset", None)
    feature_row.pop("parent_asset", None)

    bundle = joblib.load(args.model)
    model, feature_columns = bundle["model"], bundle["feature_columns"]

    missing = [c for c in feature_columns if c not in feature_row]
    x = pd.DataFrame([{c: feature_row.get(c) for c in feature_columns}])
    proba_good = model.predict_proba(x)[0, 1]
    predicted_label = "good" if proba_good >= 0.5 else "bad"

    print(f"\nasset:           {args.name}")
    print(f"ethnicity/gender: {ethnicity}/{gender}  head_id={head_id}  frame={frame}")
    if missing:
        print(f"note: {len(missing)} feature(s) unavailable for this asset, filled as NaN: {missing}")
    print(f"\npredicted_label: {predicted_label}")
    print(f"proba_good:      {proba_good:.3f}")
    print("\ndominant-feature values (these are ~all the model actually relies on):")
    for feat in DOMINANT_FEATURES:
        print(f"  {feat:<20} {feature_row.get(feat, 'N/A')}")
    print(
        "\nCaveat: cross-validated grouped-by-head_id evaluation put this model at only "
        "~0.55 balanced accuracy / 0.65 ROC-AUC -- treat this prediction as a rough signal, "
        "not a reliable verdict."
    )


if __name__ == "__main__":
    main()
