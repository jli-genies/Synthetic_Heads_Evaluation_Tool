#!/usr/bin/env python3
"""Turn lists/good.json + lists/bad.json into a training-ready feature-vector CSV.

For each labeled entry ({name, variation_folder}):
  1. Derive its authored parent (ethnicity/gender/head_id/frame) from the
     filename, and load that parent's already-extracted landmarks from
     landmarks_raw/ (built by tools/extract_landmark_dataset.py).
  2. Resolve the entry's actual file on disk under
     {dataset-root}/{variation_folder}/.
  3. Batch all resolved variants through blender/extract_landmarks_variation.py
     (one Blender process, not one per asset) to get each variant's own
     landmark positions via nearest-surface projection onto its parent's
     landmarks.
  4. Compute distances/ratios (landmarks/ratios.py), this variant's deviation
     from its own parent's distances (compare_to_reference -- the per-identity
     comparison, not a pooled population stat), and chaos-joint deviation
     magnitudes (landmarks/chaos_joints.py) where resolvable.
  5. Write one row per entry to landmarks/training_dataset.csv, with `label`
     (good/bad) plus ethnicity/gender/head_id/variation_folder as metadata
     columns -- the model should train on the dist_*/ratio_*/absdiff_*/
     pctdiff_*/chaos_* columns, not the metadata, since deviation-from-own-
     parent is already demographic-normalized by construction.

Example:
    python tools/build_training_dataset.py --limit 10   # smoke test
    python tools/build_training_dataset.py               # full run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarks import chaos_joints as chaos_math  # noqa: E402
from landmarks.feature_row import compute_feature_row  # noqa: E402
from ui import sort_assets as sa  # noqa: E402

DEFAULT_DATASET_ROOT = Path("C:/Users/auror/Documents/Gen3d_testing/synthetic_heads/dataset_5.1")


def find_blender_executable() -> Path | None:
    """Duplicated from ui/asset_tree.py so this CLI stays free of a PyQt6 import."""
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)
    program_files = Path(r"C:\Program Files\Blender Foundation")
    if program_files.is_dir():
        installs = sorted(program_files.glob("Blender */blender.exe"), reverse=True)
        if installs:
            return installs[0]
    return None


def load_labeled_entries() -> list[dict]:
    """[{name, variation_folder, label}, ...] from lists/good.json + lists/bad.json."""
    entries: list[dict] = []
    for bucket, _filename in sa.BUCKET_FILES:
        path = sa._bucket_path(PROJECT_ROOT, bucket)
        for entry in sa._load_entries(path):
            entries.append({"name": entry["name"], "variation_folder": entry.get("variation_folder"), "label": bucket})
    return entries


def resolve_variant_path(dataset_root: Path, variation_folder: str, name: str) -> Path | None:
    matches = list((dataset_root / variation_folder).rglob(name))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  note: {len(matches)} copies of '{name}' under {variation_folder}; using {matches[0]}")
    return matches[0]


def parent_stem_for(name: str) -> tuple[str, str, str, str, str] | None:
    """Returns (parent_stem, ethnicity, gender, head_id, frame), or None if the
    name doesn't match either known variant naming convention."""
    parsed = chaos_math.parse_variant_stem(Path(name).stem)
    if parsed is None:
        return None
    parent_stem = f"{parsed['ethnicity']}_{parsed['gender']}_loPoly_{parsed['head_id']}_grp"
    return parent_stem, parsed["ethnicity"], parsed["gender"], parsed["head_id"], parsed["frame"]


def run_batch_extraction(blender_exe: Path, manifest_path: Path) -> None:
    script = PROJECT_ROOT / "blender" / "extract_landmarks_variation.py"
    command = [
        str(blender_exe),
        "--background",
        "--factory-startup",
        "--python",
        str(script),
        "--",
        "--manifest",
        str(manifest_path),
    ]
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout[-8000:])  # batches can be long; keep the tail
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Blender batch extraction failed with exit code {result.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Root folder containing variation_* batches.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "landmarks_raw", help="Where authored-head raw landmark JSON lives.")
    parser.add_argument(
        "--variant-raw-dir",
        type=Path,
        default=PROJECT_ROOT / "landmarks_raw_variants",
        help="Where per-variant raw landmark JSON is written.",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "landmarks" / "training_dataset.csv", help="Output CSV path.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "landmarks" / "_variant_manifest.json", help="Scratch manifest path.")
    parser.add_argument("--blender", type=Path, default=None, help="Override the Blender executable path.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N labeled entries (for smoke testing).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    raw_dir = args.raw_dir.resolve()
    variant_raw_dir = args.variant_raw_dir.resolve()

    entries = load_labeled_entries()
    if args.limit:
        entries = entries[: args.limit]
    print(f"Loaded {len(entries)} labeled entries from lists/good.json + lists/bad.json.")

    manifest_items = []
    row_meta: dict[str, dict] = {}  # output stem -> metadata needed to build the CSV row
    skipped: list[str] = []

    for entry in entries:
        name = entry["name"]
        variation_folder = entry["variation_folder"]
        parsed = parent_stem_for(name)
        if parsed is None:
            skipped.append(f"{name}: doesn't match variant naming convention")
            continue
        parent_stem, ethnicity, gender, head_id, frame = parsed

        parent_landmarks_path = raw_dir / f"{parent_stem}.json"
        if not parent_landmarks_path.is_file():
            skipped.append(f"{name}: parent landmarks not found ({parent_landmarks_path.name})")
            continue

        asset_path = resolve_variant_path(dataset_root, variation_folder, name)
        if asset_path is None:
            skipped.append(f"{name}: file not found under {variation_folder}")
            continue

        output_stem = f"{variation_folder}__{Path(name).stem}"
        output_path = variant_raw_dir / f"{output_stem}.json"
        manifest_items.append(
            {"asset": str(asset_path), "parent_landmarks": str(parent_landmarks_path), "output": str(output_path)}
        )
        row_meta[output_stem] = {
            "label": entry["label"],
            "variation_folder": variation_folder,
            "ethnicity": ethnicity,
            "gender": gender,
            "head_id": head_id,
            "frame": frame,
            "parent_landmarks_path": parent_landmarks_path,
        }

    if skipped:
        print(f"\nSkipped {len(skipped)} entr{'y' if len(skipped) == 1 else 'ies'} before extraction:")
        for reason in skipped:
            print(f"  - {reason}")

    if not manifest_items:
        print("\nNothing to extract.")
        return

    blender_exe = args.blender.resolve() if args.blender else find_blender_executable()
    if blender_exe is None or not blender_exe.is_file():
        print("ERROR: could not find blender.exe. Install Blender or pass --blender.", file=sys.stderr)
        sys.exit(2)

    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_items, indent=2), encoding="utf-8")
    print(f"\nExtracting {len(manifest_items)} variant(s) ...")
    run_batch_extraction(blender_exe, manifest_path)

    print("\nComputing features ...")
    rows: list[dict] = []
    extraction_failed: list[str] = []
    for output_stem, meta in row_meta.items():
        output_path = variant_raw_dir / f"{output_stem}.json"
        if not output_path.is_file():
            extraction_failed.append(output_stem)
            continue

        feature_row = compute_feature_row(output_path, meta["parent_landmarks_path"])
        row = {
            "asset": feature_row.pop("asset"),
            "parent_asset": feature_row.pop("parent_asset"),
            "label": meta["label"],
            "variation_folder": meta["variation_folder"],
            "ethnicity": meta["ethnicity"],
            "gender": meta["gender"],
            "head_id": meta["head_id"],
            "frame": meta["frame"],
        }
        row.update(feature_row)
        rows.append(row)

    if extraction_failed:
        print(f"\n{len(extraction_failed)} variant(s) failed extraction (see Blender output above):")
        for stem in extraction_failed:
            print(f"  - {stem}")

    if not rows:
        print("\nNo rows produced.")
        return

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    import csv

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    good_count = sum(1 for r in rows if r["label"] == "good")
    bad_count = sum(1 for r in rows if r["label"] == "bad")
    print(f"\nWrote {len(rows)} row(s) ({good_count} good, {bad_count} bad) -> {output_path}")


if __name__ == "__main__":
    main()
