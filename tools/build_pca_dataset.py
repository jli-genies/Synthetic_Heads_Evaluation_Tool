#!/usr/bin/env python3
"""Extract dist_/ratio_ features for PCA-generated heads (no joints, no per-identity parent).

Unlike tools/build_training_dataset.py's HeadGen chaos-joint variants, these
assets aren't produced by deforming one specific authored parent through a
per-bind transform -- PCA samples a point in a shape basis trained across the
gen13 base head + 600 authored variants, so there's no bind/chaos data at all,
and no single authored parent identity to look up for these synthetic
head_ids (e.g. 9101-9500 don't exist in landmarks_raw/).

Landmark *extraction* still needs some parent to seed the nearest-surface
snap (blender/extract_landmarks_variation.py has no other way to locate
"nose_tip" on a mesh whose topology doesn't match landmark_config.json's
calibrated indices) -- but that seed only needs to be anatomically close, not
this mesh's true origin. Every asset here is seeded from the same reference,
landmarks_raw/genericGenie-0013-unified_rig.json (the PCA basis's own neutral
point and this tool's calibration reference), so seeding is unbiased across
ethnicity/gender rather than picking a different representative per group.

Because that seed is not a true parent, absdiff_/pctdiff_ (deviation from a
*specific* identity) and chaos_ (bind-transform magnitude -- there are no
binds) would be meaningless here and are not computed; only dist_/ratio_,
which come purely from this mesh's own snapped landmark positions, are kept.

Output has no `label` column -- these assets still need good/bad review
before use in landmarks/range_classifier.py.

Example:
    python tools/build_pca_dataset.py --pca-dir "C:/.../gnm_variation_001/cutted_mat_fixed" --limit 10
    python tools/build_pca_dataset.py --pca-dir "C:/.../gnm_variation_001/cutted_mat_fixed"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarks import chaos_joints as chaos_math  # noqa: E402
from landmarks import ratios as ratio_math  # noqa: E402
from tools.build_training_dataset import find_blender_executable, run_batch_extraction  # noqa: E402

DEFAULT_SEED_LANDMARKS = PROJECT_ROOT / "landmarks_raw" / "genericGenie-0013-unified_rig.json"

# The "good" PCA batch (gnm_variation_001) uses chaos_joints' existing
# frame-numbered convention (auth_<eth>_<g>_<id>_frame_<NNNN>_subdiv_...). The
# "bad" batch (gnm_variation_001_bad) drops the frame number entirely, e.g.
# auth_african_f_9601_subdiv__fbx_backup_head.glb / auth_african_f_9605_subdiv_african_head.glb
# -- the trailing "_fbx_backup"/"<ethnicity>" chunk is an export artifact, not
# meaningful metadata, so it's matched but discarded.
_NO_FRAME_PATTERN = re.compile(
    r"^auth_(?P<ethnicity>[a-z]+)_(?P<gender_letter>[mf])_(?P<head_id>\d{4})_subdiv.*_head$", re.IGNORECASE
)


def parse_pca_stem(stem: str) -> dict | None:
    """Ethnicity/gender/head_id/frame for a PCA-batch filename.

    Tries chaos_joints' existing naming conventions first, then falls back
    to the frame-less convention above -- using head_id as a stand-in frame
    value since there's no real frame concept for those assets.
    """
    parsed = chaos_math.parse_variant_stem(stem)
    if parsed is not None:
        return parsed
    match = _NO_FRAME_PATTERN.match(stem)
    if match is None:
        return None
    gender_letter = match["gender_letter"].lower()
    head_id = match["head_id"]
    ethnicity = match["ethnicity"].lower()
    return {
        "ethnicity": ethnicity,
        "gender": "male" if gender_letter == "m" else "female",
        "head_id": head_id,
        "frame": head_id,
        "auth_dir": f"auth_{ethnicity}_{gender_letter}_{head_id}",
    }


def load_pca_entries(pca_dir: Path) -> tuple[list[dict], list[str]]:
    """Parse ethnicity/gender/head_id/frame for every .glb under pca_dir."""
    entries: list[dict] = []
    skipped: list[str] = []
    for path in sorted(pca_dir.glob("*.glb")):
        parsed = parse_pca_stem(path.stem)
        if parsed is None:
            skipped.append(f"{path.name}: doesn't match known naming convention")
            continue
        entries.append({"path": path, **parsed})
    return entries, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pca-dir", type=Path, required=True, help="Folder of PCA-generated .glb heads.")
    parser.add_argument(
        "--seed-landmarks",
        type=Path,
        default=DEFAULT_SEED_LANDMARKS,
        help="Raw landmark JSON used to seed every extraction (not a true parent -- see module docstring).",
    )
    parser.add_argument(
        "--source",
        default="pca_gnm_variation_001",
        help="Tag written to the 'source' column, so this batch can be sliced/compared against other sources later.",
    )
    parser.add_argument(
        "--variant-raw-dir",
        type=Path,
        default=PROJECT_ROOT / "landmarks_raw_variants",
        help="Where per-asset extracted landmark JSON is written.",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "landmarks" / "pca_dataset.csv", help="Output CSV path.")
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "landmarks" / "_pca_manifest.json", help="Scratch manifest path."
    )
    parser.add_argument("--blender", type=Path, default=None, help="Override the Blender executable path.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N assets (for smoke testing).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pca_dir = args.pca_dir.resolve()
    seed_path = args.seed_landmarks.resolve()
    if not seed_path.is_file():
        raise SystemExit(f"Seed landmarks not found: {seed_path}")

    entries, skipped = load_pca_entries(pca_dir)
    if args.limit:
        entries = entries[: args.limit]
    print(f"Found {len(entries)} PCA asset(s) under {pca_dir}.")
    if skipped:
        print(f"Skipped {len(skipped)} entr{'y' if len(skipped) == 1 else 'ies'} before extraction:")
        for reason in skipped:
            print(f"  - {reason}")
    if not entries:
        print("\nNothing to extract.")
        return

    variant_raw_dir = args.variant_raw_dir.resolve()
    manifest_items = []
    row_meta: dict[str, dict] = {}
    for entry in entries:
        output_stem = f"{args.source}__{entry['path'].stem}"
        output_path = variant_raw_dir / f"{output_stem}.json"
        manifest_items.append(
            {"asset": str(entry["path"]), "parent_landmarks": str(seed_path), "output": str(output_path)}
        )
        row_meta[output_stem] = {
            "source": args.source,
            "ethnicity": entry["ethnicity"],
            "gender": entry["gender"],
            "head_id": entry["head_id"],
            "frame": entry["frame"],
        }

    blender_exe = args.blender.resolve() if args.blender else find_blender_executable()
    if blender_exe is None or not blender_exe.is_file():
        print("ERROR: could not find blender.exe. Install Blender or pass --blender.", file=sys.stderr)
        sys.exit(2)

    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_items, indent=2), encoding="utf-8")
    print(f"\nExtracting {len(manifest_items)} asset(s), all seeded from {seed_path.name} ...")
    run_batch_extraction(blender_exe, manifest_path)

    print("\nComputing dist_/ratio_ features (no absdiff_/pctdiff_/chaos_ -- see module docstring) ...")
    rows: list[dict] = []
    extraction_failed: list[str] = []
    for output_stem, meta in row_meta.items():
        output_path = variant_raw_dir / f"{output_stem}.json"
        if not output_path.is_file():
            extraction_failed.append(output_stem)
            continue
        candidate_data = json.loads(output_path.read_text(encoding="utf-8"))
        distances = ratio_math.compute_distances(candidate_data["landmarks"])
        ratios = ratio_math.compute_ratios(distances)
        row = {"asset": Path(candidate_data["asset"]).name, **meta}
        row.update({f"dist_{k}": v for k, v in distances.items()})
        row.update({f"ratio_{k}": v for k, v in ratios.items()})
        rows.append(row)

    if extraction_failed:
        print(f"\n{len(extraction_failed)} asset(s) failed extraction (see Blender output above):")
        for stem in extraction_failed:
            print(f"  - {stem}")

    if not rows:
        print("\nNo rows produced.")
        return

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} row(s) -> {output_path}")
    print("Note: no 'label' column yet -- these still need good/bad review before use in landmarks/range_classifier.py.")


if __name__ == "__main__":
    main()
