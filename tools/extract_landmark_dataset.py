#!/usr/bin/env python3
"""Build the authored-head landmark/ratio baseline dataset.

Walks a folder of authored heads (fixed topology), runs
blender/extract_landmarks.py once in batch mode across all of them, then
computes distances/ratios for each and writes a combined dataset.csv -- the
per-authored-head reference table tools/compare_landmark_ratios.py reads from.

Example:
    python tools/extract_landmark_dataset.py \\
        --assets-root "C:/Users/auror/Documents/Gen3d_testing/authored_heads_var_v2"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarks import ratios as ratio_math  # noqa: E402

ASSET_EXTENSIONS = {".fbx", ".glb"}
# Matches e.g. "african_female_loPoly_0001_grp.fbx"; anything else (stray
# assets) is skipped rather than guessed at, except GENERIC_REFERENCE_STEM.
AUTHORED_NAME_PATTERN = re.compile(
    r"^(?P<ethnicity>[a-z]+)_(?P<gender>male|female)_loPoly_(?P<head_id>\d{4})_grp$",
    re.IGNORECASE,
)

# The master rigged template every authored head is derived from. Same head
# topology as the authored set (verified: identical vertex/polygon hash), so
# it runs through the same extraction pipeline and is included as an explicit
# top-of-hierarchy reference row rather than skipped as a stray file.
GENERIC_REFERENCE_STEM = "genericGenie-0013-unified_rig"


def find_blender_executable() -> Path | None:
    """Locate a Blender binary (PATH first, then common Windows installs).

    Duplicated from ui/asset_tree.py rather than imported, so this CLI stays
    free of a PyQt6 import.
    """
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)
    program_files = Path(r"C:\Program Files\Blender Foundation")
    if program_files.is_dir():
        installs = sorted(program_files.glob("Blender */blender.exe"), reverse=True)
        if installs:
            return installs[0]
    return None


def collect_authored_assets(root: Path) -> list[Path]:
    """Sorted .fbx/.glb files under root matching the authored-head naming convention,
    plus GENERIC_REFERENCE_STEM if present.

    Skips packed tag-panel copies (<stem>/<stem>.fbx next to the original) and
    anything else not matching AUTHORED_NAME_PATTERN.
    """
    assets: list[Path] = []
    skipped: list[str] = []
    for path in root.rglob("*"):
        if not (path.is_file() and path.suffix.lower() in ASSET_EXTENSIONS):
            continue
        sibling_original = path.parent.parent / path.name
        if path.parent.name == path.stem and sibling_original.is_file():
            continue
        if path.stem == GENERIC_REFERENCE_STEM:
            assets.append(path)
            continue
        if not AUTHORED_NAME_PATTERN.match(path.stem):
            skipped.append(path.name)
            continue
        assets.append(path)
    if skipped:
        print(f"Skipped {len(skipped)} non-authored-pattern file(s): {skipped}")
    return sorted(assets)


def parse_authored_name(stem: str) -> tuple[str, str, str]:
    if stem == GENERIC_REFERENCE_STEM:
        return "generic", "", "0013"
    match = AUTHORED_NAME_PATTERN.match(stem)
    if not match:
        raise ValueError(f"'{stem}' does not match the authored-head naming convention")
    return match["ethnicity"].lower(), match["gender"].lower(), match["head_id"]


def run_blender_extraction(
    blender_exe: Path, manifest_path: Path, config_path: Path, output_dir: Path
) -> None:
    script = PROJECT_ROOT / "blender" / "extract_landmarks.py"
    command = [
        str(blender_exe),
        "--background",
        "--factory-startup",
        "--python",
        str(script),
        "--",
        "--manifest",
        str(manifest_path),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Blender extraction failed with exit code {result.returncode}")


def build_dataset_rows(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for json_path in sorted(raw_dir.glob("*.json")):
        if json_path.name.startswith("_"):
            continue  # e.g. a stray _manifest.json, not a per-asset result
        data = json.loads(json_path.read_text(encoding="utf-8"))
        asset_name = Path(data["asset"]).name
        stem = Path(data["asset"]).stem
        try:
            ethnicity, gender, head_id = parse_authored_name(stem)
        except ValueError:
            ethnicity, gender, head_id = "", "", ""

        distances = ratio_math.compute_distances(data["landmarks"])
        ratios = ratio_math.compute_ratios(distances)

        row = {
            "asset": asset_name,
            "ethnicity": ethnicity,
            "gender": gender,
            "head_id": head_id,
            "is_master_reference": stem == GENERIC_REFERENCE_STEM,
        }
        row.update({f"dist_{name}": value for name, value in distances.items()})
        row.update({f"ratio_{name}": value for name, value in ratios.items()})
        rows.append(row)
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        print("No rows to write.")
        return
    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s) -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets-root", required=True, type=Path, help="Authored heads folder to scan recursively.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "landmarks" / "landmark_config.json", help="landmark_config.json path."
    )
    parser.add_argument(
        "--raw-output-dir", type=Path, default=PROJECT_ROOT / "landmarks_raw", help="Per-asset raw landmark JSON output dir."
    )
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "landmarks" / "dataset.csv", help="Output CSV path.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "landmarks" / "_manifest.json", help="Scratch manifest path.")
    parser.add_argument("--blender", type=Path, default=None, help="Override the Blender executable path.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N assets (for smoke testing).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets_root = args.assets_root.resolve()
    if not assets_root.is_dir():
        print(f"ERROR: assets root not found: {assets_root}", file=sys.stderr)
        sys.exit(2)

    assets = collect_authored_assets(assets_root)
    if args.limit:
        assets = assets[: args.limit]
    if not assets:
        print("ERROR: no authored-head assets found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(assets)} authored head(s) to process.")

    blender_exe = args.blender.resolve() if args.blender else find_blender_executable()
    if blender_exe is None or not blender_exe.is_file():
        print("ERROR: could not find blender.exe. Install Blender or pass --blender.", file=sys.stderr)
        sys.exit(2)

    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps([str(a) for a in assets], indent=2), encoding="utf-8")

    run_blender_extraction(blender_exe, manifest_path, args.config.resolve(), args.raw_output_dir.resolve())

    rows = build_dataset_rows(args.raw_output_dir.resolve())
    write_csv(rows, args.dataset.resolve())


if __name__ == "__main__":
    main()
