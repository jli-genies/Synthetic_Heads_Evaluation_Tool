#!/usr/bin/env python3
"""Measure one generated variation and compare it against its authored parent.

This is the "manually build up the bad set" tool: run it against a candidate
head, read the per-measurement deviation report, and decide good/bad yourself.
Pass --label to record that decision into a growing labeled dataset.

Example:
    python tools/compare_landmark_ratios.py \\
        "C:/.../dataset_5.1/.../african_male_auth_african_m_0001_frame_0004_subdiv_head_var_0.glb" \\
        --label bad --append landmarks/variations_dataset.csv

The parent authored head is derived from the variation's filename by default
(matches the "<ethnicity>_<gender>_auth_..._<head_id>_frame_..._subdiv_head_var_..."
convention seen in lists/*.json); pass --parent-landmarks to override.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarks import chaos_joints as chaos_math  # noqa: E402
from landmarks import ratios as ratio_math  # noqa: E402

VARIATION_NAME_PATTERN = re.compile(
    r"^(?P<ethnicity>[a-z]+)_(?P<gender>male|female)_auth_.+?_(?P<head_id>\d{4})_frame_\d+_subdiv_head_var_\d+$",
    re.IGNORECASE,
)

OUT_OF_RANGE_PCT = 15.0


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


def resolve_parent_landmarks_path(variation_path: Path, raw_dir: Path) -> Path:
    match = VARIATION_NAME_PATTERN.match(variation_path.stem)
    if not match:
        raise ValueError(
            f"'{variation_path.stem}' doesn't match the expected variation naming convention; "
            "pass --parent-landmarks explicitly."
        )
    parent_stem = f"{match['ethnicity'].lower()}_{match['gender'].lower()}_loPoly_{match['head_id']}_grp"
    parent_path = raw_dir / f"{parent_stem}.json"
    if not parent_path.is_file():
        raise FileNotFoundError(
            f"Expected parent landmarks at {parent_path} (derived parent: {parent_stem}). "
            "Run tools/extract_landmark_dataset.py first, or pass --parent-landmarks."
        )
    return parent_path


def run_variation_extraction(
    blender_exe: Path, asset: Path, parent_landmarks: Path, output_path: Path, mesh_name_hint: str
) -> None:
    script = PROJECT_ROOT / "blender" / "extract_landmarks_variation.py"
    command = [
        str(blender_exe),
        "--background",
        "--factory-startup",
        "--python",
        str(script),
        "--",
        str(asset),
        "--parent-landmarks",
        str(parent_landmarks),
        "--output",
        str(output_path),
        "--mesh-name-hint",
        mesh_name_hint,
    ]
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Blender extraction failed with exit code {result.returncode}")


def print_report(report: dict[str, dict[str, float]]) -> None:
    ordered = sorted(report.items(), key=lambda item: abs(item[1]["pct_diff"]), reverse=True)
    print(f"\n{'measurement':<22}{'parent':>10}{'candidate':>12}{'diff':>10}{'pct':>9}   flag")
    for name, values in ordered:
        flag = "OUT OF RANGE" if abs(values["pct_diff"]) > OUT_OF_RANGE_PCT else ""
        print(
            f"{name:<22}{values['reference']:>10.4f}{values['candidate']:>12.4f}"
            f"{values['abs_diff']:>+10.4f}{values['pct_diff']:>+8.1f}%   {flag}"
        )


def print_chaos_report(magnitudes: dict[str, float]) -> None:
    ordered = sorted(magnitudes.items(), key=lambda item: item[1], reverse=True)
    print("\nchaos-joint deviation magnitude (relative to this variant's own generation limits):")
    for name, magnitude in ordered:
        print(f"  {name:<20} {magnitude:.3f}")


def append_label(
    path: Path,
    asset: Path,
    parent_asset: str,
    label: str,
    distances: dict[str, float],
    ratios: dict[str, float],
    chaos_magnitudes: dict[str, float] | None,
) -> None:
    row: dict[str, Any] = {"asset": asset.name, "parent_asset": Path(parent_asset).name, "label": label}
    row.update({f"dist_{name}": value for name, value in distances.items()})
    row.update({f"ratio_{name}": value for name, value in ratios.items()})
    if chaos_magnitudes:
        row.update({f"chaos_{name}": value for name, value in chaos_magnitudes.items()})

    file_exists = path.is_file() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended labeled row ({label}) -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("asset", type=Path, help="Variation asset to measure (.glb / .fbx).")
    parser.add_argument(
        "--parent-landmarks", type=Path, default=None, help="Explicit path to the parent's raw landmark JSON."
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=PROJECT_ROOT / "landmarks_raw", help="Where authored-head raw landmark JSON lives."
    )
    parser.add_argument("--mesh-name-hint", default="head", help="Head-mesh name substring hint for the variation asset.")
    parser.add_argument("--blender", type=Path, default=None, help="Override the Blender executable path.")
    parser.add_argument("--label", choices=["good", "bad"], default=None, help="Record your good/bad decision.")
    parser.add_argument(
        "--append", type=Path, default=PROJECT_ROOT / "landmarks" / "variations_dataset.csv", help="CSV to append the labeled row to."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset = args.asset.resolve()

    parent_landmarks_path = (
        args.parent_landmarks.resolve()
        if args.parent_landmarks
        else resolve_parent_landmarks_path(asset, args.raw_dir.resolve())
    )
    parent_data = json.loads(parent_landmarks_path.read_text(encoding="utf-8"))

    blender_exe = args.blender.resolve() if args.blender else find_blender_executable()
    if blender_exe is None or not blender_exe.is_file():
        print("ERROR: could not find blender.exe. Install Blender or pass --blender.", file=sys.stderr)
        sys.exit(2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        candidate_output = Path(tmp_dir) / f"{asset.stem}.json"
        run_variation_extraction(blender_exe, asset, parent_landmarks_path, candidate_output, args.mesh_name_hint)
        candidate_data = json.loads(candidate_output.read_text(encoding="utf-8"))

    parent_distances = ratio_math.compute_distances(parent_data["landmarks"])
    candidate_distances = ratio_math.compute_distances(candidate_data["landmarks"])
    report = ratio_math.compare_to_reference(candidate_distances, parent_distances)

    print(f"\nCandidate: {asset.name}")
    print(f"Parent:    {Path(parent_data['asset']).name}")
    print_report(report)

    chaos_magnitudes: dict[str, float] | None = None
    try:
        chaos_magnitudes = chaos_math.compute_bind_magnitudes(asset)
        print_chaos_report(chaos_magnitudes)
    except (ValueError, FileNotFoundError) as error:
        print(f"\nNote: couldn't load chaos-joint values for this asset ({error}).")

    if args.label:
        candidate_ratios = ratio_math.compute_ratios(candidate_distances)
        append_label(
            args.append.resolve(),
            asset,
            parent_data["asset"],
            args.label,
            candidate_distances,
            candidate_ratios,
            chaos_magnitudes,
        )


if __name__ == "__main__":
    main()
