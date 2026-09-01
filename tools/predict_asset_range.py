#!/usr/bin/env python3
"""Score a single, unlabeled asset with the trained range-anomaly model.

Same asset resolution and feature computation as tools/predict_asset.py
(shared, not reimplemented, so the two paths can't drift apart) -- the only
difference is which model does the scoring: landmarks/model_range.joblib
(see landmarks/range_classifier.py) instead of landmarks/model_hist_gb.joblib.

Two calling conventions:
  - CLI use: pass just ``name`` (+ optionally --variation-folder) and this
    searches --dataset-root for the file, exactly like tools/predict_asset.py.
  - Caller-already-knows-the-path use (e.g. the PyQt GUI, which has the exact
    file open): pass --asset-path to skip the dataset-root search entirely.

--json-out writes a machine-readable result, including a ``joint_features``
map (one good/bad verdict per chaos_* joint, keyed by the same marker names
ui/tag_panel.py's CHAOS_JOINT_MARKERS uses) so a caller can pre-fill that
tab's per-joint controls directly.

Example:
    python tools/predict_asset_range.py auth_asian_f_0022_frame_0004_subdiv_head.glb --variation-folder variation_long_faces_002
    python tools/predict_asset_range.py auth_asian_f_0022_frame_0004_subdiv_head.glb --asset-path "C:/.../auth_asian_f_0022_frame_0004_subdiv_head.glb" --json-out result.json
"""

from __future__ import annotations

import argparse
import json
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
)
from tools.predict_asset import extract_landmarks, find_asset  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "landmarks" / "model_range.joblib"
DEFAULT_RAW_DIR = PROJECT_ROOT / "landmarks_raw"
DEFAULT_VARIANT_RAW_DIR = PROJECT_ROOT / "landmarks_raw_variants"
DEFAULT_JOINT_SIGMA = 2.0
CHAOS_PREFIX = "chaos_"


def joint_features_from_z(z_row: pd.Series, sigma: float) -> dict[str, str]:
    """{marker: 'good'|'bad'} for every chaos_* feature, keyed without the prefix
    (matching ui/tag_panel.py's CHAOS_JOINT_MARKERS naming)."""
    features: dict[str, str] = {}
    for feature_name, z in z_row.items():
        if not str(feature_name).startswith(CHAOS_PREFIX):
            continue
        marker = str(feature_name)[len(CHAOS_PREFIX):]
        features[marker] = "bad" if pd.notna(z) and z > sigma else "good"
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="Asset filename, e.g. auth_asian_f_0022_frame_0004_subdiv_head.glb")
    parser.add_argument("--variation-folder", default=None, help="Disambiguate which variation_* batch to look under.")
    parser.add_argument("--asset-path", type=Path, default=None, help="Exact file path, skipping the --dataset-root search.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Where authored-head raw landmark JSON lives.")
    parser.add_argument("--variant-raw-dir", type=Path, default=DEFAULT_VARIANT_RAW_DIR, help="Where this asset's extracted landmark JSON is cached.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the joblib model saved by tools/train_range_classifier.py.")
    parser.add_argument("--blender", type=Path, default=None, help="Override the Blender executable path.")
    parser.add_argument("--joint-sigma", type=float, default=DEFAULT_JOINT_SIGMA, help="Per-joint good/bad cutoff, in standard deviations from that joint's good median.")
    parser.add_argument("--json-out", type=Path, default=None, help="Also write a machine-readable result (with joint_features) to this path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parsed = parent_stem_for(args.name)
    if parsed is None:
        raise SystemExit(f"'{args.name}' doesn't match either known variant naming convention.")
    parent_stem, ethnicity, gender, head_id, frame = parsed

    parent_landmarks_path = args.raw_dir.resolve() / f"{parent_stem}.json"
    if not parent_landmarks_path.is_file():
        raise SystemExit(f"Parent landmarks not found: {parent_landmarks_path}")

    if args.asset_path is not None:
        asset_path = args.asset_path.resolve()
        if not asset_path.is_file():
            raise SystemExit(f"--asset-path does not exist: {asset_path}")
        variation_folder = args.variation_folder
    else:
        dataset_root = args.dataset_root.resolve()
        asset_path, variation_folder = find_asset(dataset_root, args.name, args.variation_folder)
    print(f"Resolved asset: {asset_path} (variation_folder={variation_folder})")

    blender_exe = args.blender.resolve() if args.blender else find_blender_executable()
    if blender_exe is None or not blender_exe.is_file():
        raise SystemExit("Could not find blender.exe. Install Blender or pass --blender.")

    output_stem = f"{variation_folder or 'adhoc'}__{Path(args.name).stem}"
    output_path = args.variant_raw_dir.resolve() / f"{output_stem}.json"
    print("Extracting landmarks via Blender ...")
    extract_landmarks(blender_exe, asset_path, parent_landmarks_path, output_path)

    feature_row = compute_feature_row(output_path, parent_landmarks_path)
    feature_row.pop("asset", None)
    feature_row.pop("parent_asset", None)
    # Present but unused unless the loaded model was trained with --group-cols
    # (tools/train_range_classifier.py) -- then feature_columns includes these
    # names and the model conditions its good-range on them.
    feature_row["ethnicity"] = ethnicity
    feature_row["gender"] = gender

    bundle = joblib.load(args.model)
    model, feature_columns = bundle["model"], bundle["feature_columns"]

    missing = [c for c in feature_columns if c not in feature_row]
    x = pd.DataFrame([{c: feature_row.get(c) for c in feature_columns}])
    proba_good = model.predict_proba(x)[0, 1]
    predicted_label = "good" if model.predict(x)[0] == 1 else "bad"
    z_row = model.per_feature_z(x).iloc[0]
    top_contributor = model.top_contributors(x).iloc[0]
    joint_features = joint_features_from_z(z_row, args.joint_sigma)

    print(f"\nasset:            {args.name}")
    print(f"ethnicity/gender: {ethnicity}/{gender}  head_id={head_id}  frame={frame}")
    if missing:
        print(f"note: {len(missing)} feature(s) unavailable for this asset, filled as NaN: {missing}")
    print(f"\npredicted_label:  {predicted_label}")
    print(f"proba_good:       {proba_good:.3f}")
    print(f"top_contributor:  {top_contributor}  (the joint furthest from its own good range, in std-devs)")

    ranges = model.range_table()
    measurement_cols = [c for c in feature_columns if not getattr(model, "group_cols", None) or c not in model.group_cols]
    if getattr(model, "group_cols", None):
        group_key = "|".join(str(feature_row.get(c)) for c in model.group_cols)
        resolved_group = group_key if group_key in ranges.index.get_level_values("group") else "(pooled fallback)"
        print(
            f"\nGood range conditioned on {'/'.join(model.group_cols)}={group_key} "
            f"({'own range' if resolved_group == group_key else 'too few good examples -- using pooled fallback'}):"
        )
        ranges = ranges.loc[resolved_group]

    print("\nPer-joint good ranges vs. this asset's values:")
    for feat in measurement_cols:
        value = feature_row.get(feat)
        lo, hi = ranges.loc[feat, "lower_2sigma"], ranges.loc[feat, "upper_2sigma"]
        flag = " <-- outside 2-sigma range" if value is not None and not (lo <= value <= hi) else ""
        print(f"  {feat:<26} value={value if value is not None else 'N/A':<8} good_range=[{lo:.3f}, {hi:.3f}]{flag}")
    print(
        "\nCaveat: cross-validated grouped-by-head_id evaluation put this model at only "
        "~0.65 balanced accuracy / 0.69 ROC-AUC -- treat this prediction as a rough signal, "
        "not a reliable verdict."
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "asset": args.name,
                    "predicted_label": predicted_label,
                    "proba_good": float(proba_good),
                    "top_contributor": str(top_contributor),
                    "joint_features": joint_features,
                    "missing_features": missing,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote machine-readable result to {args.json_out}")


if __name__ == "__main__":
    main()
