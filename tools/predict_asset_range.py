#!/usr/bin/env python3
"""Score a single, unlabeled asset with the trained range-anomaly model.

Same asset resolution as tools/predict_asset.py (shared, not reimplemented,
so the two paths can't drift apart), except identity parsing here also
accepts assets with no true per-identity parent (e.g. PCA-generated heads --
see tools/build_pca_dataset.py): if a real authored parent exists under
--raw-dir, features are the full dist_/ratio_/absdiff_/pctdiff_/chaos_ row
(landmarks/feature_row.py); otherwise landmark extraction is seeded from
landmarks_raw/genericGenie-0013-unified_rig.json (a seed, not a true parent)
and only dist_/ratio_ are computed -- absdiff_/pctdiff_/chaos_ would compare
against an identity this asset doesn't actually have.

Two calling conventions:
  - CLI use: pass just ``name`` (+ optionally --variation-folder) and this
    searches --dataset-root for the file, exactly like tools/predict_asset.py.
  - Caller-already-knows-the-path use (e.g. the PyQt GUI, which has the exact
    file open): pass --asset-path to skip the dataset-root search entirely.

--json-out writes a machine-readable result, including a ``region_features``
map (one good/bad diagnostic flag per tag_schema.json face-region category,
keyed by category id) so a caller can pre-fill the Face Proportions tab's
per-region controls directly. That flag is a fixed 2-sigma convention per
region and is diagnostic only -- predicted_label/proba_good (the model's
single RMS-combined score against its tuned threshold) is the actual verdict,
never a tally of flagged regions.

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

from landmarks import ratios as ratio_math  # noqa: E402
from landmarks.feature_row import compute_feature_row  # noqa: E402
from tools.build_pca_dataset import DEFAULT_SEED_LANDMARKS, parse_pca_stem  # noqa: E402
from tools.build_training_dataset import DEFAULT_DATASET_ROOT, find_blender_executable  # noqa: E402
from tools.predict_asset import extract_landmarks, find_asset  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "landmarks" / "model_range_ratio_grouped.joblib"
DEFAULT_RAW_DIR = PROJECT_ROOT / "landmarks_raw"
DEFAULT_VARIANT_RAW_DIR = PROJECT_ROOT / "landmarks_raw_variants"
DEFAULT_REGION_SIGMA = 2.0


def resolve_identity_and_seed(name: str, raw_dir: Path) -> tuple[str, str, str, str, Path, bool]:
    """(ethnicity, gender, head_id, frame, landmark_seed_path, seed_is_true_parent).

    Tries every known naming convention (tools/build_pca_dataset.parse_pca_stem
    is a superset of tools/build_training_dataset.parent_stem_for's parsing).
    If a real authored parent exists for this identity under raw_dir, that's
    the seed and it IS this asset's true parent (absdiff_/pctdiff_/chaos_ are
    meaningful). Otherwise falls back to the gen13 seed used for PCA batches --
    a seed only, not this asset's identity.
    """
    parsed = parse_pca_stem(Path(name).stem)
    if parsed is None:
        raise SystemExit(f"'{name}' doesn't match any known naming convention.")
    ethnicity, gender, head_id, frame = parsed["ethnicity"], parsed["gender"], parsed["head_id"], parsed["frame"]
    true_parent_path = raw_dir / f"{ethnicity}_{gender}_loPoly_{head_id}_grp.json"
    if true_parent_path.is_file():
        return ethnicity, gender, head_id, frame, true_parent_path, True
    return ethnicity, gender, head_id, frame, DEFAULT_SEED_LANDMARKS, False


def compute_features_no_parent(landmarks_path: Path) -> dict:
    """dist_/ratio_ only, computed purely from this asset's own landmarks --
    for use when the extraction seed isn't a true parent (see module docstring)."""
    candidate_data = json.loads(landmarks_path.read_text(encoding="utf-8"))
    distances = ratio_math.compute_distances(candidate_data["landmarks"])
    ratios = ratio_math.compute_ratios(distances)
    row: dict = {}
    row.update({f"dist_{k}": v for k, v in distances.items()})
    row.update({f"ratio_{k}": v for k, v in ratios.items()})
    return row


def region_features_from_scores(scores: pd.Series, sigma: float) -> dict[str, str]:
    """{region_id: 'good'|'bad'} -- diagnostic only, see module docstring."""
    return {str(region): ("bad" if pd.notna(score) and score > sigma else "good") for region, score in scores.items()}


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
    parser.add_argument("--region-sigma", type=float, default=DEFAULT_REGION_SIGMA, help="Per-region diagnostic cutoff, in standard deviations from that region's good median (see module docstring -- not the model's own tuned verdict threshold).")
    parser.add_argument("--json-out", type=Path, default=None, help="Also write a machine-readable result (with region_features) to this path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()

    ethnicity, gender, head_id, frame, parent_landmarks_path, has_true_parent = resolve_identity_and_seed(
        args.name, raw_dir
    )
    if not parent_landmarks_path.is_file():
        raise SystemExit(f"Landmarks seed not found: {parent_landmarks_path}")

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
    print(f"Extracting landmarks via Blender (seeded from {parent_landmarks_path.name}) ...")
    extract_landmarks(blender_exe, asset_path, parent_landmarks_path, output_path)

    if has_true_parent:
        feature_row = compute_feature_row(output_path, parent_landmarks_path)
        feature_row.pop("asset", None)
        feature_row.pop("parent_asset", None)
    else:
        feature_row = compute_features_no_parent(output_path)
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
    top_contributor = model.top_contributors(x).iloc[0]

    regions = ratio_math.regions_to_features()
    region_scores = model.region_scores(x, regions).iloc[0]
    region_top_contributor = model.region_top_contributors(x, regions).iloc[0]
    region_features = region_features_from_scores(region_scores, args.region_sigma)

    print(f"\nasset:            {args.name}")
    print(f"ethnicity/gender: {ethnicity}/{gender}  head_id={head_id}  frame={frame}")
    print(f"identity source:  {'true authored parent' if has_true_parent else 'gen13 seed only (no true parent for this asset)'}")
    if missing:
        print(f"note: {len(missing)} feature(s) unavailable for this asset, filled as NaN: {missing}")
    print(f"\npredicted_label:  {predicted_label}  (model's single RMS-combined score vs. its tuned threshold -- the actual verdict)")
    print(f"proba_good:       {proba_good:.3f}")
    print(f"top_contributor:  {top_contributor}  (single measurement furthest from its own good range, in std-devs)")

    print(f"\nPer-region deviation (diagnostic only -- fixed {args.region_sigma:g}-sigma cutoff, not the verdict threshold above):")
    for region, score in region_scores.items():
        flag = " <-- outside cutoff" if region_features[region] == "bad" else ""
        contributor = region_top_contributor.get(region, "n/a")
        print(f"  {region:<14} z={score:.2f}  top={contributor}{flag}")

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

    print("\nPer-measurement good ranges vs. this asset's values:")
    for feat in measurement_cols:
        value = feature_row.get(feat)
        lo, hi = ranges.loc[feat, "lower_2sigma"], ranges.loc[feat, "upper_2sigma"]
        flag = " <-- outside 2-sigma range" if value is not None and not (lo <= value <= hi) else ""
        print(f"  {feat:<26} value={value if value is not None else 'N/A':<8} good_range=[{lo:.3f}, {hi:.3f}]{flag}")
    print(
        "\nCaveat: cross-validated evaluation on the PCA good/bad batch put this model at "
        "~0.76 balanced accuracy / 0.83 ROC-AUC -- treat this prediction as a strong signal, "
        "not a substitute for review."
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
                    "region_features": region_features,
                    "region_scores": {k: float(v) for k, v in region_scores.items()},
                    "region_top_contributor": {k: str(v) for k, v in region_top_contributor.items()},
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
