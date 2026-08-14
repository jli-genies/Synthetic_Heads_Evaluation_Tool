"""Turn one variant's extracted landmarks + its authored parent's landmarks
into the dist_/ratio_/absdiff_/pctdiff_/chaos_ feature row trained on in
landmarks/training_dataset.csv.

Shared by tools/build_training_dataset.py (labeled training rows) and
tools/predict_asset.py (scoring a new, unlabeled asset) so the two paths
can't drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

from landmarks import chaos_joints as chaos_math
from landmarks import ratios as ratio_math


def compute_feature_row(candidate_landmarks_path: Path, parent_landmarks_path: Path) -> dict:
    candidate_data = json.loads(Path(candidate_landmarks_path).read_text(encoding="utf-8"))
    parent_data = json.loads(Path(parent_landmarks_path).read_text(encoding="utf-8"))

    parent_distances = ratio_math.compute_distances(parent_data["landmarks"])
    candidate_distances = ratio_math.compute_distances(candidate_data["landmarks"])
    candidate_ratios = ratio_math.compute_ratios(candidate_distances)
    deviation = ratio_math.compare_to_reference(candidate_distances, parent_distances)

    row: dict = {
        "asset": Path(candidate_data["asset"]).name,
        "parent_asset": Path(parent_data["asset"]).name,
    }
    row.update({f"dist_{k}": v for k, v in candidate_distances.items()})
    row.update({f"ratio_{k}": v for k, v in candidate_ratios.items()})
    row.update({f"absdiff_{k}": v["abs_diff"] for k, v in deviation.items()})
    row.update({f"pctdiff_{k}": v["pct_diff"] for k, v in deviation.items()})

    try:
        chaos_magnitudes = chaos_math.compute_bind_magnitudes(Path(candidate_data["asset"]))
        row.update({f"chaos_{k}": v for k, v in chaos_magnitudes.items()})
    except (ValueError, FileNotFoundError) as error:
        print(f"  note: no chaos-joint values for {row['asset']} ({error})")

    return row
