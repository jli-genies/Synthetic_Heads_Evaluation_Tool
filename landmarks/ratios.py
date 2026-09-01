"""Distance and ratio math over named facial landmark points.

Landmark points come from ``landmark_config.json`` (calibrated vertex indices
on the authored head topology, see ``blender/pick_landmarks.py``) or from
nearest-surface-point projection onto a generated variation's own mesh (see
``blender/extract_landmarks_variation.py``). Either way, by the time code
reaches this module a landmark set is just ``{name: (x, y, z), ...}`` — no
bpy dependency, so this is plain, testable Python.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

Point = Sequence[float]

# Points computed as the midpoint of two calibrated points, rather than
# requiring their own vertex pick during calibration.
DERIVED_POINTS: dict[str, tuple[str, str]] = {
    "left_eye_center": ("left_eye_outer", "left_eye_inner"),
    "right_eye_center": ("right_eye_outer", "right_eye_inner"),
    "mouth_center": ("mouth_left_corner", "mouth_right_corner"),
    "brow_center": ("left_brow_inner", "right_brow_inner"),
}

# (measurement name, point a, point b). Point names may be raw (calibrated)
# or derived (above); both are resolved into one coordinate lookup before use.
RATIO_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    ("interocular_inner", "left_eye_inner", "right_eye_inner"),
    ("interocular_outer", "left_eye_outer", "right_eye_outer"),
    ("eye_width_left", "left_eye_outer", "left_eye_inner"),
    ("eye_width_right", "right_eye_outer", "right_eye_inner"),
    ("eye_to_nose_tip_left", "left_eye_center", "nose_tip"),
    ("eye_to_nose_tip_right", "right_eye_center", "nose_tip"),
    ("eye_to_mouth_left", "left_eye_center", "mouth_left_corner"),
    ("eye_to_mouth_right", "right_eye_center", "mouth_right_corner"),
    ("nose_width", "left_nose_ala", "right_nose_ala"),
    ("nose_length", "nose_bridge", "nose_tip"),
    ("mouth_width", "mouth_left_corner", "mouth_right_corner"),
    ("lip_height", "upper_lip_top", "lower_lip_bottom"),
    ("jaw_width", "jaw_left", "jaw_right"),
    ("cheekbone_width", "left_cheekbone", "right_cheekbone"),
    ("face_height", "brow_center", "chin_bottom"),
    ("chin_to_mouth", "mouth_center", "chin_bottom"),
    ("brow_width_left", "left_brow_inner", "left_brow_outer"),
    ("brow_width_right", "right_brow_inner", "right_brow_outer"),
    ("brow_span", "left_brow_outer", "right_brow_outer"),
)

# Normalizes every other distance, so ratios reflect proportion, not overall
# head scale.
REFERENCE_MEASUREMENT = "interocular_inner"

# Maps each RATIO_DEFINITIONS measurement to the tag_schema.json face-region
# category/categories it speaks to (category ids, so callers building UI
# labels can pull them straight from tag_schema.json's own categories rather
# than a second copy). A handful of measurements are inherently inter-regional
# (e.g. eye_to_mouth_*) and appear under every region they touch, rather than
# being forced into one.
FEATURE_REGIONS: dict[str, tuple[str, ...]] = {
    "interocular_outer": ("eyes_shape",),
    "eye_width_left": ("eyes_shape",),
    "eye_width_right": ("eyes_shape",),
    "eye_to_nose_tip_left": ("eyes_shape", "nose_shape"),
    "eye_to_nose_tip_right": ("eyes_shape", "nose_shape"),
    "eye_to_mouth_left": ("eyes_shape", "mouth_shape"),
    "eye_to_mouth_right": ("eyes_shape", "mouth_shape"),
    "nose_width": ("nose_shape",),
    "nose_length": ("nose_shape",),
    "mouth_width": ("mouth_shape",),
    "lip_height": ("lips_shape",),
    "jaw_width": ("jaw_shape",),
    "cheekbone_width": ("cheeks_shape",),
    "face_height": ("head_shape",),
    "chin_to_mouth": ("chin_shape", "mouth_shape"),
    "brow_width_left": ("brow_shape",),
    "brow_width_right": ("brow_shape",),
    "brow_span": ("brow_shape",),
}


def regions_to_features(prefix: str = "ratio_") -> dict[str, list[str]]:
    """{region: [feature_column, ...]}, inverse of FEATURE_REGIONS with the
    feature-column prefix applied -- what range_classifier's region_scores()
    and any caller building a per-region breakdown needs."""
    out: dict[str, list[str]] = {}
    for measurement, regions in FEATURE_REGIONS.items():
        for region in regions:
            out.setdefault(region, []).append(f"{prefix}{measurement}")
    return out


def load_config(path: str | Path) -> dict:
    import json

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [name for name, idx in config["points"].items() if idx is None]
    if missing:
        raise ValueError(
            f"landmark_config.json is missing vertex indices for: {', '.join(missing)}. "
            "Run blender/pick_landmarks.py to calibrate them first."
        )
    if config.get("topology_hash") is None:
        raise ValueError(
            "landmark_config.json has no topology_hash yet. "
            "Run blender/pick_landmarks.py against a reference head first."
        )
    return config


def _resolve_points(landmarks: dict[str, Point]) -> dict[str, tuple[float, float, float]]:
    resolved: dict[str, tuple[float, float, float]] = {
        name: (float(p[0]), float(p[1]), float(p[2])) for name, p in landmarks.items()
    }
    for derived_name, (a, b) in DERIVED_POINTS.items():
        if a in resolved and b in resolved:
            pa, pb = resolved[a], resolved[b]
            resolved[derived_name] = tuple((pa[i] + pb[i]) / 2 for i in range(3))
    return resolved


def compute_distances(landmarks: dict[str, Point]) -> dict[str, float]:
    """Raw Euclidean distance for every entry in RATIO_DEFINITIONS.

    ``landmarks`` must contain every raw point name used below (derived
    midpoints are computed automatically). Raises KeyError naming the first
    missing point if the landmark set is incomplete.
    """
    points = _resolve_points(landmarks)
    distances: dict[str, float] = {}
    for name, point_a, point_b in RATIO_DEFINITIONS:
        for point_name in (point_a, point_b):
            if point_name not in points:
                raise KeyError(f"landmark set is missing point '{point_name}' (needed for '{name}')")
        distances[name] = math.dist(points[point_a], points[point_b])
    return distances


def compute_ratios(
    distances: dict[str, float], reference: str = REFERENCE_MEASUREMENT
) -> dict[str, float]:
    """Every distance divided by ``reference``, so results are scale-invariant."""
    if reference not in distances:
        raise KeyError(f"reference measurement '{reference}' not found in distances")
    scale = distances[reference]
    if scale <= 0:
        raise ValueError(f"reference measurement '{reference}' must be positive, got {scale}")
    return {name: value / scale for name, value in distances.items() if name != reference}


def compare_to_reference(
    candidate_distances: dict[str, float], reference_distances: dict[str, float]
) -> dict[str, dict[str, float]]:
    """Per-measurement absolute and percent deviation of candidate vs. one specific reference.

    This compares against one authored head's own values (per-identity baseline),
    not a pooled population statistic.
    """
    report: dict[str, dict[str, float]] = {}
    for name, ref_value in reference_distances.items():
        if name not in candidate_distances:
            continue
        cand_value = candidate_distances[name]
        abs_diff = cand_value - ref_value
        pct_diff = (abs_diff / ref_value * 100) if ref_value else float("inf")
        report[name] = {
            "reference": ref_value,
            "candidate": cand_value,
            "abs_diff": abs_diff,
            "pct_diff": pct_diff,
        }
    return report
