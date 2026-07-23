"""Pluggable geometry → discrete tag value heuristics."""

from __future__ import annotations

from typing import Any, Callable

from .features import FeatureSet

HeuristicFn = Callable[[FeatureSet, dict[str, Any]], str | list[str] | None]


def _ternary(
    value: float,
    low_max: float,
    high_min: float,
    low_label: str,
    mid_label: str,
    high_label: str,
) -> str:
    if value <= low_max:
        return low_label
    if value >= high_min:
        return high_label
    return mid_label


def head_size_from_aspect(features: FeatureSet, params: dict[str, Any]) -> str | None:
    face = features.regions.get("Core_Face") or features.regions.get("Face")
    if not face or face.aspect <= 0:
        return None
    thr = params.get("thresholds") or {}
    return _ternary(
        face.aspect,
        float(thr.get("thin_max", 0.72)),
        float(thr.get("wide_min", 0.92)),
        "thin",
        "mid",
        "wide",
    )


def head_length_from_aspect(features: FeatureSet, params: dict[str, Any]) -> str | None:
    """Tall face (low width/height) → long; wide face → short."""
    face = features.regions.get("Core_Face") or features.regions.get("Face")
    if not face or face.aspect <= 0:
        return None
    thr = params.get("thresholds") or {}
    return _ternary(
        face.aspect,
        float(thr.get("long_max", 0.72)),
        float(thr.get("short_min", 0.92)),
        "long",
        "mid",
        "short",
    )


def eyes_size_from_area(features: FeatureSet, params: dict[str, Any]) -> str | None:
    eyes = features.aggregates.get("eyes")
    if not eyes:
        return None
    thr = params.get("thresholds") or {}
    return _ternary(
        float(eyes["area_frac_face"]),
        float(thr.get("small_max", 0.012)),
        float(thr.get("large_min", 0.028)),
        "small",
        "mid",
        "large",
    )


def eyes_height_placement(features: FeatureSet, params: dict[str, Any]) -> str | None:
    """Eye centroid Y relative to Core_Face (0=top, 1=bottom)."""
    face = features.regions.get("Core_Face") or features.regions.get("Face")
    left = features.regions.get("Left Eye")
    right = features.regions.get("Right Eye")
    if not face or (not left and not right):
        return None
    ys = [r.centroid[1] for r in (left, right) if r]
    eye_y = sum(ys) / len(ys)
    frac = (eye_y - face.bbox[1]) / face.height if face.height > 1e-6 else 0.5
    thr = params.get("thresholds") or {}
    # Smaller frac = higher on face.
    return _ternary(
        frac,
        float(thr.get("high_max", 0.38)),
        float(thr.get("low_min", 0.52)),
        "high",
        "mid",
        "low",
    )


def eyes_width_placement(features: FeatureSet, params: dict[str, Any]) -> str | None:
    """Inter-eye distance / face width → close-set / mid-set / wide-set."""
    face = features.regions.get("Core_Face") or features.regions.get("Face")
    eyes = features.aggregates.get("eyes")
    if not face or not eyes or eyes.get("inter_distance", 0) <= 0:
        return None
    ratio = float(eyes["inter_distance"]) / face.width if face.width > 1e-6 else 0.0
    thr = params.get("thresholds") or {}
    return _ternary(
        ratio,
        float(thr.get("close_max", 0.28)),
        float(thr.get("wide_min", 0.40)),
        "close-set",
        "mid-set",
        "wide-set",
    )


def lips_proportion_from_hw(features: FeatureSet, params: dict[str, Any]) -> str | None:
    """Lip height/width extremes only; mid band left blank for the user."""
    lips = features.aggregates.get("lips")
    if not lips:
        return None
    thr = params.get("thresholds") or {}
    value = float(lips.get("height_over_width", 0.0))
    small_max = float(thr.get("small_max", 0.28))
    large_min = float(thr.get("large_min", 0.45))
    if value <= small_max:
        return "small"
    if value >= large_min:
        return "large"
    return None


def lips_proportion_from_upper_lower(
    features: FeatureSet, params: dict[str, Any]
) -> str | None:
    """Lip size from Upper+Lower Lip heights vs face height; mid band blank."""
    face = features.regions.get("Core_Face") or features.regions.get("Face")
    upper = features.regions.get("Upper Lip")
    lower = features.regions.get("Lower Lip")
    if not face or (not upper and not lower):
        return None
    lip_h = (upper.height if upper else 0.0) + (lower.height if lower else 0.0)
    value = lip_h / face.height if face.height > 1e-6 else 0.0
    thr = params.get("thresholds") or {}
    small_max = float(thr.get("small_max", 0.045))
    large_min = float(thr.get("large_min", 0.090))
    if value <= small_max:
        return "small"
    if value >= large_min:
        return "large"
    return None


def lips_shape_from_upper_lower(
    features: FeatureSet, params: dict[str, Any]
) -> list[str] | None:
    """Infer lips_shape.shape from separate Upper/Lower Lip boxes.

    Priority: top/bottom-heavy balance, then flat/thin/full from combined
    thickness. Mid / ambiguous cases return None so the user decides.
    """
    face = features.regions.get("Core_Face") or features.regions.get("Face")
    upper = features.regions.get("Upper Lip")
    lower = features.regions.get("Lower Lip")
    if not upper or not lower or not face:
        return None

    thr = params.get("thresholds") or {}
    top_heavy_min = float(thr.get("top_heavy_min", 1.2))
    bottom_heavy_min = float(thr.get("bottom_heavy_min", 1.2))
    thin_max = float(thr.get("thin_max", 0.045))
    full_min = float(thr.get("full_min", 0.075))
    flat_max_hw = float(thr.get("flat_max_hw", 0.25))

    uh = upper.height
    lh = lower.height
    if uh <= 1e-6 or lh <= 1e-6:
        return None

    # Balance: compare vertical extent of each lip box.
    if uh / lh >= top_heavy_min:
        return ["top-heavy"]
    if lh / uh >= bottom_heavy_min:
        return ["bottom-heavy"]

    # Thickness: combined lip height vs face, plus union aspect for flat.
    rel = (uh + lh) / face.height if face.height > 1e-6 else 0.0
    lips = features.aggregates.get("lips") or {}
    union_hw = float(lips.get("height_over_width", 0.0))
    if union_hw <= 0.0:
        union_w = max(upper.width, lower.width)
        union_h = (max(upper.bbox[3], lower.bbox[3]) - min(upper.bbox[1], lower.bbox[1]))
        union_hw = union_h / union_w if union_w > 1e-6 else 0.0

    if union_hw <= flat_max_hw and rel <= thin_max:
        return ["flat"]
    if rel <= thin_max:
        return ["thin"]
    if rel >= full_min:
        return ["full"]
    return None


def mouth_proportion_from_width(features: FeatureSet, params: dict[str, Any]) -> str | None:
    face = features.regions.get("Core_Face") or features.regions.get("Face")
    lips = features.aggregates.get("lips")
    if not face or not lips:
        return None
    ratio = float(lips["union_width"]) / face.width if face.width > 1e-6 else 0.0
    thr = params.get("thresholds") or {}
    narrow_max = float(thr.get("narrow_max", 0.32))
    wide_min = float(thr.get("wide_min", 0.48))
    if ratio <= narrow_max:
        return "narrow"
    if ratio >= wide_min:
        return "wide"
    return None


def brow_proportion_from_aspect(features: FeatureSet, params: dict[str, Any]) -> str | None:
    """High aspect (thin long brows) → shallow; low aspect → heavy."""
    brows = features.aggregates.get("brows")
    if not brows or brows.get("mean_aspect", 0) <= 0:
        return None
    aspect = float(brows["mean_aspect"])
    thr = params.get("thresholds") or {}
    shallow_min = float(thr.get("shallow_min", 6.0))
    heavy_max = float(thr.get("heavy_max", 3.5))
    if aspect >= shallow_min:
        return "shallow"
    if aspect <= heavy_max:
        return "heavy"
    return "wide"


HEURISTICS: dict[str, HeuristicFn] = {
    "head_size_from_aspect": head_size_from_aspect,
    "head_length_from_aspect": head_length_from_aspect,
    "eyes_size_from_area": eyes_size_from_area,
    "eyes_height_placement": eyes_height_placement,
    "eyes_width_placement": eyes_width_placement,
    "lips_proportion_from_hw": lips_proportion_from_hw,
    "lips_proportion_from_upper_lower": lips_proportion_from_upper_lower,
    "lips_shape_from_upper_lower": lips_shape_from_upper_lower,
    "mouth_proportion_from_width": mouth_proportion_from_width,
    "brow_proportion_from_aspect": brow_proportion_from_aspect,
}


def run_heuristic(
    name: str, features: FeatureSet, params: dict[str, Any]
) -> str | list[str] | None:
    fn = HEURISTICS.get(name)
    if not fn:
        return None
    return fn(features, params)
