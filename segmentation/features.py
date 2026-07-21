"""Parse GenieSAM ISAT JSON into per-region geometric features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegionFeatures:
    """Geometry summary for one semantic (or derived) region."""

    category: str
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    area: float
    centroid: tuple[float, float]
    width: float
    height: float
    aspect: float  # width / height
    area_frac_image: float = 0.0
    area_frac_face: float = 0.0
    source: str = "isat"  # "isat" | "derived" | "aggregate"


@dataclass
class FeatureSet:
    """All region features for one segmented image."""

    image_width: int
    image_height: int
    regions: dict[str, RegionFeatures] = field(default_factory=dict)
    aggregates: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def image_area(self) -> float:
        return float(max(1, self.image_width * self.image_height))


def _bbox_from_points(points: list[list[float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = bbox
    return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0


def _polygon_area(points: list[list[float]]) -> float:
    """Shoelace area; falls back to bbox area if degenerate."""
    if len(points) < 3:
        bbox = _bbox_from_points(points)
        if not bbox:
            return 0.0
        xmin, ymin, xmax, ymax = bbox
        return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = float(points[i][0]), float(points[i][1])
        x2, y2 = float(points[(i + 1) % n][0]), float(points[(i + 1) % n][1])
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _union_bbox(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _features_from_bbox(
    category: str,
    bbox: tuple[float, float, float, float],
    area: float,
    image_area: float,
    face_area: float,
    source: str = "isat",
) -> RegionFeatures:
    xmin, ymin, xmax, ymax = bbox
    width = max(0.0, xmax - xmin)
    height = max(0.0, ymax - ymin)
    aspect = width / height if height > 1e-6 else 0.0
    return RegionFeatures(
        category=category,
        bbox=bbox,
        area=area,
        centroid=_centroid(bbox),
        width=width,
        height=height,
        aspect=aspect,
        area_frac_image=area / image_area,
        area_frac_face=area / face_area if face_area > 1e-6 else 0.0,
        source=source,
    )


def load_isat(path_or_data: Any) -> dict[str, Any]:
    """Load ISAT dict from a path or return the dict unchanged."""
    if isinstance(path_or_data, dict):
        return path_or_data
    from pathlib import Path
    import json

    path = Path(path_or_data)
    return json.loads(path.read_text(encoding="utf-8"))


def extract_features(isat: dict[str, Any]) -> FeatureSet:
    """Build RegionFeatures for each ISAT category (union of multi-contour)."""
    info = isat.get("info") or {}
    image_width = int(info.get("width") or 1)
    image_height = int(info.get("height") or 1)
    image_area = float(max(1, image_width * image_height))

    by_category: dict[str, list[dict[str, Any]]] = {}
    for obj in isat.get("objects") or []:
        category = str(obj.get("category") or "")
        if not category or category == "__background__":
            continue
        by_category.setdefault(category, []).append(obj)

    pending: dict[str, tuple[tuple[float, float, float, float], float]] = {}
    for category, objects in by_category.items():
        boxes: list[tuple[float, float, float, float]] = []
        total_area = 0.0
        for obj in objects:
            bbox = obj.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            else:
                pts = obj.get("segmentation") or []
                maybe = _bbox_from_points(pts)
                if maybe is None:
                    continue
                box = maybe
            boxes.append(box)
            area_val = obj.get("area")
            if isinstance(area_val, (int, float)) and area_val > 0:
                total_area += float(area_val)
            else:
                total_area += _polygon_area(obj.get("segmentation") or [])
        union = _union_bbox(boxes)
        if union is None:
            continue
        pending[category] = (union, total_area)

    face_area = pending.get("Core_Face", (None, 0.0))[1] or pending.get("Face", (None, 0.0))[1] or 0.0
    if face_area <= 0:
        face_area = image_area

    regions: dict[str, RegionFeatures] = {}
    for category, (bbox, area) in pending.items():
        regions[category] = _features_from_bbox(
            category, bbox, area, image_area, face_area, source="isat"
        )

    features = FeatureSet(
        image_width=image_width,
        image_height=image_height,
        regions=regions,
    )
    features.aggregates = _build_aggregates(regions, face_area)
    return features


def _build_aggregates(
    regions: dict[str, RegionFeatures],
    face_area: float,
) -> dict[str, dict[str, float]]:
    """Symmetric L/R aggregates for eyes, brows, pupils, ears."""
    aggregates: dict[str, dict[str, float]] = {}

    def pair(left: str, right: str, key: str) -> None:
        l_reg = regions.get(left)
        r_reg = regions.get(right)
        if not l_reg and not r_reg:
            return
        widths = [r.width for r in (l_reg, r_reg) if r]
        heights = [r.height for r in (l_reg, r_reg) if r]
        areas = [r.area for r in (l_reg, r_reg) if r]
        mean_w = sum(widths) / len(widths)
        mean_h = sum(heights) / len(heights)
        mean_area = sum(areas) / len(areas)
        inter_eye = 0.0
        if l_reg and r_reg:
            inter_eye = abs(r_reg.centroid[0] - l_reg.centroid[0])
        boxes = [r.bbox for r in (l_reg, r_reg) if r]
        union = _union_bbox(boxes)
        aggregates[key] = {
            "mean_width": mean_w,
            "mean_height": mean_h,
            "mean_area": mean_area,
            "mean_aspect": mean_w / mean_h if mean_h > 1e-6 else 0.0,
            "area_frac_face": mean_area / face_area if face_area > 1e-6 else 0.0,
            "inter_distance": inter_eye,
            "union_width": (union[2] - union[0]) if union else mean_w,
            "union_height": (union[3] - union[1]) if union else mean_h,
        }

    pair("Left Eye", "Right Eye", "eyes")
    pair("Left Eyebrow", "Right Eyebrow", "brows")
    pair("Left Pupil", "Right Pupil", "pupils")
    pair("Left Ear", "Right Ear", "ears")

    upper = regions.get("Upper Lip")
    lower = regions.get("Lower Lip")
    lip_boxes = [r.bbox for r in (upper, lower) if r]
    lip_union = _union_bbox(lip_boxes)
    if lip_union:
        lip_area = sum(r.area for r in (upper, lower) if r)
        lip_w = lip_union[2] - lip_union[0]
        lip_h = lip_union[3] - lip_union[1]
        aggregates["lips"] = {
            "mean_width": lip_w,
            "mean_height": lip_h,
            "mean_area": lip_area,
            "mean_aspect": lip_w / lip_h if lip_h > 1e-6 else 0.0,
            "area_frac_face": lip_area / face_area if face_area > 1e-6 else 0.0,
            "inter_distance": 0.0,
            "union_width": lip_w,
            "union_height": lip_h,
            "height_over_width": lip_h / lip_w if lip_w > 1e-6 else 0.0,
        }

    return aggregates


def add_region(
    features: FeatureSet,
    category: str,
    bbox: tuple[float, float, float, float],
    area: float | None = None,
    source: str = "derived",
) -> RegionFeatures:
    """Insert or replace a region and recompute face-normalized fractions."""
    xmin, ymin, xmax, ymax = bbox
    if area is None:
        area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    face = features.regions.get("Core_Face") or features.regions.get("Face")
    face_area = face.area if face else features.image_area
    region = _features_from_bbox(
        category, bbox, area, features.image_area, face_area, source=source
    )
    features.regions[category] = region
    return region
