"""Derive missing anatomical ROIs from Core_Face + eye/lip anchors."""

from __future__ import annotations

from .features import FeatureSet, add_region


def _clamp_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    face: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    fx0, fy0, fx1, fy1 = face
    xmin = max(xmin, fx0)
    ymin = max(ymin, fy0)
    xmax = min(xmax, fx1)
    ymax = min(ymax, fy1)
    if xmax - xmin < 2 or ymax - ymin < 2:
        return None
    return xmin, ymin, xmax, ymax


def apply_face_partitions(features: FeatureSet) -> FeatureSet:
    """Estimate forehead/nose/cheeks/chin/jaw bands inside Core_Face.

    Uses eye and lip bboxes as vertical anchors when available; otherwise falls
    back to fractional bands of the face box. Regions are scaffolding for
    heuristics — not ground-truth masks.
    """
    face = features.regions.get("Core_Face") or features.regions.get("Face")
    if not face:
        return features

    fx0, fy0, fx1, fy1 = face.bbox
    face_w = face.width
    face_h = face.height

    eyes = features.aggregates.get("eyes")
    brows = features.aggregates.get("brows")
    lips = features.aggregates.get("lips")

    left_eye = features.regions.get("Left Eye")
    right_eye = features.regions.get("Right Eye")
    upper_lip = features.regions.get("Upper Lip")
    lower_lip = features.regions.get("Lower Lip")

    # Vertical anchors (image y increases downward).
    brow_y = None
    if brows and "Left Eyebrow" in features.regions and "Right Eyebrow" in features.regions:
        brow_y = min(
            features.regions["Left Eyebrow"].bbox[1],
            features.regions["Right Eyebrow"].bbox[1],
        )
    elif left_eye or right_eye:
        eye_tops = [r.bbox[1] for r in (left_eye, right_eye) if r]
        brow_y = min(eye_tops) - 0.05 * face_h if eye_tops else None

    eye_bottom = None
    if left_eye or right_eye:
        eye_bottoms = [r.bbox[3] for r in (left_eye, right_eye) if r]
        eye_bottom = max(eye_bottoms) if eye_bottoms else None
    elif eyes:
        # Approximate from face thirds if only aggregate exists without regions.
        eye_bottom = fy0 + 0.45 * face_h

    lip_top = upper_lip.bbox[1] if upper_lip else (fy0 + 0.62 * face_h)
    lip_bottom = lower_lip.bbox[3] if lower_lip else (fy0 + 0.78 * face_h)
    if lips and not upper_lip and not lower_lip:
        # Unlikely without regions, but keep consistent.
        lip_top = fy0 + 0.62 * face_h
        lip_bottom = fy0 + 0.78 * face_h

    if brow_y is None:
        brow_y = fy0 + 0.28 * face_h
    if eye_bottom is None:
        eye_bottom = fy0 + 0.48 * face_h

    # Horizontal thirds for nose (center) and cheeks (sides).
    third = face_w / 3.0
    mid_x0 = fx0 + third
    mid_x1 = fx1 - third

    # Forehead: above brow line.
    forehead = _clamp_box(fx0, fy0, fx1, brow_y, face.bbox)
    if forehead:
        add_region(features, "derived_forehead", forehead)

    # Nose: between eyes and upper lip, center third.
    nose = _clamp_box(mid_x0, eye_bottom, mid_x1, lip_top, face.bbox)
    if nose:
        add_region(features, "derived_nose", nose)

    # Cheeks: lateral thirds of mid-face (eye bottom → lip top).
    left_cheek = _clamp_box(fx0, eye_bottom, mid_x0, lip_top, face.bbox)
    right_cheek = _clamp_box(mid_x1, eye_bottom, fx1, lip_top, face.bbox)
    if left_cheek:
        add_region(features, "derived_cheek_left", left_cheek)
    if right_cheek:
        add_region(features, "derived_cheek_right", right_cheek)
    if left_cheek and right_cheek:
        cheeks_union = (
            min(left_cheek[0], right_cheek[0]),
            min(left_cheek[1], right_cheek[1]),
            max(left_cheek[2], right_cheek[2]),
            max(left_cheek[3], right_cheek[3]),
        )
        add_region(features, "derived_cheeks", cheeks_union)

    # Mouth: lip union already in aggregates; also store a bbox region.
    if upper_lip or lower_lip:
        boxes = [r.bbox for r in (upper_lip, lower_lip) if r]
        mouth = (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
        add_region(features, "derived_mouth", mouth)

    # Chin: below lower lip, center half of face width.
    chin_x0 = fx0 + 0.25 * face_w
    chin_x1 = fx1 - 0.25 * face_w
    chin = _clamp_box(chin_x0, lip_bottom, chin_x1, fy1, face.bbox)
    if chin:
        add_region(features, "derived_chin", chin)

    # Jaw: below lips, full face width (wider than chin).
    jaw = _clamp_box(fx0, lip_bottom, fx1, fy1, face.bbox)
    if jaw:
        add_region(features, "derived_jaw", jaw)

    return features
