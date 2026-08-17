"""Load a generated variant's generation-time chaos-joint parameters.

Each variant's actual chaos_joints values (the per-bind location/rotation/scale
HeadGen applied to produce it) live in a sibling JSON file, derivable purely
from the variant's own path:

    {variation_name}/{cutted_variant}/{...}_auth_{eth}_{g}_{id}_frame_{NNNN}_subdiv_head_var_{V}.glb
    {variation_name}/auth_{eth}_{g}_{id}/frame_{NNNN}/final_frame*.json

This reduces each bind's 7 raw numbers (3 location + 4 quaternion + 3 scale)
into one deviation-magnitude scalar, normalized against the generator's own
configured chaos limits (transform_max/rotate_max/scale_max, read from the
same JSON's config_snapshot) so location/rotation/scale contribute comparably
despite being in different units.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

VARIANT_NAME_PATTERN = re.compile(
    r"^(?P<ethnicity>[a-z]+)_(?P<gender>male|female)_(?P<auth_dir>auth_[a-z]+_[mf]_\d{4})"
    r"_frame_(?P<frame>\d+)_subdiv_head_var_(?P<var>\d+)$",
    re.IGNORECASE,
)

# The pre-final "cutted" export (bare cutted/ subfolder) drops the leading
# <ethnicity>_<gender> prefix and the trailing _var_N suffix, e.g.
# "auth_african_f_0022_frame_0001_subdiv_head.glb".
BARE_CUTTED_NAME_PATTERN = re.compile(
    r"^auth_(?P<ethnicity>[a-z]+)_(?P<gender_letter>[mf])_(?P<head_id>\d{4})_frame_(?P<frame>\d+)_subdiv_head$",
    re.IGNORECASE,
)

# The literal authored head itself (no chaos variation applied at all -- these
# have no final_frame*.json to look up), e.g.
# "auth_african_f_0012_frame_0012_subdiv_auth_african_f_0012_head.glb". The
# <eth>_<g>_<id> triple is repeated before "_head" instead of the plain
# "_subdiv_head" suffix BARE_CUTTED_NAME_PATTERN expects.
AUTHORED_HEAD_NAME_PATTERN = re.compile(
    r"^auth_(?P<ethnicity>[a-z]+)_(?P<gender_letter>[mf])_(?P<head_id>\d{4})_frame_(?P<frame>\d+)"
    r"_subdiv_auth_[a-z]+_[mf]_\d{4}_head$",
    re.IGNORECASE,
)


def parse_variant_stem(stem: str) -> dict[str, str] | None:
    """Parse a generated-variant filename stem into its authored-parent identity.

    Handles three naming conventions seen in the dataset:
      - final tagged form: <eth>_<gender>_auth_<eth>_<g>_<id>_frame_<NNNN>_subdiv_head_var_<N>
      - bare pre-cut form: auth_<eth>_<g>_<id>_frame_<NNNN>_subdiv_head (no _var_N suffix)
      - literal authored head: auth_<eth>_<g>_<id>_frame_<NNNN>_subdiv_auth_<eth>_<g>_<id>_head
        (no chaos variation applied -- callers should expect no final_frame*.json
        for these, i.e. compute_bind_magnitudes will raise FileNotFoundError)

    Returns {"ethnicity", "gender", "head_id", "frame", "auth_dir"}, or None if
    none of the conventions match.
    """
    match = VARIANT_NAME_PATTERN.match(stem)
    if match:
        return {
            "ethnicity": match["ethnicity"].lower(),
            "gender": match["gender"].lower(),
            "head_id": match["auth_dir"].rsplit("_", 1)[-1],
            "frame": match["frame"],
            "auth_dir": match["auth_dir"],
        }
    match = BARE_CUTTED_NAME_PATTERN.match(stem) or AUTHORED_HEAD_NAME_PATTERN.match(stem)
    if match:
        gender_letter = match["gender_letter"].lower()
        gender = "male" if gender_letter == "m" else "female"
        return {
            "ethnicity": match["ethnicity"].lower(),
            "gender": gender,
            "head_id": match["head_id"],
            "frame": match["frame"],
            "auth_dir": f"auth_{match['ethnicity'].lower()}_{gender_letter}_{match['head_id']}",
        }
    return None


def find_chaos_joints_json(variant_path: Path) -> Path:
    """Derive the sibling final_frame*.json path from a variant glb/fbx's own path."""
    parsed = parse_variant_stem(variant_path.stem)
    if parsed is None:
        raise ValueError(
            f"'{variant_path.stem}' doesn't match either known variant naming convention "
            "(<ethnicity>_<gender>_auth_..._frame_NNNN_subdiv_head_var_N, or "
            "auth_<eth>_<g>_<id>_frame_NNNN_subdiv_head)."
        )
    variation_root = variant_path.parent.parent
    frame_dir = variation_root / parsed["auth_dir"] / f"frame_{parsed['frame']}"
    candidates = sorted(frame_dir.glob("final_frame*.json"))
    if not candidates:
        raise FileNotFoundError(f"No chaos-joints JSON found under {frame_dir}")
    if len(candidates) > 1:
        print(f"WARNING: multiple chaos-joints JSON in {frame_dir}; using {candidates[-1].name}")
    return candidates[-1]


def load_chaos_joints(json_path: Path) -> tuple[dict[str, dict], dict[str, float]]:
    """Returns (chaos_joints dict, {transform_max, rotate_max, scale_max})."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    limits_cfg = data.get("config_snapshot", {}).get("chaos_joints", {})
    limits = {
        "transform_max": float(limits_cfg.get("transform_max") or 0.2),
        "rotate_max": float(limits_cfg.get("rotate_max") or 10.0),
        "scale_max": float(limits_cfg.get("scale_max") or 0.2),
    }
    return data["chaos_joints"], limits


def _quaternion_angle_degrees(quat: list[float]) -> float:
    """Rotation angle (degrees) a quaternion represents relative to identity."""
    w = max(-1.0, min(1.0, quat[0]))
    return math.degrees(2 * math.acos(abs(w)))


def bind_deviation_magnitude(bind: dict, limits: dict[str, float]) -> float:
    """One scalar per bind marker: location + rotation + scale deviation from
    neutral, each normalized against the generator's own configured chaos
    limits, combined as a Euclidean norm."""
    loc = bind.get("location", [0.0, 0.0, 0.0])
    loc_mag = math.sqrt(sum(c * c for c in loc))

    quat = bind.get("rotation_quaternion", [1.0, 0.0, 0.0, 0.0])
    rot_deg = _quaternion_angle_degrees(quat)

    scale = bind.get("scale", [1.0, 1.0, 1.0])
    scale_mag = math.sqrt(sum((s - 1.0) ** 2 for s in scale))

    norm_loc = loc_mag / limits["transform_max"]
    norm_rot = rot_deg / limits["rotate_max"]
    norm_scale = scale_mag / limits["scale_max"]

    return math.sqrt(norm_loc**2 + norm_rot**2 + norm_scale**2)


def compute_bind_magnitudes(variant_path: Path) -> dict[str, float]:
    """{bind_name: deviation_magnitude} for every chaos joint applied to this variant."""
    json_path = find_chaos_joints_json(variant_path)
    chaos_joints, limits = load_chaos_joints(json_path)
    return {name: bind_deviation_magnitude(bind, limits) for name, bind in chaos_joints.items()}
