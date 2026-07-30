"""Sort assets into good / mid / bad lists from joint-feature ratings.

Buckets (by count of ``\"bad\"`` joint markers):
  - good: 0 bad
  - mid:  1–2 bad
  - bad:  3+ bad

List files live under ``lists/good.json``, ``lists/mid.json``, ``lists/bad.json``.
Each file is a JSON array of asset names (e.g. ``\"foo.fbx\"``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

Bucket = Literal["good", "mid", "bad"]
BUCKET_FILES: tuple[tuple[Bucket, str], ...] = (
    ("good", "good.json"),
    ("mid", "mid.json"),
    ("bad", "bad.json"),
)


def count_bad_joint_features(joint_features: dict[str, Any] | None) -> int:
    """Return how many joint markers are rated ``bad``."""
    if not joint_features:
        return 0
    return sum(1 for value in joint_features.values() if value == "bad")


def classify_joint_features(joint_features: dict[str, Any] | None) -> Bucket:
    """Map joint-feature ratings to a quality bucket."""
    bad_count = count_bad_joint_features(joint_features)
    if bad_count == 0:
        return "good"
    if bad_count <= 2:
        return "mid"
    return "bad"


def _lists_dir(project_root: Path) -> Path:
    return Path(project_root) / "lists"


def _bucket_path(project_root: Path, bucket: Bucket) -> Path:
    return _lists_dir(project_root) / f"{bucket}.json"


def _load_asset_names(path: Path) -> list[str]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return [str(item) for item in data if isinstance(item, str)]
    if isinstance(data, dict):
        assets = data.get("assets", [])
        if isinstance(assets, list):
            return [str(item) for item in assets if isinstance(item, str)]
    return []


def _write_asset_names(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_sorted = sorted(dict.fromkeys(names), key=str.lower)
    path.write_text(json.dumps(unique_sorted, indent=2) + "\n", encoding="utf-8")


def sort_asset_by_joint_features(
    project_root: Path | str,
    asset_name: str,
    joint_features: dict[str, Any] | None,
) -> Bucket:
    """Place ``asset_name`` into the matching list and remove it from the others.

    Returns the bucket the asset was assigned to.
    """
    root = Path(project_root)
    target = classify_joint_features(joint_features)
    name = Path(asset_name).name

    for bucket, _filename in BUCKET_FILES:
        path = _bucket_path(root, bucket)
        names = _load_asset_names(path)
        if bucket == target:
            if name not in names:
                names.append(name)
        else:
            names = [item for item in names if item != name]
        _write_asset_names(path, names)

    return target
