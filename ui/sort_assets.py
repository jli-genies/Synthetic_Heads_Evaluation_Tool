"""Sort assets into good / bad lists from joint-feature ratings.

Buckets (by count of ``"bad"`` joint markers):
  - good: 0 bad
  - bad:  1+ bad

List files live under ``lists/good.json`` and ``lists/bad.json``. Each file is
a JSON array of ``{"name": <asset filename>, "variation_folder": <str|null>}``
entries. ``variation_folder`` disambiguates generated-variation assets whose
filename repeats across different HeadGen variation batches (the same
``..._subdiv_head_var_0.glb`` name can exist under several ``variation_*``
folders with different chaos-joint values); it's null for authored heads,
which don't have that collision. Legacy plain-string entries are still read
(as ``variation_folder: None``) for backward compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

Bucket = Literal["good", "bad"]
BUCKET_FILES: tuple[tuple[Bucket, str], ...] = (
    ("good", "good.json"),
    ("bad", "bad.json"),
)


def count_bad_joint_features(joint_features: dict[str, Any] | None) -> int:
    """Return how many joint markers are rated ``bad``."""
    if not joint_features:
        return 0
    return sum(1 for value in joint_features.values() if value == "bad")


def classify_joint_features(joint_features: dict[str, Any] | None) -> Bucket:
    """Map joint-feature ratings to a quality bucket."""
    if count_bad_joint_features(joint_features) == 0:
        return "good"
    return "bad"


def _lists_dir(project_root: Path) -> Path:
    return Path(project_root) / "lists"


def _bucket_path(project_root: Path, bucket: Bucket) -> Path:
    return _lists_dir(project_root) / f"{bucket}.json"


def _load_entries(path: Path) -> list[dict[str, Any]]:
    """Each entry: {"name": str, "variation_folder": str | None}.

    Accepts legacy plain-string-array files (from before variation_folder was
    tracked) for backward compatibility.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        data = data.get("assets", [])
    if not isinstance(data, list):
        return []

    entries: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, str):
            entries.append({"name": item, "variation_folder": None})
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            entries.append({"name": item["name"], "variation_folder": item.get("variation_folder")})
    return entries


def _write_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique: dict[tuple[str, str | None], dict[str, Any]] = {
        (entry["name"], entry.get("variation_folder")): entry for entry in entries
    }
    ordered = sorted(unique.values(), key=lambda e: (e["name"].lower(), e.get("variation_folder") or ""))
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")


def sort_asset_by_joint_features(
    project_root: Path | str,
    asset_name: str,
    joint_features: dict[str, Any] | None,
    variation_folder: str | None = None,
) -> Bucket:
    """Place ``asset_name`` (scoped to ``variation_folder``, if any) into the
    matching list and remove it from the others.

    Returns the bucket the asset was assigned to.
    """
    root = Path(project_root)
    target = classify_joint_features(joint_features)
    name = Path(asset_name).name
    key = (name, variation_folder)
    # A legacy (name, None) entry predates variation_folder tracking and is
    # ambiguous once we know this asset's real folder -- treat it as the same
    # entry so re-tagging actually moves it instead of leaving a stale
    # duplicate behind in its old bucket.
    keys_to_remove = {key}
    if variation_folder is not None:
        keys_to_remove.add((name, None))

    for bucket, _filename in BUCKET_FILES:
        path = _bucket_path(root, bucket)
        entries = [
            e for e in _load_entries(path) if (e["name"], e.get("variation_folder")) not in keys_to_remove
        ]
        if bucket == target:
            entries.append({"name": name, "variation_folder": variation_folder})
        _write_entries(path, entries)

    return target
