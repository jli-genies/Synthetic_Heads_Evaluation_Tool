"""Sort assets into good / bad lists from the reviewer's overall verdict.

The verdict comes from a single Good/Bad control on the Face Proportions tab
(pre-filled from landmarks/range_classifier.py's RMS-combined score against
its tuned threshold, then reviewer-correctable) -- never derived by tallying
flagged regions, which would reintroduce the false-positive compounding the
RMS design exists to avoid (see landmarks/range_classifier.py's docstring).

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


def variation_folder_for(asset_path: Path | str) -> str | None:
    """Best-effort recovery of the ``variation_*`` batch folder two levels above
    ``asset_path`` (``<variation_folder>/<subfolder>/<asset>``), or ``None`` for
    authored heads that aren't nested under one.
    """
    grandparent = Path(asset_path).resolve().parent.parent.name
    return grandparent if grandparent.startswith("variation_") else None


def render_cache_key(asset_path: Path | str) -> str:
    """Cache key for ``renders/<key>/``.

    Plain ``asset.stem`` collides whenever a variation batch reuses the same
    mesh filename across sibling ``variation_*`` folders (e.g.
    ``variation_small_lips_001`` vs. ``_002``/``_003``), silently serving one
    asset's previews for another. Folding ``variation_folder`` into the key
    keeps each variation's renders in their own folder.
    """
    stem = Path(asset_path).stem
    variation_folder = variation_folder_for(asset_path)
    return f"{variation_folder}__{stem}" if variation_folder else stem


def classify_overall_verdict(overall_verdict: str | None) -> Bucket:
    """Map the reviewer's single overall Good/Bad control to a quality bucket.

    Defaults to "good" for "Not specified" -- matches the pre-region-panel
    behavior where every rating defaulted to good unless a reviewer flagged
    something.
    """
    return "bad" if overall_verdict == "bad" else "good"


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


def sort_asset_by_overall_verdict(
    project_root: Path | str,
    asset_name: str,
    overall_verdict: str | None,
    variation_folder: str | None = None,
) -> Bucket:
    """Place ``asset_name`` (scoped to ``variation_folder``, if any) into the
    matching list and remove it from the others.

    Returns the bucket the asset was assigned to.
    """
    root = Path(project_root)
    target = classify_overall_verdict(overall_verdict)
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
