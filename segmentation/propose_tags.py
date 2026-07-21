"""Orchestrate ISAT → features → partitions → proposed tag values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .face_partitions import apply_face_partitions
from .features import extract_features, load_isat
from .heuristics import run_heuristic

_TAG_MAP_PATH = Path(__file__).resolve().parent / "tag_map.json"


def load_tag_map(path: Path | None = None) -> dict[str, Any]:
    map_path = path or _TAG_MAP_PATH
    return json.loads(map_path.read_text(encoding="utf-8"))


def propose_tags_from_isat(
    isat_path_or_data: Any,
    tag_map: dict[str, Any] | None = None,
) -> dict[str, dict[str, str | list[str]]]:
    """Return proposed tags shaped like TagPanel.tags() from an ISAT JSON."""
    isat = load_isat(isat_path_or_data)
    features = extract_features(isat)
    apply_face_partitions(features)

    mapping = tag_map if tag_map is not None else load_tag_map()
    proposed: dict[str, dict[str, str | list[str]]] = {}

    for entry in mapping.get("mappings") or []:
        heuristic = entry.get("heuristic")
        category = entry.get("category")
        field = entry.get("field")
        if not heuristic or not category or not field:
            continue
        value = run_heuristic(str(heuristic), features, entry)
        if value is None or value == "":
            continue
        proposed.setdefault(str(category), {})[str(field)] = value

    return proposed


def _field_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, list):
        return len(value) == 0
    return False


def merge_tags(
    existing: dict[str, dict[str, Any]] | None,
    proposed: dict[str, dict[str, str | list[str]]] | None,
) -> tuple[dict[str, dict[str, str | list[str]]], int]:
    """Fill empty fields from proposed; keep existing values.

    Returns (merged_tags, number_of_fields_filled).
    """
    existing = existing or {}
    proposed = proposed or {}
    merged: dict[str, dict[str, str | list[str]]] = {
        cat: dict(fields) for cat, fields in existing.items()
    }
    filled = 0

    for category, fields in proposed.items():
        for field, value in fields.items():
            current = merged.get(category, {}).get(field)
            if not _field_is_empty(current):
                continue
            if _field_is_empty(value):
                continue
            merged.setdefault(category, {})[field] = value
            filled += 1

    return merged, filled
