"""One-time landmark calibration helper — run manually inside Blender's Scripting tab.

Not a headless CLI script (unlike render_head.py / extract_landmarks.py): open a
reference authored head in Blender's GUI, use it interactively.

Usage (pick mode):
    1. Import one reference authored head, e.g.
       authored_heads_var_v2/african/african_female_loPoly_0001_grp.fbx
    2. Select the head mesh object (named like `*_head_*_geo`), enter Edit Mode,
       vertex select, and click exactly one vertex for the landmark you're picking.
    3. In this file, set LANDMARK_NAME below to that landmark's name (must match
       a key in landmarks/landmark_config.json's "points").
    4. Open this file in Blender's Text Editor and run it (Alt+P).
    5. Repeat steps 2-4 for every landmark (see PROJECT_ROOT/landmarks/landmark_config.json
       for the full list).

Usage (verify mode):
    1. Set MODE = "verify" below and VERIFY_ASSET_PATH to a *different* reference
       head (ideally a different ethnicity/gender) already imported into the scene.
    2. Run this file. It spawns a small Empty at every configured landmark's
       vertex position on that mesh, named after the landmark, so you can
       visually confirm each one lands on the right facial feature.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import bpy

# Edit this if the repo moves.
PROJECT_ROOT = Path(r"c:\Users\auror\Documents\Github\Synthetic_Heads_Evaluation_Tool")
CONFIG_PATH = PROJECT_ROOT / "landmarks" / "landmark_config.json"

MODE = "pick"  # "pick" or "verify"

# --- pick mode ---
LANDMARK_NAME = "left_eye_outer"  # <- change this before each run

# --- verify mode ---
VERIFY_ASSET_PATH = ""  # informational only; the mesh must already be imported/selected


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _save_config(config: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _active_mesh_object():
    obj = bpy.context.active_object
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Select the head mesh object (enter Edit Mode on it) before running.")
    return obj


def _topology_hash(mesh) -> str:
    """Hash of the polygon vertex-index list; detects topology drift between assets."""
    digest = hashlib.sha256()
    for polygon in mesh.polygons:
        for index in polygon.vertices:
            digest.update(index.to_bytes(4, "little"))
    return digest.hexdigest()


def pick_landmark() -> None:
    obj = _active_mesh_object()
    obj.update_from_editmode()  # required for vertex.select to reflect the viewport selection
    selected = [v.index for v in obj.data.vertices if v.select]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected exactly 1 selected vertex, found {len(selected)}. "
            "Select one vertex in Edit Mode and try again."
        )
    index = selected[0]

    config = _load_config()
    if LANDMARK_NAME not in config["points"]:
        raise KeyError(f"'{LANDMARK_NAME}' is not a known landmark name in {CONFIG_PATH}")

    config["points"][LANDMARK_NAME] = index
    if config.get("topology_hash") is None:
        config["topology_hash"] = _topology_hash(obj.data)
        config["expected_vertex_count"] = len(obj.data.vertices)
        print(f"Recorded topology hash from '{obj.name}' ({len(obj.data.vertices)} verts).")
    _save_config(config)

    remaining = [name for name, idx in config["points"].items() if idx is None]
    print(f"Set '{LANDMARK_NAME}' -> vertex {index}.")
    if remaining:
        print(f"{len(remaining)} landmark(s) still unset: {', '.join(remaining)}")
    else:
        print("All landmarks are calibrated.")


def verify_landmarks() -> None:
    obj = _active_mesh_object()
    config = _load_config()

    hash_now = _topology_hash(obj.data)
    if config.get("topology_hash") and hash_now != config["topology_hash"]:
        print(
            "WARNING: this mesh's topology hash does not match landmark_config.json. "
            "Vertex indices will NOT line up with the correct features on this asset."
        )

    collection = bpy.data.collections.get("Landmark Verification")
    if collection is None:
        collection = bpy.data.collections.new("Landmark Verification")
        bpy.context.scene.collection.children.link(collection)

    placed = 0
    for name, index in config["points"].items():
        if index is None:
            continue
        if index >= len(obj.data.vertices):
            print(f"SKIP '{name}': index {index} is out of range for this mesh ({len(obj.data.vertices)} verts).")
            continue
        world_pos = obj.matrix_world @ obj.data.vertices[index].co
        empty = bpy.data.objects.new(f"landmark_{name}", None)
        empty.empty_display_type = "SPHERE"
        empty.empty_display_size = 0.003
        empty.location = world_pos
        collection.objects.link(empty)
        placed += 1
    print(f"Placed {placed} verification marker(s) in the 'Landmark Verification' collection.")


def main() -> None:
    if MODE == "pick":
        pick_landmark()
    elif MODE == "verify":
        verify_landmarks()
    else:
        raise ValueError(f"Unknown MODE: {MODE!r} (expected 'pick' or 'verify')")


if __name__ == "__main__":
    main()
