"""Extract landmark positions from generated variation assets via nearest-surface projection.

Generated variations (e.g. Gen3d_testing/synthetic_heads/dataset_5.1/.../cutted_mat_fixed/*.glb)
do not share the authored heads' fixed mesh topology -- vertex count differs per
file. So instead of indexing a fixed vertex, this takes each landmark's 3D
position from the variation's authored *parent* (already extracted via
extract_landmarks.py) and snaps it onto the closest point on the variation's
own mesh surface (a read-only BVH nearest-point query -- no topology edits).

Run with Blender (not a plain Python interpreter).

Single asset:
    blender --background --factory-startup --python blender/extract_landmarks_variation.py -- \\
        path/to/variation.glb --parent-landmarks landmarks_raw/<parent_stem>.json \\
        [--output out.json] [--mesh-name-hint head]

Batch (many variants in one Blender process -- each can have a different parent):
    blender --background --factory-startup --python blender/extract_landmarks_variation.py -- \\
        --manifest manifest.json

``manifest.json`` is a JSON array of
``{"asset": ..., "parent_landmarks": ..., "output": ..., "mesh_name_hint": "head"}``
objects (``mesh_name_hint`` optional, defaults to "head").

A large snap distance for a given point is itself a signal -- it means that
facial feature moved far from where the parent's version of it was.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
except ImportError as error:  # pragma: no cover - guidance for misuse
    raise SystemExit(
        "This script requires Blender's Python (bpy). Run it with:\n"
        "  blender --background --factory-startup --python blender/extract_landmarks_variation.py -- <asset> ..."
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
IMPORTABLE_EXTENSIONS = {".fbx", ".gltf", ".glb"}
DEFAULT_MESH_NAME_HINT = "head"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("asset", type=Path, nargs="?", help="Single variation asset to measure (.glb / .fbx).")
    parser.add_argument(
        "--parent-landmarks",
        type=Path,
        default=None,
        help="Raw landmark JSON for this variation's authored parent (from extract_landmarks.py).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: <asset_stem>.json next to the asset).")
    parser.add_argument(
        "--mesh-name-hint",
        default=DEFAULT_MESH_NAME_HINT,
        help="Substring (case-insensitive) preferred when choosing the head mesh among imported objects.",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="JSON manifest of many variants to process.")
    args = parser.parse_args(argv)
    if args.manifest is None and (args.asset is None or args.parent_landmarks is None):
        parser.error("provide either <asset> + --parent-landmarks, or --manifest")
    return args


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a JSON array: {path}")
    return data


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_asset(path: Path) -> list:
    suffix = path.suffix.lower()
    before = set(bpy.data.objects)
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix in {".gltf", ".glb"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"Unsupported asset type: {suffix}")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        raise RuntimeError(f"No objects were imported from {path}")
    return imported


def find_head_mesh(imported_objects: list, name_hint: str):
    """Prefer a mesh whose name contains ``name_hint``; otherwise the largest mesh by vertex count.

    Variation files don't follow one fixed naming convention (unlike the
    authored set), so this is more permissive than extract_landmarks.py's
    matcher. All candidate mesh names are printed for visibility.
    """
    meshes = [obj for obj in imported_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects were imported.")
    print(f"Mesh objects found: {[(o.name, len(o.data.vertices)) for o in meshes]}")

    hint = name_hint.lower()
    named_matches = [obj for obj in meshes if hint in obj.name.lower() and "body" not in obj.name.lower()]
    if named_matches:
        if len(named_matches) > 1:
            print(f"WARNING: multiple '{name_hint}'-matching meshes; using the largest by vertex count.")
        return max(named_matches, key=lambda o: len(o.data.vertices))

    print(f"No mesh name matched '{name_hint}'; falling back to the largest mesh by vertex count.")
    return max(meshes, key=lambda o: len(o.data.vertices))


def build_bvh(obj) -> BVHTree:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return BVHTree.FromObject(obj, depsgraph)


def project_landmarks(bvh: BVHTree, parent_landmarks: dict[str, list[float]]) -> tuple[dict, dict]:
    landmarks: dict[str, list[float]] = {}
    snap_distances: dict[str, float] = {}
    for name, position in parent_landmarks.items():
        location, _normal, _index, distance = bvh.find_nearest(Vector(position))
        if location is None:
            raise RuntimeError(f"BVH nearest-point query failed for landmark '{name}'.")
        landmarks[name] = [location.x, location.y, location.z]
        snap_distances[name] = distance
    return landmarks, snap_distances


def process_variant(asset: Path, parent_landmarks_path: Path, output_path: Path, mesh_name_hint: str) -> None:
    parent_data = json.loads(parent_landmarks_path.read_text(encoding="utf-8"))
    parent_landmarks = parent_data["landmarks"]

    reset_scene()
    imported = import_asset(asset)
    head = find_head_mesh(imported, mesh_name_hint)
    bvh = build_bvh(head)
    landmarks, snap_distances = project_landmarks(bvh, parent_landmarks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "asset": str(asset),
                "parent_asset": parent_data.get("asset"),
                "mesh_object": head.name,
                "vertex_count": len(head.data.vertices),
                "landmarks": landmarks,
                "snap_distances": snap_distances,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    max_snap = max(snap_distances.values()) if snap_distances else 0.0
    print(f"  -> {output_path} (max snap distance: {max_snap:.5f})")


def main() -> None:
    args = parse_args()

    if args.manifest:
        entries = load_manifest(args.manifest.resolve())
        succeeded = 0
        failed = 0
        for entry in entries:
            asset = Path(entry["asset"]).resolve()
            parent_landmarks_path = Path(entry["parent_landmarks"]).resolve()
            output_path = Path(entry["output"]).resolve()
            mesh_name_hint = entry.get("mesh_name_hint", DEFAULT_MESH_NAME_HINT)
            print(f"Processing {asset.name} ...")
            try:
                process_variant(asset, parent_landmarks_path, output_path, mesh_name_hint)
                succeeded += 1
            except Exception as error:  # noqa: BLE001 - keep batch runs going, report at the end
                print(f"ERROR processing {asset}: {error}")
                failed += 1
        print(f"Done. {succeeded} succeeded, {failed} failed, out of {len(entries)}.")
        if failed and succeeded == 0:
            sys.exit(1)
        return

    asset = args.asset.resolve()
    output_path = args.output.resolve() if args.output else asset.parent / f"{asset.stem}.json"
    process_variant(asset, args.parent_landmarks.resolve(), output_path, args.mesh_name_hint)


if __name__ == "__main__":
    main()
