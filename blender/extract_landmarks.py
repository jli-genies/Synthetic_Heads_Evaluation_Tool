"""Extract calibrated landmark positions from authored-topology head assets.

Run with Blender (not a plain Python interpreter). Unlike render_head.py this
does not load cameraSetup.blend or do any camera/alignment work -- it only
reads mesh geometry.

Single asset:
    blender --background --factory-startup --python blender/extract_landmarks.py -- \\
        path/to/asset.fbx [--config landmarks/landmark_config.json] [--output-dir DIR]

Batch (many assets in one Blender process -- much faster than one process per asset):
    blender --background --factory-startup --python blender/extract_landmarks.py -- \\
        --manifest manifest.json [--config ...] [--output-dir DIR]

``manifest.json`` is a JSON array of asset paths, or ``{"assets": [{"path": ...}, ...]}``
(the shape tools/select_random_assets.py already writes).

Each asset requires a head mesh object (name contains "head" and "geo", not
"body") whose topology matches landmark_config.json's stored hash -- this is a
hard guard against silently reading the wrong vertex for a different topology.
Writes one JSON per asset: ``{"asset", "mesh_object", "vertex_count", "landmarks"}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import bpy
except ImportError as error:  # pragma: no cover - guidance for misuse
    raise SystemExit(
        "This script requires Blender's Python (bpy). Run it with:\n"
        "  blender --background --factory-startup --python blender/extract_landmarks.py -- <asset> ..."
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "landmarks" / "landmark_config.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "landmarks_raw"

IMPORTABLE_EXTENSIONS = {".fbx", ".gltf", ".glb"}
HEAD_NAME_INCLUDE = ("head",)
HEAD_NAME_EXCLUDE = ("body", "eye")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("asset", type=Path, nargs="?", help="Single asset to process.")
    parser.add_argument("--manifest", type=Path, default=None, help="JSON manifest of many assets to process.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="landmark_config.json path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output JSON.")
    parser.add_argument(
        "--allow-topology-mismatch",
        action="store_true",
        help="Log a warning instead of failing when an asset's topology hash doesn't match.",
    )
    args = parser.parse_args(argv)
    if args.asset is None and args.manifest is None:
        parser.error("provide either an <asset> or --manifest")
    return args


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name, idx in config["points"].items() if idx is None]
    if missing:
        raise ValueError(
            f"{path} is missing vertex indices for: {', '.join(missing)}. "
            "Run blender/pick_landmarks.py to calibrate them first."
        )
    if config.get("topology_hash") is None:
        raise ValueError(f"{path} has no topology_hash yet. Run blender/pick_landmarks.py first.")
    return config


def load_manifest(path: Path) -> list[Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("assets", [])
    paths: list[Path] = []
    for entry in data:
        if isinstance(entry, str):
            paths.append(Path(entry))
        elif isinstance(entry, dict) and "path" in entry:
            paths.append(Path(entry["path"]))
        else:
            raise ValueError(f"Unrecognized manifest entry: {entry!r}")
    return paths


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


def find_head_mesh(imported_objects: list):
    candidates = []
    for obj in imported_objects:
        if obj.type != "MESH":
            continue
        name = obj.name.lower()
        if not any(token in name for token in HEAD_NAME_INCLUDE):
            continue
        if any(token in name for token in HEAD_NAME_EXCLUDE):
            continue
        candidates.append(obj)
    if not candidates:
        names = [obj.name for obj in imported_objects if obj.type == "MESH"]
        raise RuntimeError(f"No head mesh found. Mesh objects present: {names}")
    if len(candidates) > 1:
        print(f"WARNING: multiple head-mesh candidates {[o.name for o in candidates]}; using the first.")
    return candidates[0]


def topology_hash(mesh) -> str:
    digest = hashlib.sha256()
    for polygon in mesh.polygons:
        for index in polygon.vertices:
            digest.update(index.to_bytes(4, "little"))
    return digest.hexdigest()


def extract_landmarks(obj, config: dict) -> dict[str, list[float]]:
    landmarks: dict[str, list[float]] = {}
    for name, index in config["points"].items():
        vertex = obj.data.vertices[index]
        world = obj.matrix_world @ vertex.co
        landmarks[name] = [world.x, world.y, world.z]
    return landmarks


def process_asset(asset: Path, config: dict, output_dir: Path, allow_mismatch: bool) -> bool:
    print(f"Processing {asset.name} ...")
    reset_scene()
    imported = import_asset(asset)
    head = find_head_mesh(imported)

    vertex_count = len(head.data.vertices)
    if vertex_count != config["expected_vertex_count"]:
        message = (
            f"{asset.name}: head mesh '{head.name}' has {vertex_count} vertices, "
            f"expected {config['expected_vertex_count']}."
        )
        if allow_mismatch:
            print(f"WARNING: {message} Skipping (topology mismatch).")
            return False
        raise RuntimeError(message)

    actual_hash = topology_hash(head.data)
    if actual_hash != config["topology_hash"]:
        message = f"{asset.name}: head mesh '{head.name}' topology hash does not match landmark_config.json."
        if allow_mismatch:
            print(f"WARNING: {message} Skipping (topology mismatch).")
            return False
        raise RuntimeError(message)

    landmarks = extract_landmarks(head, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{asset.stem}.json"
    output_path.write_text(
        json.dumps(
            {
                "asset": str(asset),
                "mesh_object": head.name,
                "vertex_count": vertex_count,
                "landmarks": landmarks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  -> {output_path}")
    return True


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())

    assets = load_manifest(args.manifest.resolve()) if args.manifest else [args.asset.resolve()]

    succeeded = 0
    failed = 0
    for asset in assets:
        try:
            if process_asset(asset.resolve(), config, args.output_dir.resolve(), args.allow_topology_mismatch):
                succeeded += 1
            else:
                failed += 1
        except Exception as error:  # noqa: BLE001 - keep batch runs going, report at the end
            print(f"ERROR processing {asset}: {error}")
            failed += 1

    print(f"Done. {succeeded} succeeded, {failed} failed, out of {len(assets)}.")
    if failed and succeeded == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
