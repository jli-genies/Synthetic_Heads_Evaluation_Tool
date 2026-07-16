"""Render four standard head views (front, side_l, side_r, back) with Blender.

This script must be run through Blender itself, not a plain Python interpreter,
since it depends on the `bpy` module:

    blender --background --factory-startup --python blender/render_head.py -- \\
        path/to/head.fbx [--output-dir DIR] [--resolution 1024] \\
        [--samples 64] [--engine EEVEE|CYCLES] [--margin 1.35]

Output images are written as <output-dir>/<view>.png, defaulting to
<asset_dir>/<asset_stem>_renders/ so `ui.render_panel.RenderPanel` finds them
automatically (it searches for a `<stem>_renders` folder next to the asset).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ImportError as error:  # pragma: no cover - guidance for misuse
    raise SystemExit(
        "This script requires Blender's Python (bpy). Run it with:\n"
        "  blender --background --factory-startup --python blender/render_head.py -- <asset> ..."
    ) from error


IMPORTABLE_EXTENSIONS = {".blend", ".fbx", ".obj", ".gltf", ".glb", ".usd", ".usda", ".usdc"}

# (view name, azimuth in degrees around +Z, measured from the front/-Y axis).
VIEWS = (
    ("front", 0.0),
    ("side_l", 90.0),
    ("side_r", -90.0),
    ("back", 180.0),
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path, help="Path to the head asset to render.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write renders to (default: <asset_dir>/<asset_stem>_renders).",
    )
    parser.add_argument("--resolution", type=int, default=1024, help="Square render resolution in pixels.")
    parser.add_argument("--samples", type=int, default=64, help="Render samples (per view).")
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE", help="Render engine to use.")
    parser.add_argument(
        "--margin",
        type=float,
        default=1.35,
        help="Framing margin multiplier applied to the head's bounding radius.",
    )
    return parser.parse_args(argv)


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_asset(path: Path) -> list:
    """Import `path` into the current scene and return the newly created objects."""
    suffix = path.suffix.lower()
    before = set(bpy.data.objects)

    if suffix == ".blend":
        with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.scene.collection.objects.link(obj)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif suffix in {".gltf", ".glb"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix in {".usd", ".usda", ".usdc"}:
        bpy.ops.wm.usd_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported asset type: {suffix}")

    return [obj for obj in bpy.data.objects if obj not in before]


def compute_bounds(imported_objects: list) -> tuple[Vector, float]:
    """Return the world-space center and bounding-sphere radius of the head mesh."""
    meshes = [obj for obj in imported_objects if obj.type == "MESH"]
    if not meshes:
        meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects were found in the imported asset.")

    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    min_corner = Vector((min(c[axis] for c in corners) for axis in range(3)))
    max_corner = Vector((max(c[axis] for c in corners) for axis in range(3)))
    center = (min_corner + max_corner) / 2
    radius = max((max_corner - min_corner).length / 2, 0.01)
    return center, radius


def setup_lighting() -> None:
    def add_area_light(name: str, location, rotation_deg, energy: float, size: float) -> None:
        light_data = bpy.data.lights.new(name=name, type="AREA")
        light_data.energy = energy
        light_data.size = size
        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        light_obj.location = location
        light_obj.rotation_euler = tuple(math.radians(angle) for angle in rotation_deg)
        bpy.context.scene.collection.objects.link(light_obj)

    add_area_light("KeyLight", (2.5, -2.5, 2.5), (60, 0, 45), 800.0, 2.0)
    add_area_light("FillLight", (-2.5, -2.0, 1.5), (70, 0, -55), 300.0, 2.5)
    add_area_light("RimLight", (0.0, 2.5, 2.0), (110, 0, 180), 400.0, 1.5)

    world = bpy.data.worlds.new("Studio")
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs[0].default_value = (0.86, 0.87, 0.89, 1.0)
        background.inputs[1].default_value = 1.0
    bpy.context.scene.world = world


def setup_camera(radius: float, margin: float) -> tuple:
    cam_data = bpy.data.cameras.new("RenderCam")
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(40)
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    distance = (radius * margin) / math.sin(cam_data.angle / 2)
    return cam_obj, distance


def point_camera(cam_obj, center: Vector, distance: float, azimuth_deg: float) -> None:
    azimuth = math.radians(azimuth_deg)
    # Slightly above eye-level to avoid a dead-on, flat framing.
    offset = Vector((math.sin(azimuth), -math.cos(azimuth), 0.08)) * distance
    cam_obj.location = center + offset
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def configure_render(resolution: int, samples: int, engine: str) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    if engine == "CYCLES":
        bpy.ops.preferences.addon_enable(module="cycles")
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
    else:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = samples


def render_views(cam_obj, center: Vector, distance: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for view_name, azimuth in VIEWS:
        point_camera(cam_obj, center, distance, azimuth)
        bpy.context.scene.render.filepath = str(output_dir / f"{view_name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {view_name} -> {bpy.context.scene.render.filepath}")


def main() -> None:
    args = parse_args()

    if args.asset.suffix.lower() not in IMPORTABLE_EXTENSIONS:
        raise ValueError(f"Unsupported asset type: {args.asset.suffix}")
    if not args.asset.is_file():
        raise FileNotFoundError(f"Asset not found: {args.asset}")

    output_dir = args.output_dir or args.asset.parent / f"{args.asset.stem}_renders"

    reset_scene()
    imported_objects = import_asset(args.asset)
    center, radius = compute_bounds(imported_objects)
    setup_lighting()
    cam_obj, distance = setup_camera(radius, args.margin)
    configure_render(args.resolution, args.samples, args.engine)
    render_views(cam_obj, center, distance, output_dir)


if __name__ == "__main__":
    main()
