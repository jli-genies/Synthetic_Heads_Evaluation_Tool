"""Render multiview stills using ``blender/cameraSetup.blend`` as the scene template.

Run with Blender (5.1+ for this .blend), not a plain Python interpreter.

Import + auto-align an asset (headless):

    "C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe" --background \\
        blender/cameraSetup.blend --python blender/render_head.py -- \\
        path/to/asset.fbx [--align center|feet|origin] [--output-dir DIR] \\
        [--views front side_r bottom] [--resolution 1024] [--samples 64] \\
        [--engine EEVEE|CYCLES|SCENE]

Render whatever is already in the open .blend (you place the head yourself).
Omit the asset path and pass ``--output-dir``. Do **not** pass the template as
``--background blender/cameraSetup.blend`` unless that file already contains your
subject — start Blender on your saved scene instead:

    blender path/to/my_posed_scene.blend --background --python blender/render_head.py -- \\
        --output-dir path/to/out [--views front side_r bottom]

Or from an open Blender UI (Scripting workspace / Text Editor), run the script
with the same args after ``--``.

The template provides a shared ``Empty`` look-at / placement pivot with cameras
parented to it. Full-body assets snap to that Empty; cut heads auto-align to the
orbit-camera height so they sit in the horizontal framing band.

Camera roles are inferred from world position around the Empty (the .blend uses
generic names like Camera, Camera.001, ...):

    front, side_l, side_r, back,
    front_left, front_right, back_left, back_right, top, bottom

``Camera.009`` is mapped to ``bottom`` by name (underside / chin view).

With an asset path, primary outputs land in the repo cache:

    <project_root>/renders/<asset_stem>/front.png
    ...

Pass ``--views`` to include more angles, or ``--output-dir`` to override the cache path.
In scene-only mode ``--output-dir`` is required.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from typing import Any

# #region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parent.parent / "debug-c44263.log"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "c44263",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass


# #endregion

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError as error:  # pragma: no cover - guidance for misuse
    raise SystemExit(
        "This script requires Blender's Python (bpy). Run it with:\n"
        "  blender --background blender/cameraSetup.blend "
        "--python blender/render_head.py -- <asset> ...\n"
        "Or scene-only (current .blend, no import):\n"
        "  blender my_scene.blend --background --python blender/render_head.py -- "
        "--output-dir DIR"
    ) from error


IMPORTABLE_EXTENSIONS = {".blend", ".fbx", ".obj", ".gltf", ".glb", ".usd", ".usda", ".usdc"}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SCENE = SCRIPT_DIR / "cameraSetup.blend"
SUBJECT_NAMES = ("empty", "subject", "assetroot", "asset_root", "asset_pivot", "origin_marker")

# All views the cameraSetup.blend rig can provide.
ALL_VIEWS = (
    "front",
    "side_l",
    "side_r",
    "back",
    "front_left",
    "front_right",
    "back_left",
    "back_right",
    "top",
    "bottom",
)

# Default tagging preview angles (front, profile, underside).
DEFAULT_VIEWS = ("front", "side_r", "bottom")

# Shared framing for all multiview cameras (matches front / side_r / bottom).
TARGET_ORTHO_SCALE = 0.4

# The underside/chin camera is taken directly by name (as authored in
# cameraSetup.blend) rather than inferred from position.
BOTTOM_CAMERA_NAME = "Camera.009"

# Horizontal angle bins in degrees (atan2(x, -y): 0 = front/-Y, +90 = side_l/+X).
ANGLE_VIEWS = (
    (0.0, "front"),
    (45.0, "front_right"),
    (90.0, "side_l"),
    (135.0, "back_right"),
    (180.0, "back"),
    (-45.0, "front_left"),
    (-90.0, "side_r"),
    (-135.0, "back_left"),
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "asset",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Path to the head or full-body asset to import. Omit to render the "
            "already-open scene (requires --output-dir)."
        ),
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help=(
            f"Multiview .blend template loaded before import (default: {DEFAULT_SCENE.name}). "
            "Ignored in scene-only mode (no asset)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for PNGs. Default with an asset: "
            "<project_root>/renders/<asset_stem>. Required when no asset is given."
        ),
    )
    parser.add_argument(
        "--align",
        choices=["center", "head", "feet", "origin"],
        default="center",
        help=(
            "Snap imported asset into the camera rig: center (auto head/body), head "
            "(camera-ring height), feet, or object origin. Ignored in scene-only mode."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Optional square resolution override. Omit to keep cameraSetup.blend settings.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Optional sample override. Omit to keep cameraSetup.blend settings.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=ALL_VIEWS,
        default=list(DEFAULT_VIEWS),
        help=f"Which mapped cameras to render (default: {' '.join(DEFAULT_VIEWS)}).",
    )
    parser.add_argument(
        "--engine",
        choices=["EEVEE", "CYCLES", "SCENE"],
        default="SCENE",
        help="Render engine override, or SCENE to keep the template setting.",
    )
    args = parser.parse_args(argv)
    if args.asset is None and args.output_dir is None:
        parser.error("scene-only mode (no asset) requires --output-dir")
    return args


def load_scene(scene_path: Path) -> None:
    if not scene_path.is_file():
        raise FileNotFoundError(
            f"Multiview scene not found: {scene_path}\n"
            "Expected blender/cameraSetup.blend (or pass --scene)."
        )
    # Always reload so CLI invocation is deterministic even if Blender was
    # started with a different file.
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))


def clear_placeholders() -> None:
    """Remove optional stand-in geometry without touching the camera rig."""
    removed = 0
    for collection in list(bpy.data.collections):
        if collection.name.lower() == "placeholder":
            for obj in list(collection.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
    for obj in list(bpy.data.objects):
        if obj.name.lower().startswith("placeholder"):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    if removed:
        print(f"Removed {removed} placeholder object(s).")


def import_asset(path: Path) -> list:
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

    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        raise RuntimeError(f"No objects were imported from {path}")
    return imported


def find_subject_marker():
    """Prefer the camera-rig Empty from cameraSetup.blend."""
    # Exact name used by the authored template.
    if "Empty" in bpy.data.objects and bpy.data.objects["Empty"].type == "EMPTY":
        return bpy.data.objects["Empty"]

    for obj in bpy.data.objects:
        if obj.type != "EMPTY":
            continue
        normalized = obj.name.lower().replace(" ", "_")
        if normalized in SUBJECT_NAMES:
            return obj

    # Fall back to the parent of the camera cluster, if shared.
    camera_parents = {
        cam.parent for cam in bpy.data.objects if cam.type == "CAMERA" and cam.parent is not None
    }
    if len(camera_parents) == 1:
        return next(iter(camera_parents))
    return None


def compute_bounds(objects: list) -> tuple[Vector, Vector]:
    meshes = [obj for obj in objects if obj.type == "MESH"]
    if not meshes:
        meshes = [obj for obj in objects if obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}]
    if not meshes:
        locations = [obj.matrix_world.translation for obj in objects]
        if not locations:
            raise RuntimeError("No objects available to compute bounds.")
        point = Vector(locations[0])
        return Vector(point), Vector(point)

    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    min_corner = Vector((min(c[axis] for c in corners) for axis in range(3)))
    max_corner = Vector((max(c[axis] for c in corners) for axis in range(3)))
    return min_corner, max_corner


def _orbit_camera_height(marker) -> float:
    """Mean world Z of orbit cameras (excludes top/bottom cameras).

    cameraSetup.blend cameras keep a near-horizontal look direction, so the
    visible band is around camera Z (~1.25), not Empty Z (~0.625).
    """
    marker_z = marker.matrix_world.translation.z if marker else 0.0
    marker_xy = (
        Vector((marker.matrix_world.translation.x, marker.matrix_world.translation.y, 0.0))
        if marker
        else Vector((0.0, 0.0, 0.0))
    )
    heights: list[float] = []
    for cam in bpy.data.objects:
        if cam.type != "CAMERA":
            continue
        if cam.name == BOTTOM_CAMERA_NAME:
            continue
        loc = cam.matrix_world.translation
        horizontal = Vector((loc.x, loc.y, 0.0)) - marker_xy
        # Top camera sits nearly above the marker; skip it for head-height framing.
        if horizontal.length < 1e-4 and loc.z > marker_z + 0.2:
            continue
        heights.append(loc.z)
    if heights:
        return sum(heights) / len(heights)
    return marker_z + 0.625


# Assets shorter than this (meters) are treated as cut heads for framing.
_HEAD_HEIGHT_THRESHOLD = 0.75

# For full-body assets, imported meshes whose name contains one of these tokens
# are torso/limb geometry that would occlude the chin from the bottom camera
# (which looks straight up through the asset's vertical centerline). They are
# hidden for just the "bottom" render so that view isolates the head, matching
# Genies' "<style>_body_<idx>_geo" / "<style>_head_<idx>_geo" naming.
BODY_MESH_TOKENS = ("body",)


def align_asset(imported_objects: list, align: str) -> None:
    marker = find_subject_marker()
    empty_pos = (
        Vector(marker.matrix_world.translation) if marker else Vector((0.0, 0.0, 0.0))
    )
    camera_z = _orbit_camera_height(marker)
    target = Vector(empty_pos)

    min_corner, max_corner = compute_bounds(imported_objects)
    center = (min_corner + max_corner) / 2
    size = max_corner - min_corner
    is_head_like = size.z < _HEAD_HEIGHT_THRESHOLD

    # Default "center": full-body snaps to Empty; cut heads snap to camera-ring
    # height so they land in the horizontal frustum.
    effective_align = align
    if align == "center" and is_head_like:
        effective_align = "head"
    if effective_align == "head":
        target = Vector((empty_pos.x, empty_pos.y, camera_z))

    if marker:
        print(
            f"Aligning asset to '{marker.name}' with mode '{effective_align}' "
            f"(requested '{align}', height={size.z:.3f}m, camera_z={camera_z:.3f})."
        )
    else:
        print(f"No Empty/Subject marker found; aligning to world ({effective_align}).")

    # #region agent log
    mesh_info = []
    for obj in imported_objects:
        dims = list(obj.dimensions) if hasattr(obj, "dimensions") else None
        mesh_info.append(
            {
                "name": obj.name,
                "type": obj.type,
                "hide_render": bool(getattr(obj, "hide_render", False)),
                "visible_get": bool(obj.visible_get()) if hasattr(obj, "visible_get") else None,
                "users_collection": [c.name for c in getattr(obj, "users_collection", [])],
                "parent": obj.parent.name if obj.parent else None,
                "loc": list(obj.matrix_world.translation),
                "scale": list(obj.scale),
                "dimensions": dims,
                "num_materials": len(obj.data.materials)
                if getattr(obj, "data", None) and hasattr(obj.data, "materials")
                else 0,
            }
        )
    _agent_log(
        "L",
        "render_head.py:align_asset:pre",
        "alignment target selection",
        {
            "align": align,
            "effective_align": effective_align,
            "is_head_like": is_head_like,
            "height": size.z,
            "camera_z": camera_z,
            "empty": list(empty_pos),
            "target": list(target),
            "marker": marker.name if marker else None,
            "count": len(imported_objects),
            "objects": mesh_info[:40],
        },
    )
    # #endregion

    if effective_align == "origin":
        origins = [obj.matrix_world.translation for obj in imported_objects]
        reference = sum(origins, Vector((0.0, 0.0, 0.0))) / max(len(origins), 1)
    elif effective_align == "feet":
        reference = Vector((center.x, center.y, min_corner.z))
    else:
        # "center" and "head" both translate the bbox center onto target.
        reference = center

    offset = target - reference
    translation = Matrix.Translation(offset)
    for obj in imported_objects:
        if obj.parent is None or obj.parent not in imported_objects:
            obj.matrix_world = translation @ obj.matrix_world
    bpy.context.view_layer.update()

    # #region agent log
    post_min, post_max = compute_bounds(imported_objects)
    post_size = post_max - post_min
    _agent_log(
        "L",
        "render_head.py:align_asset:post",
        "bounds after align",
        {
            "runId": "post-fix",
            "pre_min": list(min_corner),
            "pre_max": list(max_corner),
            "pre_center": list(center),
            "reference": list(reference),
            "offset": list(offset),
            "post_min": list(post_min),
            "post_max": list(post_max),
            "post_size": list(post_size),
            "post_center": list((post_min + post_max) / 2),
            "effective_align": effective_align,
            "target": list(target),
        },
    )
    # #endregion


def _nearest_angle_view(angle_deg: float) -> str:
    best_view = "front"
    best_delta = 180.0
    for target, view in ANGLE_VIEWS:
        delta = abs((angle_deg - target + 180) % 360 - 180)
        if delta < best_delta:
            best_delta = delta
            best_view = view
    return best_view


def discover_cameras() -> dict[str, Any]:
    """Map semantic view names to cameras: bottom by exact name, others by Empty-relative pose."""
    cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]
    if not cameras:
        raise RuntimeError("No cameras found in cameraSetup.blend.")

    marker = find_subject_marker()
    origin = marker.matrix_world.translation if marker else Vector((0.0, 0.0, 0.0))
    assigned: dict[str, Any] = {}

    for cam in cameras:
        rel = cam.matrix_world.translation - origin
        horizontal = Vector((rel.x, rel.y, 0.0))
        if cam.name == BOTTOM_CAMERA_NAME:
            view = "bottom"
        elif horizontal.length < 1e-4 and rel.z > 1e-4:
            # Top camera in cameraSetup.blend sits directly above the Empty.
            view = "top"
        else:
            angle = math.degrees(math.atan2(rel.x, -rel.y))
            view = _nearest_angle_view(angle)

        if view in assigned:
            print(
                f"Warning: both '{assigned[view].name}' and '{cam.name}' map to '{view}'; "
                "keeping the first."
            )
            continue
        assigned[view] = cam
        print(
            f"Mapped '{cam.name}' -> {view} "
            f"(offset=({rel.x:.3f}, {rel.y:.3f}, {rel.z:.3f}))"
        )

    missing = [view for view in ("front", "side_r") if view not in assigned]
    if missing:
        raise RuntimeError(f"Could not map required cameras for views: {missing}")
    return assigned


def normalize_camera_ortho_scales(
    cameras: dict[str, Any],
    ortho_scale: float = TARGET_ORTHO_SCALE,
) -> None:
    """Force every mapped camera to the same orthographic scale.

    cameraSetup.blend historically mixed 0.4 (front/side/bottom) with 0.7 on the
    other angles. Re-apply on every render so a later Save of the working .blend
    cannot silently revive the mismatch.
    """
    for view, cam in cameras.items():
        data = cam.data
        data.type = "ORTHO"
        old = float(data.ortho_scale)
        if abs(old - ortho_scale) < 1e-6:
            continue
        data.ortho_scale = ortho_scale
        print(f"Set ortho_scale on '{cam.name}' ({view}): {old:.3f} -> {ortho_scale:.3f}")


def normalize_camera_object_scales(cameras: dict[str, Any]) -> None:
    """Reset non-unit camera object scales that warp ortho frustums / clipping.

    Scaling a camera object (common after viewport grab/snaps) does not change
    ``ortho_scale`` in the UI but still distorts the view matrix — often read as
    mysterious occlusion or hollow clipped shells.

    Location, rotation, and Camera data (ortho_scale, shift, clip, type) are left
    as authored in cameraSetup.blend.
    """
    for view, cam in cameras.items():
        scale = cam.matrix_world.to_scale()
        if (
            abs(scale.x - 1.0) < 1e-5
            and abs(scale.y - 1.0) < 1e-5
            and abs(scale.z - 1.0) < 1e-5
        ):
            continue
        loc, rot, _ = cam.matrix_world.decompose()
        # Bake unit scale into the world matrix while keeping parent + local intent.
        if cam.parent is not None:
            parent_inv = cam.parent.matrix_world.inverted()
            local = parent_inv @ Matrix.LocRotScale(loc, rot, Vector((1.0, 1.0, 1.0)))
            cam.location = local.to_translation()
            cam.rotation_euler = local.to_euler(cam.rotation_mode)
            cam.scale = (1.0, 1.0, 1.0)
        else:
            cam.matrix_world = Matrix.LocRotScale(loc, rot, Vector((1.0, 1.0, 1.0)))
        print(
            f"Normalized scale on '{cam.name}' ({view}): "
            f"was ({scale.x:.4f}, {scale.y:.4f}, {scale.z:.4f})"
        )
    bpy.context.view_layer.update()


def log_camera_settings(cameras: dict[str, Any], views: list[str]) -> None:
    for view in views:
        cam = cameras.get(view)
        if cam is None:
            continue
        data = cam.data
        scale = cam.matrix_world.to_scale()
        print(
            f"Camera '{cam.name}' ({view}): type={data.type} "
            f"ortho_scale={getattr(data, 'ortho_scale', None)} "
            f"shift=({data.shift_x:.4f}, {data.shift_y:.4f}) "
            f"clip=({data.clip_start:.4f}, {data.clip_end:.1f}) "
            f"loc={[round(v, 4) for v in cam.matrix_world.translation]} "
            f"scale={[round(v, 4) for v in scale]} "
            f"parent={cam.parent.name if cam.parent else None}"
        )


def configure_render(resolution: int | None, samples: int | None, engine: str) -> None:
    """Apply optional CLI overrides; otherwise keep cameraSetup.blend render settings."""
    scene = bpy.context.scene
    # Keep template image settings (RGBA/transparent/etc.); only ensure PNG stills.
    scene.render.image_settings.file_format = "PNG"

    if resolution is not None:
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        scene.render.resolution_percentage = 100

    if engine == "CYCLES":
        bpy.ops.preferences.addon_enable(module="cycles")
        scene.render.engine = "CYCLES"
    elif engine == "EEVEE":
        engine_items = {
            item.identifier
            for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        }
        scene.render.engine = (
            "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engine_items else "BLENDER_EEVEE"
        )
    # engine == "SCENE": leave template engine (currently CYCLES in cameraSetup.blend).

    if samples is not None:
        if scene.render.engine == "CYCLES":
            scene.cycles.samples = samples
        elif hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = samples

    cycles_samples = getattr(getattr(scene, "cycles", None), "samples", None)
    eevee_samples = getattr(getattr(scene, "eevee", None), "taa_render_samples", None)
    print(
        "Render settings: "
        f"engine={scene.render.engine} "
        f"res={scene.render.resolution_x}x{scene.render.resolution_y} "
        f"film_transparent={bool(getattr(scene.render, 'film_transparent', False))} "
        f"cycles.samples={cycles_samples} eevee.taa={eevee_samples} "
        f"(overrides: resolution={resolution}, samples={samples}, engine={engine})"
    )


def _hide_body_meshes_for_bottom(imported_objects: list) -> list:
    """Hide torso/limb meshes (by name) so the bottom camera isn't blocked by
    the body on full-body assets. Returns the objects hidden, to restore after."""
    hidden = []
    for obj in imported_objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        name = obj.name.lower()
        if any(token in name for token in BODY_MESH_TOKENS):
            obj.hide_render = True
            hidden.append(obj)
    if hidden:
        print(f"Hid {len(hidden)} body mesh(es) for bottom view: {[o.name for o in hidden]}")
    return hidden


def _restore_hidden_meshes(objects: list) -> None:
    for obj in objects:
        obj.hide_render = False


def render_views(
    cameras: dict[str, Any],
    output_dir: Path,
    views: list[str],
    imported_objects: list | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    # #region agent log
    _agent_log(
        "H",
        "render_head.py:render_views",
        "render settings",
        {
            "engine": scene.render.engine,
            "film_transparent": bool(getattr(scene.render, "film_transparent", False)),
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "filepath_template": scene.render.filepath,
            "views": views,
            "camera_locs": {
                name: list(cam.matrix_world.translation) for name, cam in cameras.items()
            },
        },
    )
    # #endregion
    for view in views:
        cam = cameras.get(view)
        if cam is None:
            raise RuntimeError(f"No camera mapped for requested view '{view}'.")
        hidden = []
        if view == "bottom" and imported_objects:
            hidden = _hide_body_meshes_for_bottom(imported_objects)
        scene.camera = cam
        scene.render.filepath = str(output_dir / f"{view}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {view} -> {scene.render.filepath}")
        if hidden:
            _restore_hidden_meshes(hidden)


def main() -> None:
    args = parse_args()
    views = list(dict.fromkeys(args.views))  # preserve order, drop duplicates
    scene_only = args.asset is None

    if scene_only:
        # Keep the currently open .blend (user-positioned subject + camera rig).
        output_dir = args.output_dir.resolve()
        imported: list | None = None
        # #region agent log
        _agent_log(
            "F",
            "render_head.py:main",
            "starting scene-only render",
            {"output_dir": str(output_dir), "views": views},
        )
        # #endregion
        print(f"Scene-only mode: rendering current file -> {output_dir}")
    else:
        asset = args.asset.resolve()
        scene_path = args.scene.resolve()

        if asset.suffix.lower() not in IMPORTABLE_EXTENSIONS:
            raise ValueError(f"Unsupported asset type: {asset.suffix}")
        if not asset.is_file():
            raise FileNotFoundError(f"Asset not found: {asset}")
        if asset == scene_path:
            raise ValueError("Asset path must differ from the --scene template.")

        output_dir = (args.output_dir or (PROJECT_ROOT / "renders" / asset.stem)).resolve()

        # #region agent log
        _agent_log(
            "F",
            "render_head.py:main",
            "starting render",
            {
                "asset": str(asset),
                "suffix": asset.suffix.lower(),
                "align": args.align,
                "output_dir": str(output_dir),
                "views": views,
            },
        )
        # #endregion
        load_scene(scene_path)
        clear_placeholders()
        imported = import_asset(asset)
        align_asset(imported, args.align)

    cameras = discover_cameras()
    # Keep authored camera transforms from cameraSetup.blend, but always unify
    # ortho framing so all angles match front / side / bottom.
    normalize_camera_ortho_scales(cameras)
    # Only repair non-unit object scale (a common Blender gotcha that warps ortho
    # without changing the Camera data block the user edited).
    # normalize_camera_object_scales(cameras)
    configure_render(args.resolution, args.samples, args.engine)
    log_camera_settings(cameras, views)
    render_views(cameras, output_dir, views, imported)
    print(f"Done. Wrote renders to {output_dir}")


if __name__ == "__main__":
    main()
