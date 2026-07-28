#!/usr/bin/env python3
"""Select or verify mesh assets (.fbx / .glb) against a folder tree.

select — recursively find .fbx / .glb files, sample N at random, write JSON.
check  — compare a folder against selected_assets.json and report missing assets.

Examples:
    python tools/select_random_assets.py select --folder path/to/assets --count 50
    python tools/select_random_assets.py check --folder path/to/subset
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ASSET_EXTENSIONS = {".fbx", ".glb"}


def find_assets(root: Path) -> list[Path]:
    """Return all .fbx / .glb files under root, sorted for stable ordering."""
    assets: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ASSET_EXTENSIONS:
            assets.append(path.resolve())
    assets.sort(key=lambda p: str(p).lower())
    return assets


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load asset entries from selected_assets.json."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = data.get("assets")
    if not isinstance(assets, list):
        raise ValueError(f"no 'assets' list in {manifest_path}")
    return assets


def check_missing_assets(
    folder: Path,
    manifest_path: Path,
) -> dict:
    """Compare manifest assets against files present under folder.

    Matching is by filename (case-insensitive), so nested layout differences
    between the original source tree and the checked folder are allowed.
    """
    expected = load_manifest(manifest_path)
    present_names = {path.name.lower() for path in find_assets(folder)}

    found: list[dict] = []
    missing: list[dict] = []
    for entry in expected:
        name = entry.get("name") or Path(entry.get("path", "")).name
        if not name:
            missing.append({**entry, "reason": "no name/path in manifest"})
            continue
        if name.lower() in present_names:
            found.append(entry)
        else:
            missing.append(entry)

    return {
        "folder": str(folder.resolve()),
        "manifest": str(manifest_path.resolve()),
        "expected_count": len(expected),
        "found_count": len(found),
        "missing_count": len(missing),
        "found": found,
        "missing": missing,
    }


def _add_select_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Root folder to search recursively for assets",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of assets to select (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("selected_assets.json"),
        help="Output JSON path (default: selected_assets.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible selection",
    )


def _add_check_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Folder to check for assets listed in the manifest",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("selected_assets.json"),
        help="Manifest JSON to check against (default: selected_assets.json)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the check report JSON",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select or verify .fbx / .glb assets from a folder tree"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser(
        "select", help="Randomly select assets and write selected_assets.json"
    )
    _add_select_args(select_parser)

    check_parser = subparsers.add_parser(
        "check",
        help="Report which manifest assets are missing from a folder",
    )
    _add_check_args(check_parser)

    return parser.parse_args(argv)


def run_select(args: argparse.Namespace) -> int:
    folder = args.folder.resolve()

    if not folder.is_dir():
        print(f"ERROR: folder not found: {folder}", file=sys.stderr)
        return 2
    if args.count < 1:
        print(f"ERROR: --count must be >= 1 (got {args.count})", file=sys.stderr)
        return 2

    assets = find_assets(folder)
    if not assets:
        print(
            f"ERROR: no .fbx or .glb files found under {folder}",
            file=sys.stderr,
        )
        return 1

    n = min(args.count, len(assets))
    if n < args.count:
        print(
            f"WARNING: only {len(assets)} assets found; selecting all of them",
            file=sys.stderr,
        )

    if args.seed is not None:
        random.seed(args.seed)

    selected = sorted(random.sample(assets, n), key=lambda p: str(p).lower())

    payload = {
        "source_folder": str(folder),
        "requested_count": args.count,
        "available_count": len(assets),
        "selected_count": len(selected),
        "seed": args.seed,
        "assets": [
            {
                "path": str(path),
                "name": path.name,
                "stem": path.stem,
                "extension": path.suffix.lower(),
            }
            for path in selected
        ],
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"OK: selected {len(selected)} of {len(assets)} assets -> {output}")
    return 0


def run_check(args: argparse.Namespace) -> int:
    folder = args.folder.resolve()
    manifest = args.manifest.resolve()

    if not folder.is_dir():
        print(f"ERROR: folder not found: {folder}", file=sys.stderr)
        return 2
    if not manifest.is_file():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 2

    try:
        report = check_missing_assets(folder, manifest)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"ERROR: failed to check assets: {error}", file=sys.stderr)
        return 2

    if args.report is not None:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report: {report_path}")

    missing = report["missing"]
    print(
        f"Checked {report['expected_count']} assets: "
        f"{report['found_count']} found, {report['missing_count']} missing"
    )
    if missing:
        print("Missing:")
        for entry in missing:
            print(f"  - {entry.get('name') or entry.get('path')}")
        return 1

    print("OK: all manifest assets are present in the folder")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "select":
        return run_select(args)
    if args.command == "check":
        return run_check(args)
    print(f"ERROR: unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
