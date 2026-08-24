#!/usr/bin/env python3
"""Rebuild landmarks/training_dataset.csv from lists/good.json + bad.json,
then retrain both models on it.

Chains, in order:
  1. tools/build_training_dataset.py  (needs Blender; re-extracts landmarks
     for every labeled entry -- the slow step, usually several minutes)
  2. tools/train_classifier.py        (gradient-boosted classifier)
  3. tools/train_range_classifier.py  (range-anomaly classifier)

Nothing watches lists/good.json / lists/bad.json automatically -- run this
by hand whenever you've added or relabeled entries there and want both
models to reflect it. Stops at the first step that fails so you don't train
on a stale or partial rebuild.

Example:
    python tools/retrain_all.py
    python tools/retrain_all.py --skip-rebuild   # after a model-code change, not a labeling change
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    ("Rebuild training_dataset.csv from lists/good.json + bad.json", PROJECT_ROOT / "tools" / "build_training_dataset.py"),
    ("Train gradient-boosted classifier", PROJECT_ROOT / "tools" / "train_classifier.py"),
    ("Train range-anomaly classifier", PROJECT_ROOT / "tools" / "train_range_classifier.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="Skip re-extracting landmarks and retrain directly on the existing "
        "landmarks/training_dataset.csv -- use this after a model-code change, not a labeling change.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = STEPS[1:] if args.skip_rebuild else STEPS

    for description, script in steps:
        print(f"\n{'=' * 70}\n{description}\n{'=' * 70}", flush=True)
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f"\n'{script.name}' failed (exit code {result.returncode}) -- stopping.", file=sys.stderr)
            sys.exit(result.returncode)

    print(f"\n{'=' * 70}\nDone -- both models retrained on the current lists/good.json + bad.json.\n{'=' * 70}")


if __name__ == "__main__":
    main()
