#!/usr/bin/env python3
"""Merge dist_/ratio_ rows from every dataset source into one CSV.

landmarks/training_dataset.csv (joint/HeadGen variants) and landmarks/pca_dataset.csv
(PCA-generated heads) have different feature families available -- chaos_/
absdiff_/pctdiff_ only exist for the joint-based source, since those need a
bind or a true per-identity parent that PCA heads don't have. dist_/ratio_
(computed purely from each asset's own landmarks) are the only features every
source can produce, so they're the only ones carried into the merged file --
this is what feeds landmarks/range_classifier.py's group-conditioned ranges.

pca_dataset.csv has no `label` column yet (nothing's been reviewed as bad),
so its label is passed in explicitly rather than assumed.

Example:
    python tools/merge_ratio_datasets.py --pca-label good
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS = [
    str(PROJECT_ROOT / "landmarks" / "training_dataset.csv"),
    f"{PROJECT_ROOT / 'landmarks' / 'pca_dataset.csv'}=good",
    f"{PROJECT_ROOT / 'landmarks' / 'pca_dataset_bad.csv'}=bad",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "landmarks" / "combined_ratio_dataset.csv"

KEEP_META = ["asset", "label", "source", "ethnicity", "gender", "head_id", "frame"]


def load_source(csv_path: Path, default_label: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "source" not in df.columns:
        df["source"] = csv_path.stem
    if "label" not in df.columns:
        if default_label is None:
            raise ValueError(f"{csv_path} has no 'label' column and no default label was given (use PATH=LABEL).")
        df["label"] = default_label
    feature_cols = [c for c in df.columns if c.startswith("dist_") or c.startswith("ratio_")]
    return df[KEEP_META + feature_cols]


def parse_dataset_spec(spec: str) -> tuple[Path, str | None]:
    """'path/to.csv' or 'path/to.csv=label' -- '=' can't appear in a bare path so this is unambiguous."""
    if "=" in spec:
        path_str, label = spec.rsplit("=", 1)
        return Path(path_str), label
    return Path(spec), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="CSV[=LABEL]",
        help=(
            "A dataset CSV to merge, repeatable. Add '=LABEL' if the CSV has no 'label' column of its own "
            "(e.g. a PCA batch whose good/bad split is which folder it came from, not a per-row column). "
            f"Defaults to: {', '.join(DEFAULT_DATASETS)}"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = args.dataset or DEFAULT_DATASETS
    frames = [load_source(*parse_dataset_spec(spec)) for spec in specs]

    shared_features = sorted(set.intersection(*(set(c for c in f.columns if c.startswith(("dist_", "ratio_"))) for f in frames)))
    all_features = set.union(*(set(c for c in f.columns if c.startswith(("dist_", "ratio_"))) for f in frames))
    dropped = all_features - set(shared_features)
    if dropped:
        print(f"Dropping source-specific features not shared across every dataset: {sorted(dropped)}")

    combined = pd.concat([f[KEEP_META + shared_features] for f in frames], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    print(f"\nWrote {len(combined)} rows ({len(shared_features)} dist_/ratio_ features) -> {args.output}")
    print("\nBy source/label:")
    print(combined.groupby(["source", "label"]).size())
    print("\nBy ethnicity/gender/label:")
    print(combined.groupby(["ethnicity", "gender", "label"]).size())


if __name__ == "__main__":
    main()
