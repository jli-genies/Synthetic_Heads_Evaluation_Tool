#!/usr/bin/env python3
"""Train a per-joint robust-range anomaly classifier on landmarks/training_dataset.csv.

Alternative to tools/train_classifier.py's HistGradientBoostingClassifier:
instead of a black-box model, this learns one "good" range per chaos_*
joint (median +/- scaled MAD from good-labeled rows), combines every
joint's standardized deviation into a single RMS anomaly score, and tunes
one threshold on labeled data. See landmarks/range_classifier.py for the
model itself and why a naive "any joint out of range -> bad" rule isn't
used (it compounds false positives across joints).

Uses the same StratifiedGroupKFold-by-head_id evaluation as
tools/train_classifier.py so results are directly comparable.

Example:
    python tools/train_range_classifier.py
    python tools/train_range_classifier.py --score-agg max
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_validate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarks.range_classifier import RobustRangeAnomalyClassifier  # noqa: E402

DEFAULT_CSV = PROJECT_ROOT / "landmarks" / "training_dataset.csv"
DEFAULT_MODEL_OUT = PROJECT_ROOT / "landmarks" / "model_range.joblib"
DEFAULT_OOF_OUT = PROJECT_ROOT / "landmarks" / "oof_predictions_range.csv"

META_COLUMNS = ["asset", "parent_asset", "label", "variation_folder", "ethnicity", "gender", "head_id", "frame"]
SCORING = ["f1", "balanced_accuracy", "roc_auc"]


def load_dataset(csv_path: Path, feature_prefix: str) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    df = pd.read_csv(csv_path)
    if feature_prefix == "all":
        feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    else:
        feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    x = df[feature_cols]
    y = (df["label"] == "good").astype(int)
    groups = df["head_id"]
    return x, y, groups, df["asset"], feature_cols


def evaluate(model, x: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int) -> dict[str, np.ndarray]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_validate(model, x, y, groups=groups, cv=cv, scoring=SCORING)


def out_of_fold_predictions(
    model, x: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int, asset_names: pd.Series
) -> pd.DataFrame:
    """Each row's prediction/top-contributor comes only from a model whose
    good-range was built without ever seeing its head_id group."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    proba_good = cross_val_predict(model, x, y, groups=groups, cv=cv, method="predict_proba")[:, 1]

    top_contributor = pd.Series(index=x.index, dtype=object)
    for train_idx, test_idx in cv.split(x, y, groups):
        fold_model = RobustRangeAnomalyClassifier(score_agg=model.score_agg)
        fold_model.fit(x.iloc[train_idx], y.iloc[train_idx])
        top_contributor.iloc[test_idx] = fold_model.top_contributors(x.iloc[test_idx]).values

    predicted_label = np.where(proba_good >= 0.5, "good", "bad")
    actual_label = np.where(y == 1, "good", "bad")
    return pd.DataFrame(
        {
            "asset": asset_names,
            "head_id": groups,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "proba_good": proba_good,
            "top_contributor": top_contributor.values,
            "correct": predicted_label == actual_label,
        }
    )


def report_populated_vs_pooled(oof: pd.DataFrame, x: pd.DataFrame) -> None:
    """Split OOF metrics by whether a row has any chaos_* data at all.

    Rows with every chaos_* feature missing (e.g. authored-head assets with no
    HeadGen transform recorded) are trivially easy -- there's no discriminating
    signal, so the "no evidence" fallback in RobustRangeAnomalyClassifier.predict
    gets them right by construction, not by learning anything. Pooling them in
    with real, populated-feature rows inflates balanced_accuracy/roc_auc without
    reflecting any actual improvement at telling good variants from bad ones.
    """
    chaos_cols = [c for c in x.columns if c.startswith("chaos_")]
    if not chaos_cols:
        return
    has_signal = (~x[chaos_cols].isna().all(axis=1)).reset_index(drop=True)
    oof = oof.reset_index(drop=True)

    print("Populated-only vs. pooled (see report_populated_vs_pooled docstring for why this split matters):")
    for label, mask in [("no chaos_* data at all", ~has_signal), ("has chaos_* data (the real test)", has_signal)]:
        subset = oof[mask.values]
        if len(subset) == 0:
            continue
        y_true = (subset["actual_label"] == "good").astype(int)
        y_pred = (subset["predicted_label"] == "good").astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, subset["proba_good"]) if y_true.nunique() > 1 else float("nan")
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"  {label:<32} n={len(subset):<4} balanced_accuracy={bal_acc:.3f}  roc_auc={auc:.3f}  f1={f1:.3f}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to training_dataset.csv.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of StratifiedGroupKFold folds.")
    parser.add_argument(
        "--feature-prefix",
        default="chaos_",
        help="Column prefix to build ranges over ('chaos_', 'dist_', ... or 'all' for every feature column).",
    )
    parser.add_argument("--score-agg", choices=["rms", "max"], default="rms", help="How to combine per-joint z-scores.")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT, help="Where to save the final fitted model.")
    parser.add_argument("--oof-out", type=Path, default=DEFAULT_OOF_OUT, help="Where to save per-asset out-of-fold predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y, groups, asset_names, feature_cols = load_dataset(args.csv, args.feature_prefix)
    print(f"Loaded {len(x)} rows, {len(feature_cols)} features ({args.feature_prefix}), {groups.nunique()} distinct head_id groups.")
    print(f"Label balance: {y.sum()} good / {(1 - y).sum()} bad.\n")

    model = RobustRangeAnomalyClassifier(score_agg=args.score_agg)
    scores = evaluate(model, x, y, groups, args.n_splits)
    print(f"[range_anomaly:{args.score_agg}] {args.n_splits}-fold StratifiedGroupKFold (grouped by head_id):")
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        print(f"  {metric:<18} {values.mean():.3f} +/- {values.std():.3f}  {np.round(values, 3)}")

    final_model = RobustRangeAnomalyClassifier(score_agg=args.score_agg)
    final_model.fit(x, y)
    print(f"\nPer-joint good ranges (fit on all {len(x)} rows), threshold={final_model.threshold_:.3f}:")
    print(final_model.range_table().round(3))

    oof = out_of_fold_predictions(model, x, y, groups, args.n_splits, asset_names)
    oof.to_csv(args.oof_out, index=False)
    report_populated_vs_pooled(oof, x)
    wrong = oof[~oof["correct"]].assign(confidence=lambda d: (d["proba_good"] - 0.5).abs()).sort_values(
        "confidence", ascending=False
    )
    print(f"\n{len(wrong)}/{len(oof)} assets misclassified out-of-fold (never-seen-this-head predictions).")
    print(f"Full per-asset predictions written to {args.oof_out}")
    if len(wrong):
        print("\nMost confidently wrong (biggest surprises):")
        for _, row in wrong.head(10).iterrows():
            print(
                f"  {row['asset']:<55} actual={row['actual_label']:<4} "
                f"predicted={row['predicted_label']:<4} proba_good={row['proba_good']:.2f} "
                f"top_contributor={row['top_contributor']}"
            )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "feature_columns": feature_cols}, args.model_out)
    print(f"\nFinal model (fit on all {len(x)} rows) saved to {args.model_out}")


if __name__ == "__main__":
    main()
