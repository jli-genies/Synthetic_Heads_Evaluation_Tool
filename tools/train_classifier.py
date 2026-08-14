#!/usr/bin/env python3
"""Train a good/bad classifier on landmarks/training_dataset.csv.

Evaluation is group-aware: the dataset has only ~18 distinct authored heads
(head_id), and most heads contribute both good and bad rows, so a plain
random split would leak per-identity signal into the test fold and overstate
accuracy. StratifiedGroupKFold keeps every row for a given head_id entirely
in one fold while still balancing the good/bad ratio across folds.

Two models are compared:
  - HistGradientBoostingClassifier: tolerates the NaN chaos_* columns
    natively (variation_narrower_jaw_001 never resolved 4 of the chaos
    binds), so it trains on the raw feature matrix with no imputation.
  - LogisticRegression: an interpretable baseline, run through a pipeline
    that imputes (median) and standardizes first since linear models can't
    handle NaN or unscaled features the way trees can.

Example:
    python tools/train_classifier.py
    python tools/train_classifier.py --n-splits 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "landmarks" / "training_dataset.csv"
DEFAULT_MODEL_OUT = PROJECT_ROOT / "landmarks" / "model_hist_gb.joblib"
DEFAULT_OOF_OUT = PROJECT_ROOT / "landmarks" / "oof_predictions.csv"

META_COLUMNS = ["asset", "parent_asset", "label", "variation_folder", "ethnicity", "gender", "head_id", "frame"]
SCORING = ["f1", "balanced_accuracy", "roc_auc"]


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    x = df[feature_cols]
    y = (df["label"] == "good").astype(int)
    groups = df["head_id"]
    return x, y, groups, df["asset"], feature_cols


def build_models() -> dict[str, object]:
    return {
        "hist_gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
        "logistic_regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
            ]
        ),
    }


def evaluate(model, x: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int) -> dict[str, np.ndarray]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_validate(model, x, y, groups=groups, cv=cv, scoring=SCORING)


def report_feature_importance(model, x: pd.DataFrame, y: pd.Series, top_n: int = 15) -> None:
    model.fit(x, y)
    result = permutation_importance(model, x, y, n_repeats=20, random_state=42, scoring="f1")
    order = np.argsort(result.importances_mean)[::-1][:top_n]
    print(f"\nTop {top_n} features by permutation importance (hist_gradient_boosting, fit on all data):")
    for idx in order:
        print(f"  {x.columns[idx]:<28} {result.importances_mean[idx]:.4f} +/- {result.importances_std[idx]:.4f}")


def out_of_fold_predictions(
    model, x: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int, asset_names: pd.Series
) -> pd.DataFrame:
    """Each row's prediction comes only from folds that never saw its head_id
    group during training -- this is the honest per-asset stand-in for
    'what would the model say about a head it has never seen before'."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    proba_good = cross_val_predict(model, x, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    predicted_label = np.where(proba_good >= 0.5, "good", "bad")
    actual_label = np.where(y == 1, "good", "bad")
    return pd.DataFrame(
        {
            "asset": asset_names,
            "head_id": groups,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "proba_good": proba_good,
            "correct": predicted_label == actual_label,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to training_dataset.csv.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of StratifiedGroupKFold folds.")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT, help="Where to save the final fitted model.")
    parser.add_argument("--oof-out", type=Path, default=DEFAULT_OOF_OUT, help="Where to save per-asset out-of-fold predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y, groups, asset_names, feature_cols = load_dataset(args.csv)
    print(f"Loaded {len(x)} rows, {len(feature_cols)} features, {groups.nunique()} distinct head_id groups.")
    print(f"Label balance: {y.sum()} good / {(1 - y).sum()} bad.\n")

    for name, model in build_models().items():
        scores = evaluate(model, x, y, groups, args.n_splits)
        print(f"[{name}] {args.n_splits}-fold StratifiedGroupKFold (grouped by head_id):")
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            print(f"  {metric:<18} {values.mean():.3f} +/- {values.std():.3f}  {np.round(values, 3)}")
        print()

    hist_gb = build_models()["hist_gradient_boosting"]
    report_feature_importance(hist_gb, x, y)

    oof = out_of_fold_predictions(build_models()["hist_gradient_boosting"], x, y, groups, args.n_splits, asset_names)
    oof.to_csv(args.oof_out, index=False)
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
                f"predicted={row['predicted_label']:<4} proba_good={row['proba_good']:.2f}"
            )

    final_model = build_models()["hist_gradient_boosting"]
    final_model.fit(x, y)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "feature_columns": feature_cols}, args.model_out)
    print(f"\nFinal model (fit on all {len(x)} rows) saved to {args.model_out}")


if __name__ == "__main__":
    main()
