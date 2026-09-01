"""Per-joint robust-range anomaly classifier.

Alternative to the HistGradientBoostingClassifier trained in
tools/train_classifier.py: instead of a black-box model, this learns one
"good" range per feature joint (median +/- scaled MAD, computed from good-
labeled rows only) and scores a candidate by how far it strays from those
ranges, combined into a single anomaly score across joints.

A naive "any single joint outside its own range -> bad" rule compounds
false positives across joints (13 independent 95%-coverage bounds already
gives ~49% chance a genuinely good asset trips at least one by chance).
This instead combines every joint's standardized deviation into one RMS
anomaly score, then tunes a single threshold on labeled data -- so one
noisy joint can't flag an asset bad on its own, and the resulting score is
still interpretable per-joint (see ``top_contributor``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

# 1.4826 rescales median-absolute-deviation to be a consistent estimator of
# standard deviation for normally-distributed data.
MAD_TO_STD = 1.4826
MIN_SCALE = 1e-6


class RobustRangeAnomalyClassifier(ClassifierMixin, BaseEstimator):
    """Per-feature robust z-score anomaly detector with a single tuned threshold.

    fit(X, y): y==1 rows ("good") define each feature's median/scale. The
    anomaly score (RMS of per-feature |z|, NaN-safe) is then computed for
    every row in X (good and bad), and a threshold is chosen to maximize
    F1 against y -- so the *shape* of "good" comes only from good rows,
    but the *cutoff* is calibrated against labeled examples of both.

    ``group_cols``, if given, names non-numeric columns of X (e.g.
    ["ethnicity", "gender"]) to condition the good range on: each group gets
    its own median/scale, computed from that group's good rows only, instead
    of one range pooled across every group. A group with fewer than
    ``min_group_size`` good rows falls back to the pooled range instead --
    with as few as 9-12 good examples for some ethnicity/gender combos in
    this dataset, a standalone per-group median/MAD would mostly be noise.
    """

    def __init__(self, score_agg: str = "rms", group_cols: list[str] | None = None, min_group_size: int = 8):
        self.score_agg = score_agg
        self.group_cols = group_cols
        self.min_group_size = min_group_size

    def _split_groups(self, x: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        """(numeric feature columns, group key per row) -- group key is None if group_cols wasn't set."""
        x = pd.DataFrame(x)
        if not self.group_cols:
            return x, None
        feature_cols = [c for c in x.columns if c not in self.group_cols]
        group_key = x[self.group_cols].astype(str).agg("|".join, axis=1)
        return x[feature_cols], group_key

    def fit(self, x: pd.DataFrame, y: pd.Series):
        x = pd.DataFrame(x)
        y = pd.Series(np.asarray(y), index=x.index)
        self.classes_ = np.array([0, 1])  # cross_val_predict/roc_auc need this to pick the "good" column
        features, group_key = self._split_groups(x)
        good = features[y == 1]

        self.medians_ = good.median()
        mad = (good - self.medians_).abs().median()
        self.scale_ = (mad * MAD_TO_STD).clip(lower=MIN_SCALE)

        self.group_medians_: dict[str, pd.Series] = {}
        self.group_scale_: dict[str, pd.Series] = {}
        if group_key is not None:
            good_groups = group_key[y == 1]
            for key, idx in good_groups.groupby(good_groups).groups.items():
                if len(idx) < self.min_group_size:
                    continue  # too few good examples to trust a standalone range -- falls back to pooled
                group_good = good.loc[idx]
                group_median = group_good.median()
                group_mad = (group_good - group_median).abs().median()
                self.group_medians_[key] = group_median
                self.group_scale_[key] = (group_mad * MAD_TO_STD).clip(lower=MIN_SCALE)

        scores = self._score(x)
        self.threshold_ = self._best_threshold(scores, y)
        return self

    def _z(self, x: pd.DataFrame) -> pd.DataFrame:
        # A caller building a single-row frame from a dict (e.g. scoring one
        # asset) gets `object` dtype on any column holding a bare `None` for a
        # missing feature -- pandas can't infer "this column is numeric" from
        # a single None with no other rows to compare against. That breaks
        # np.sqrt() downstream on a mixed-dtype row, so coerce to numeric
        # (turning None/anything unparseable into NaN) before computing z.
        features, group_key = self._split_groups(pd.DataFrame(x))
        features = features.apply(pd.to_numeric, errors="coerce")

        if group_key is None or not self.group_medians_:
            return (features - self.medians_).abs() / self.scale_

        z = pd.DataFrame(index=features.index, columns=features.columns, dtype=float)
        for key, idx in group_key.groupby(group_key).groups.items():
            median = self.group_medians_.get(key, self.medians_)
            scale = self.group_scale_.get(key, self.scale_)
            z.loc[idx] = (features.loc[idx] - median).abs() / scale
        return z

    def _score(self, x: pd.DataFrame) -> np.ndarray:
        z = self._z(pd.DataFrame(x))
        if self.score_agg == "max":
            return z.max(axis=1, skipna=True).to_numpy()
        return np.sqrt((z**2).mean(axis=1, skipna=True)).to_numpy()

    @staticmethod
    def _best_threshold(scores: np.ndarray, y: pd.Series) -> float:
        candidates = np.unique(scores[~np.isnan(scores)])
        if len(candidates) == 0:
            return 0.0
        best_thresh, best_f1 = candidates[0], -1.0
        for thresh in candidates:
            predicted_good = (scores <= thresh).astype(int)
            score = f1_score(y, predicted_good, zero_division=0)
            if score > best_f1:
                best_f1, best_thresh = score, thresh
        return float(best_thresh)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        scores = self._score(x)
        predicted_good = np.where(scores <= self.threshold_, 1, 0)
        predicted_good[np.isnan(scores)] = 1  # no evidence -> don't flag
        return predicted_good

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        scores = self._score(x)
        # Monotonic decreasing transform of an unbounded anomaly score into
        # (0, 1]. Not a calibrated probability -- downstream metrics here
        # (roc_auc, predict-based f1/balanced_accuracy) only need correct
        # ranking and a consistent threshold, not calibration.
        proba_good = 1.0 / (1.0 + np.nan_to_num(scores, nan=0.0))
        return np.column_stack([1 - proba_good, proba_good])

    def top_contributors(self, x: pd.DataFrame) -> pd.Series:
        """Name of the joint with the largest |z| for each row (diagnostic).

        NaN for rows with no usable feature at all (e.g. an asset with no
        chaos_* data) -- pandas' idxmax raises on an all-NaN row even with
        skipna=True, so those rows are handled separately instead of crashing.
        """
        z = self._z(pd.DataFrame(x))
        has_any = z.notna().any(axis=1)
        result = pd.Series(np.nan, index=z.index, dtype=object)
        if has_any.any():
            result.loc[has_any] = z.loc[has_any].idxmax(axis=1, skipna=True)
        return result

    def per_feature_z(self, x: pd.DataFrame) -> pd.DataFrame:
        """Public per-feature |z|-score table, one row per input row."""
        return self._z(pd.DataFrame(x))

    def region_scores(self, x: pd.DataFrame, regions: dict[str, list[str]]) -> pd.DataFrame:
        """RMS of per-feature |z| within each named region (see
        landmarks/ratios.py's regions_to_features()) -- the same aggregation
        predict()/predict_proba() use overall, scoped to one region's features.

        Diagnostic only: this exists to show a reviewer *why* the overall
        score landed where it did, not to replace it. Deriving a bucket
        decision by tallying flagged regions would reintroduce exactly the
        false-positive compounding this class's RMS design exists to avoid
        (see module docstring) -- the single combined predict()/predict_proba()
        score stays the actual verdict.
        """
        z = self._z(pd.DataFrame(x))
        scores = {}
        for region, cols in regions.items():
            present = [c for c in cols if c in z.columns]
            if not present:
                continue
            scores[region] = np.sqrt((z[present] ** 2).mean(axis=1, skipna=True))
        return pd.DataFrame(scores, index=z.index)

    def region_top_contributors(self, x: pd.DataFrame, regions: dict[str, list[str]]) -> pd.DataFrame:
        """Name of the single feature with the largest |z| within each region, per row."""
        z = self._z(pd.DataFrame(x))
        result = {}
        for region, cols in regions.items():
            present = [c for c in cols if c in z.columns]
            if not present:
                continue
            sub = z[present]
            has_any = sub.notna().any(axis=1)
            col = pd.Series(np.nan, index=sub.index, dtype=object)
            if has_any.any():
                col.loc[has_any] = sub.loc[has_any].idxmax(axis=1, skipna=True)
            result[region] = col
        return pd.DataFrame(result, index=z.index)

    def range_table(self) -> pd.DataFrame:
        """Human-readable per-joint good range (median +/- 2*scale).

        One row per feature, or -- if group_cols was set -- one row per
        (group, feature), plus a "(pooled fallback)" group holding the range
        any group too small for its own stats falls back to.
        """
        if not self.group_cols:
            return pd.DataFrame(
                {
                    "median": self.medians_,
                    "scale": self.scale_,
                    "lower_2sigma": self.medians_ - 2 * self.scale_,
                    "upper_2sigma": self.medians_ + 2 * self.scale_,
                }
            )

        rows: list[dict] = []
        for group_name, median, scale in [
            *[(key, self.group_medians_[key], self.group_scale_[key]) for key in sorted(self.group_medians_)],
            ("(pooled fallback)", self.medians_, self.scale_),
        ]:
            for feat in median.index:
                rows.append(
                    {
                        "group": group_name,
                        "feature": feat,
                        "median": median[feat],
                        "scale": scale[feat],
                        "lower_2sigma": median[feat] - 2 * scale[feat],
                        "upper_2sigma": median[feat] + 2 * scale[feat],
                    }
                )
        return pd.DataFrame(rows).set_index(["group", "feature"])
