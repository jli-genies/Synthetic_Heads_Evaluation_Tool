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
    """

    def __init__(self, score_agg: str = "rms"):
        self.score_agg = score_agg

    def fit(self, x: pd.DataFrame, y: pd.Series):
        x = pd.DataFrame(x)
        y = pd.Series(np.asarray(y), index=x.index)
        self.classes_ = np.array([0, 1])  # cross_val_predict/roc_auc need this to pick the "good" column
        good = x[y == 1]

        self.medians_ = good.median()
        mad = (good - self.medians_).abs().median()
        self.scale_ = (mad * MAD_TO_STD).clip(lower=MIN_SCALE)

        scores = self._score(x)
        self.threshold_ = self._best_threshold(scores, y)
        return self

    def _z(self, x: pd.DataFrame) -> pd.DataFrame:
        return (x - self.medians_).abs() / self.scale_

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
        """Name of the joint with the largest |z| for each row (diagnostic)."""
        z = self._z(pd.DataFrame(x))
        return z.idxmax(axis=1, skipna=True)

    def per_feature_z(self, x: pd.DataFrame) -> pd.DataFrame:
        """Public per-feature |z|-score table, one row per input row."""
        return self._z(pd.DataFrame(x))

    def range_table(self) -> pd.DataFrame:
        """Human-readable per-joint good range (median +/- 2*scale)."""
        return pd.DataFrame(
            {
                "median": self.medians_,
                "scale": self.scale_,
                "lower_2sigma": self.medians_ - 2 * self.scale_,
                "upper_2sigma": self.medians_ + 2 * self.scale_,
            }
        )
