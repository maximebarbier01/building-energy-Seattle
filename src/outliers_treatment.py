# src/outliers_treatment.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew


class OutlierLogCapper(BaseEstimator, TransformerMixin):
    """
    Compatible numpy.ndarray (sortie de ColumnTransformer).
    Applique:
      - log1p sur colonnes positives très skewed (selon seuil)
      - winsorisation IQR (clip) sur toutes les colonnes
    """

    def __init__(self, log_skew_threshold=1.0, cap_factor=3.0, eps=1e-9):
        self.log_skew_threshold = log_skew_threshold
        self.cap_factor = cap_factor
        self.eps = eps

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        # stats par colonne (ignore NaN)
        col_min = np.nanmin(X, axis=0)
        col_skew = skew(X, axis=0, nan_policy="omit", bias=False)

        # colonnes à log: min >= 0 et |skew| >= seuil
        self.log_mask_ = (col_min >= 0) & (np.abs(col_skew) >= self.log_skew_threshold)

        # on calcule les bornes IQR sur X après log (sur train uniquement)
        Xt = X.copy()
        Xt[:, self.log_mask_] = np.log1p(Xt[:, self.log_mask_])

        q1 = np.nanpercentile(Xt, 25, axis=0)
        q3 = np.nanpercentile(Xt, 75, axis=0)
        iqr = q3 - q1

        self.low_ = q1 - self.cap_factor * iqr
        self.up_ = q3 + self.cap_factor * iqr

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()

        # log (mêmes colonnes qu'au fit)
        X[:, self.log_mask_] = np.log1p(X[:, self.log_mask_])

        # clip
        X = np.clip(X, self.low_, self.up_)

        return X
