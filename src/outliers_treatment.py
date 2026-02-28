import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierLogCapper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        log_keywords=("kBtu", "GFA", "GHG", "Emissions", "EUI"),
        log_skew_threshold=1.0,
        cap_factor=3.0,
        eps=1e-6,
    ):
        self.log_keywords = log_keywords
        self.log_skew_threshold = log_skew_threshold
        self.cap_factor = cap_factor
        self.eps = eps

    @staticmethod
    def _is_binary_series(s: pd.Series) -> bool:
        vals = pd.Series(s.dropna().unique())
        if len(vals) <= 2:
            try:
                return set(vals.astype(float)) <= {0.0, 1.0}
            except Exception:
                return False
        return False

    @staticmethod
    def _iqr_bounds(s: pd.Series, factor: float):
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        return q1 - factor * iqr, q3 + factor * iqr

    def fit(self, X, y=None):
        X = X.copy()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        self.binary_cols_ = [c for c in num_cols if self._is_binary_series(X[c])]
        self.cont_cols_ = [c for c in num_cols if c not in self.binary_cols_]

        # décider quelles colonnes logguer (décision FIT sur train uniquement)
        self.log_cols_ = []
        for c in self.cont_cols_:
            s = X[c].dropna()
            if s.empty:
                continue
            if s.min() >= 0 and abs(s.skew()) >= self.log_skew_threshold:
                if any(k.lower() in c.lower() for k in self.log_keywords):
                    self.log_cols_.append(c)

        # calculer les bornes IQR (après log) sur train uniquement
        Xt = X.copy()
        for c in self.log_cols_:
            Xt[c] = np.log1p(Xt[c])

        self.clip_bounds_ = {}
        for c in self.cont_cols_:
            s = Xt[c].dropna()
            if s.empty:
                continue
            low, up = self._iqr_bounds(s, factor=self.cap_factor)
            self.clip_bounds_[c] = (low, up)

        return self

    def transform(self, X):
        X = X.copy()

        # appliquer log sur les mêmes colonnes que le train
        for c in getattr(self, "log_cols_", []):
            if c in X.columns:
                X[c] = np.log1p(X[c])

        # appliquer les mêmes bornes apprises sur train
        for c, (low, up) in getattr(self, "clip_bounds_", {}).items():
            if c in X.columns:
                X[c] = X[c].clip(low, up)

        return X
