import numpy as np
import pandas as pd
import time

import lightgbm as lgb

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_validate,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MultiLabelBinarizer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVR

import src.outliers_treatment as ot


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"Durée d'exécution de {f.__name__}: {te-ts:.3f}s")
        return result

    return timed


# =========================
# 1) Split and scale
# =========================


def make_preprocessor(numeric_features, categorical_features, sparse_output=False):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("outliers", ot.OutlierLogCapper(cap_factor=3.0, log_skew_threshold=1.0)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed)


# =========================
# 2) Evaluation du modèle
# =========================


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": rmse,
    }


def evaluate_fitted(estimator, X_test, y_test, y_is_log=False):
    y_pred = estimator.predict(X_test)
    # Si tu utilises TransformedTargetRegressor avec inverse_func=expm1, y_pred est déjà en "réel".
    # Donc y_is_log sert surtout à documenter / tracer.
    return regression_metrics(y_test, y_pred)


# =========================
# 3) Preprocessor
# =========================


def make_pipeline(preprocessor, model, use_log_target=False):
    if use_log_target:
        reg = TransformedTargetRegressor(
            regressor=model, func=np.log1p, inverse_func=np.expm1
        )
        return Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", reg),
            ]
        )
    else:
        return Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", model),
            ]
        )


# =========================
# 4) Cross Validation
# =========================


@timeit
def cross_validation(pipe, param_grid, X_train, y_train, seed=42, cv=5):
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",  # RMSE (CV)
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs


# =========================
# 4) Model comparator
# =========================*


@timeit
def compare_models(
    models,
    param_grids,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
    use_log_target=False,
    cv=5,
    seed=42,
):
    rows = []

    for name, model in models:
        pipe = make_pipeline(preprocessor, model, use_log_target=use_log_target)

        grid = param_grids.get(name, {})
        # si TransformedTargetRegressor, les params sont model__regressor__*
        if use_log_target and grid:
            grid = {f"model__regressor__{k}": v for k, v in grid.items()}
        else:
            grid = {f"model__{k}": v for k, v in grid.items()}

        if grid:
            gs = cross_validation(pipe, grid, X_train, y_train, seed=seed, cv=cv)
            best_est = gs.best_estimator_
            best_cv_rmse = -float(gs.best_score_)
            best_params = gs.best_params_
        else:
            best_est = pipe.fit(X_train, y_train)
            best_cv_rmse = np.nan
            best_params = {}

        test_metrics = evaluate_fitted(
            best_est, X_test, y_test, y_is_log=use_log_target
        )

        rows.append(
            {
                "model": name,
                "use_log_target": use_log_target,
                "cv_rmse": best_cv_rmse,
                **test_metrics,
                "best_params": best_params,
                "estimator": best_est,  # on stocke le pipeline fitted
            }
        )

    return pd.DataFrame(rows).sort_values(["RMSE", "MAE"], ascending=True)
