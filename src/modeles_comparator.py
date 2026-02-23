import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import lightgbm as lgb

from xgboost import XGBRegressor


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"Durée d'exécution de {f.__name__}: {te-ts:.3f}s")
        return result
    return timed


@timeit
def regression_pipeline(model, name, X, y, preprocessor, cv=10):
    """
    Compare un modèle via cross_validate sur une Pipeline(preprocessor -> model)
    """
    pipe = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
    }

    score = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # Moyennes
    test_r2 = score["test_r2"].mean()
    train_r2 = score["train_r2"].mean()

    test_mae = -score["test_mae"].mean()
    train_mae = -score["train_mae"].mean()

    test_rmse = np.sqrt(-score["test_mse"].mean())
    train_rmse = np.sqrt(-score["train_mse"].mean())

    # Variabilité (utile pour juger la stabilité)
    test_r2_std = score["test_r2"].std()
    test_mae_std = score["test_mae"].std()

    return pd.DataFrame({
        "modèle": [name],
        "fit_time_mean": [score["fit_time"].mean()],
        "score_time_mean": [score["score_time"].mean()],
        "Test R2 (mean)": [test_r2],
        "Test R2 (std)": [test_r2_std],
        "Train R2 (mean)": [train_r2],
        "Test MAE (mean)": [test_mae],
        "Test MAE (std)": [test_mae_std],
        "Train MAE (mean)": [train_mae],
        "Test RMSE (mean)": [test_rmse],
        "Train RMSE (mean)": [train_rmse],
    })


@timeit
def compare_resultat_pipeline(X_train, y_train, preprocessor, seed=42, cv=10):
    """
    Compare plusieurs modèles avec le même preprocessing, mêmes folds, mêmes métriques.
    """
    models = [
        ("dummy_mean", DummyRegressor(strategy="mean")),
        ("linear_regression", LinearRegression()),
        ("elastic_net", ElasticNet(random_state=seed)),
        ("svr", SVR()),
        ("random_forest", RandomForestRegressor(
            random_state=seed, n_estimators=200, n_jobs=-1
        )),
        ("lgbm", lgb.LGBMRegressor(random_state=seed, verbosity=-1)),
        ("xgboost", XGBRegressor(
            random_state=seed, n_estimators=200, n_jobs=-1
        ))
    ]

    frames = []
    for name, mod in models:
        frames.append(regression_pipeline(mod, name, X_train, y_train, preprocessor, cv=cv))

    df = pd.concat(frames, ignore_index=True)

    # Tri : on privilégie MAE (plus bas = mieux), puis R2 (plus haut = mieux)
    df = df.sort_values(by=["Test MAE (mean)", "Test R2 (mean)"], ascending=[True, False])

    return df.style.hide(axis="index")

@timeit
def compare_resultat_pipeline_final(X_train, y_train, preprocessor, seed=42, cv=10):
    final_models = [
        ("dummy_mean", DummyRegressor(strategy="mean")),
        ("linear_regression", LinearRegression()),

        ("elastic_net_opt", ElasticNet(
            random_state=seed,
            alpha=0.003,
            l1_ratio=0.3
        )),

        ("svr_opt", SVR(
            kernel="rbf",
            C=1,
            epsilon=0.05
            # gamma laissé par défaut ("scale") car non sélectionné dans ton best_params
        )),

        ("random_forest_opt", RandomForestRegressor(
            random_state=seed,
            n_estimators=500,
            max_depth=25,
            min_samples_split=4,
            n_jobs=-1
        )),

        ("lgbm", lgb.LGBMRegressor(
            random_state=seed,
            verbosity=-1
            # optionnel : ajoute ici tes best params si tu veux
        )),

        ("xgboost_opt", XGBRegressor(
            random_state=seed,
            n_estimators=300,
            max_depth=2,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=1.0,
            objective="reg:squarederror",
            n_jobs=-1
        ))
    ]

    models_dict = {name: Pipeline([("preprocessing", preprocessor), ("model", model)])
                   for name, model in final_models}


    frames = [
        regression_pipeline(mod, name, X_train, y_train, preprocessor, cv=cv)
        for name, mod in final_models
    ]

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(by=["Test RMSE (mean)", "Test MAE (mean)"], ascending=[True, True])
    return df.style.hide(axis="index"), models_dict