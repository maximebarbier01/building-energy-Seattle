import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def timeit(f):
    def timed(*args, **kw):
        import time
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"Durée d'exécution de {f.__name__}: {te-ts:.3f}s")
        return result
    return timed


@timeit
def tune_model(pipe, param_grid, X_train, y_train, cv=5):
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )
    grid.fit(X_train, y_train)
    return grid


@timeit
def tune_and_compare_models(models_with_grids, preprocessor, X_train, y_train, X_test, y_test, cv=5, seed=42):
    rows = []

    for name, estimator, param_grid in models_with_grids:
        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("model", estimator),
        ])

        grid = tune_model(pipe, param_grid, X_train, y_train, cv=cv)

        best_pipe = grid.best_estimator_
        y_pred = best_pipe.predict(X_test)

        rows.append({
            "model": name,
            "best_params": grid.best_params_,
            "cv_RMSE_best": -grid.best_score_,  # repasse en positif
            "test_RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "test_MAE": mean_absolute_error(y_test, y_pred),
            "test_R2": r2_score(y_test, y_pred),
        })

    df = pd.DataFrame(rows).sort_values(["test_RMSE", "cv_RMSE_best"], ascending=True)
    return df