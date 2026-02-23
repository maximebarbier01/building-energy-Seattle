import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_log_and_real(y_true_log, y_pred_log): # y_true_log = y_test_log
    
    # métriques en log
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)

    # métriques en échelle réelle avec expm1
    y_true_real = np.expm1(y_true_log)
    y_pred_real = np.expm1(y_pred_log)

    rmse_real = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    mae_real = mean_absolute_error(y_true_real, y_pred_real)
    mape_real = np.mean(np.abs((y_true_real - y_pred_real) / y_true_real)) * 100

    return {
        "RMSE_log": rmse_log,
        "MAE_log": mae_log,
        "R2_log": r2_log,
        "RMSE_real": rmse_real,
        "MAE_real": mae_real,
        "MAPE_real_%": mape_real,
    }

def tune_compare_models_once(
    models_with_grids,
    preprocessor,
    X_train, y_train_log,
    X_test, y_test_log,
    cv=5,
    seed=42,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
):
    rows = []
    best_models = {}

    # Important : même split CV pour tous
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=seed)

    for name, estimator, param_grid in models_with_grids:
        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("model", estimator),
        ])

        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=True
        )
        grid.fit(X_train, y_train_log)

        best_pipe = grid.best_estimator_
        best_models[name] = best_pipe

        # prédictions test (en log)
        y_pred_log = best_pipe.predict(X_test)

        metrics = evaluate_log_and_real(y_test_log, y_pred_log)

        rows.append({
            "model": name,
            "best_params": grid.best_params_,
            "CV_best_RMSE_log": -grid.best_score_,
            **metrics
        })

    df = pd.DataFrame(rows).sort_values(
        by=["MAPE_real_%", "RMSE_real", "CV_best_RMSE_log"],
        ascending=True
    )
    return df, best_models

# Exemple d'utilisation 
#models_with_grids = [
#    ("elastic_net", ElasticNet(random_state=42), {
#        "model__alpha": [1e-4, 1e-3, 1e-2],
#        "model__l1_ratio": [0.2, 0.5, 0.8]
#    }),
#    ("lgbm", lgb.LGBMRegressor(random_state=42, verbosity=-1), {
#        "model__n_estimators": [300, 800],
#       "model__learning_rate": [0.03, 0.08],
#        "model__num_leaves": [31, 63],
#        "model__subsample": [0.8, 1.0]
#    }),
#]

# Puis
#df_results, best_models = tune_compare_models_once(
#    models_with_grids,
#    preprocessor,
#    X_train, y_train_energy,   # log target
#    X_test,  y_test_energy,    # log target
#    cv=5
#)
#display(df_results)