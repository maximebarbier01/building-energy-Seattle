# =========================
# Standard library
# =========================
import importlib
import math
import os
import re
import sys
import time

# =========================
# Machine Learning
# =========================
import lightgbm as lgb
from catboost import CatBoostRegressor

# =========================
# Visualisation
# =========================
import matplotlib.pyplot as plt
import missingno as msno

# =========================
# Data / scientific stack
# =========================
import numpy as np
import pandas as pd
import pingouin as pg

# =========================
# Stats
# =========================
import scipy.stats as stats
import seaborn as sns
from dython.nominal import (
    associations,
    correlation_ratio,  # eta²
    cramers_v,  # V de Cramér
    theils_u,  # Alternative asymétrique
)
from scipy.stats import kruskal, randint, uniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_validate,
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
    RepeatedKFold,
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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

# Pour mise en forme des résultats
import colorama

import sys

print(sys.executable)

# =========================
# Local imports (src/)
# =========================
sys.path.append(os.path.abspath(".."))  # si notebook dans /notebooks

import src.analyse_bivariee as ab
import src.association_report_function as ar
import src.fast_modeles_comparator as qc
import src.modeles_comparator as mdl
import src.outliers_function as of
import src.outliers_treatment as ot
import src.tunning_parameter as tp

importlib.reload(of)

# =========================
# Pandas display options
# =========================
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{:,.2f}".format

# *****************************************
# *          IMPORT DES TABLES            *
# *****************************************

df_backup = pd.read_csv(
    "/home/maxime/projects/building-energy-Seattle/data/processed/data_to_use.csv"
)
df_backup.drop("Unnamed: 0", axis=1, inplace=True)

df = df_backup.copy()
df["CouncilDistrictCode"] = df["CouncilDistrictCode"].astype("category")
df["ZipCode"] = df["ZipCode"].astype("category")
df.info()

# 1) Preprocessing
standard_features = [
    "Latitude",
    "Longitude",
    "building_age",
    "EnergyProfileScore",
]

robust_features = [
    "NumberofBuildings",
    "NumberofFloors",
    "PropertyGFATotal",
    "nb_property_uses",
    "buildings_gfa_ratio",
    "parking_gfa_ratio",
]

categorical_features = [
    "BuildingType",
    "PrimaryPropertyGroup",
    "EnergyProfileGroup",
    "Neighborhood",
]

col_sel = standard_features + robust_features + categorical_features

X = df[col_sel].copy()
y = df["SiteEnergyUse(kBtu)"]

X = X.copy()
X.columns = X.columns.astype(str)
standard_features = [c for c in map(str, standard_features) if c in X.columns]
robust_features = [c for c in map(str, robust_features) if c in X.columns]
categorical_features = [c for c in map(str, categorical_features) if c in X.columns]
X.info()

seed = 42


# 2) split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

# *****************************************
# *         Parte 1 :: modele de base     *
# *****************************************

cat_features_idx = [X_train.columns.get_loc(col) for col in categorical_features]

regressor = CatBoostRegressor(
    random_state=seed, loss_function="RMSE", eval_metric="RMSE", verbose=0
)

cat_model = TransformedTargetRegressor(
    regressor=regressor, func=np.log1p, inverse_func=np.expm1
)

cat_model.fit(X_train, y_train, cat_features=cat_features_idx)

y_pred = cat_model.predict(X_test)

print(
    f"R² : {cat_model.score(X_train, y_train):.3f} (train) et {colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} {cat_model.score(X_test, y_test):.3f} {colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred_test)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred_test):.4}")

# ! R² : 0.853 (train) et  0.685  (test)
# ! RMSE : 6.783e+06
# ! MAE : 3.102e+06

# *************************************************
# *      Parte 2 :: modele avec early stopping    *
# *************************************************

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=seed
)

cat_features_idx = [X_tr.columns.get_loc(col) for col in selected_categorical_features]

regressor = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_state=seed,
    verbose=0,
)

cat_model = TransformedTargetRegressor(
    regressor=regressor, func=np.log1p, inverse_func=np.expm1
)

cat_model.fit(
    X_tr,
    y_tr,
    cat_features=cat_features_idx,
    eval_set=(X_val, y_val),
    use_best_model=True,
    early_stopping_rounds=100,
)

y_pred = cat_model.predict(X_test)

print(
    f"R² : {cat_model.score(X_train, y_train):.3f} (train) et {colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} {cat_model.score(X_test, y_test):.3f} {colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

# ! Résultats :
# ! R² : 0.864 (train) et  0.659  (test)
# ! RMSE : 7.286e+06
# ! MAE : 3.512e+06

# *************************************************
# *             Parte 3 :: GridSearch CV          *
# *************************************************


param_grid = {
    "regressor__depth": [4, 6, 8],
    "regressor__learning_rate": [0.02, 0.03, 0.05],
    "regressor__l2_leaf_reg": [3, 5, 10],
    "regressor__iterations": [1000, 2000],
}

regressor = CatBoostRegressor(
    random_state=seed,
    loss_function="RMSE",
    eval_metric="RMSE",
    verbose=0,
)

cat_model = TransformedTargetRegressor(
    regressor=regressor, func=np.log1p, inverse_func=np.expm1
)

gs = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    refit=True,
)

gs.fit(X_train, y_train, cat_features=cat_features_idx)

best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)

print(gs.best_params_)
print(gs.best_score_)

best_model = gs.best_estimator_  # pipeline complet entraîné sur tout X_train

y_pred = best_model.predict(X_test)

print(
    f"R² : {best_model.score(X_train, y_train):.3f} (train) et {colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} {best_model.score(X_test, y_test):.3f} {colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

# ! Résultats :
# ! R² : 0.891 (train) et  0.676  (test)
# ! RMSE : 6.908e+06
# ! MAE : 3.313e+06

best_param = gs.best_params_

cat_params = {k.replace("model__", ""): v for k, v in best_param.items()}


# *************************************************
# *           Parte 4 :: Validation croisée       *
# *************************************************

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)

scores_te = []
scores_tr = []
rmse_scores = []
mae_scores = []

for train_idx, test_idx in cv.split(X):
    X_tr = X.iloc[train_idx]
    X_te = X.iloc[test_idx]
    y_tr = y.iloc[train_idx]
    y_te = y.iloc[test_idx]

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        random_state=42,
        verbose=0,
    )

    model.fit(X_tr, y_tr, cat_features=cat_features_idx)

    y_pred_te = model.predict(X_te)
    scores_te.append(r2_score(y_te, y_pred_te))

    y_pred_tr = model.predict(X_tr)
    scores_tr.append(r2_score(y_tr, y_pred_tr))
    rmse_scores.append(np.sqrt(mean_squared_error(y_te, y_pred_te)))
    mae_scores.append(mean_absolute_error(y_te, y_pred_te))

print("R² train CV mean:", np.mean(scores_tr).round(3))
print("R² test CV mean:", np.mean(scores_te).round(3))
print("R² train CV std :", np.std(scores_tr).round(3))
print("R² test CV std :", np.std(scores_te).round(3))
print("RMSE CV mean:", np.mean(rmse_scores).round(0))
print("RMSE CV std :", np.std(rmse_scores).round(0))
print("MAE CV mean :", np.mean(mae_scores).round(0))
print("MAE CV std  :", np.std(mae_scores).round(0))

# ! Résultats :
# ! R² CV mean : 0.97 (train) et 0.66 (test)
# ! R² train CV std : 0.01 (train) et 0.06 (test)

# *************************************************
# *     Parte 4 :: RandomForest vs CatBoost       *
# *************************************************

# =========================
# 1. Données RF
# =========================
X_rf = df[col_sel].copy()
y_rf = df["SiteEnergyUse(kBtu)"].copy()

# 3) Scallling

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
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

# =========================
# 2. Pipeline RF
# =========================
rf_model = RandomForestRegressor(
    max_depth=25,
    max_features=0.6,
    min_samples_split=5,
    n_estimators=400,
    n_jobs=-1,
    random_state=seed,
)

rf_pipe = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", rf_model),
    ]
)

# =========================
# 3. Données CatBoost
# =========================
X_cat = df[col_sel].copy()
y_cat = df["SiteEnergyUse(kBtu)"].copy()

X_cat[numeric_features] = X_cat[numeric_features].fillna(
    X_cat[numeric_features].median()
)

for col in categorical_features:
    X_cat[col] = X_cat[col].astype(str).fillna("missing")

cat_features_idx = [X_cat.columns.get_loc(col) for col in categorical_features]

cat_model = CatBoostRegressor(
    random_state=seed, loss_function="RMSE", eval_metric="RMSE", verbose=0, **cat_params
)

# ===================================
# 4. Boucle de comparaison RepeatCV
# ===================================

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=seed)

results = []

for model_name in ["RandomForest", "CatBoost"]:
    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for train_idx, test_idx in cv.split(X_rf):
        # -------------------------
        # RandomForest
        # -------------------------
        if model_name == "RandomForest":
            X_train = X_rf.iloc[train_idx]
            X_test = X_rf.iloc[test_idx]
            y_train = y_rf.iloc[train_idx]
            y_test = y_rf.iloc[test_idx]

            model = clone(rf_pipe)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # -------------------------
        # CatBoost
        # -------------------------
        elif model_name == "CatBoost":
            X_train = X_cat.iloc[train_idx]
            X_test = X_cat.iloc[test_idx]
            y_train = y_cat.iloc[train_idx]
            y_test = y_cat.iloc[test_idx]

            model = CatBoostRegressor(
                random_state=seed, loss_function="RMSE", eval_metric="RMSE", verbose=0
            )

            model.fit(X_train, y_train, cat_features=cat_features_idx)
            y_pred = model.predict(X_test)

        # -------------------------
        # Metrics
        # -------------------------
        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

    results.append(
        {
            "model": model_name,
            "r2_mean": np.mean(r2_scores),
            "r2_std": np.std(r2_scores),
            "rmse_mean": np.mean(rmse_scores),
            "rmse_std": np.std(rmse_scores),
            "mae_mean": np.mean(mae_scores),
            "mae_std": np.std(mae_scores),
        }
    )

results_df = pd.DataFrame(results).sort_values("r2_mean", ascending=False)
results_df
