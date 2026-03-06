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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression
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

# Import de la table

df_backup = pd.read_csv(
    "/home/maxime/projects/building-energy-Seattle/data/processed/data_to_use.csv"
)
df_backup.drop("Unnamed: 0", axis=1, inplace=True)

df = df_backup.copy()
df["CouncilDistrictCode"] = df["CouncilDistrictCode"].astype("category")
df["ZipCode"] = df["ZipCode"].astype("category")
df.info()

# 1) Preprocessing
numeric_features = [
    "Latitude",
    "Longitude",
    "NumberofBuildings",
    "NumberofFloors",
    "PropertyGFATotal",
    "nb_property_uses",
    "building_age",
    "buildings_gfa_ratio",
    "parking_gfa_ratio",
]

categorical_features = [
    "BuildingType",
    "PrimaryPropertyGroup",
]

col_sel = numeric_features + categorical_features

X = df[col_sel].copy()
y = df["TotalGHGEmissions"].copy()

X = X.copy()
X.columns = X.columns.astype(str)
numeric_features = [c for c in map(str, numeric_features) if c in X.columns]
categorical_features = [c for c in map(str, categorical_features) if c in X.columns]
X.info()

seed = 42

# *****************************************
# *         Parte 1 :: modele de base     *
# *****************************************

# 2) split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

# 3) Modelling

cat_features_idx = [X.columns.get_loc(col) for col in categorical_features]

cat_model = CatBoostRegressor(
    random_state=seed, loss_function="RMSE", eval_metric="RMSE", verbose=0
)

cat_model.fit(X_train, y_train, cat_features=cat_features_idx)

y_pred = cat_model.predict(X_test)

print(
    f"R² : {cat_model.score(X_train, y_train):.3f} (train) et {colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} {cat_model.score(X_test, y_test):.3f} {colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

# ! R² : 0.945 (train) et  0.058  (test)
# ! RMSE : 229.7
# ! MAE : 104.3

# *************************************************
# *      Parte 2 :: modele avec early stopping    *
# *************************************************

# 2) split train/val

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=seed
)

# 3) Modelling

cat_model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_state=seed,
    verbose=0,
)

cat_model.fit(
    X_tr,
    y_tr,
    eval_set=(X_val, y_val),
    cat_features=cat_features_idx,
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
# ! R² : 0.659 (train) et  0.314  (test)
# ! RMSE : 196.1
# ! MAE : 101.9

# *************************************************
# *           Parte 3 :: Validation croisée       *
# *************************************************

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)

scores_te = []
scores_tr = []

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
        random_state=seed,
        verbose=0,
    )

    model.fit(X_tr, y_tr, cat_features=cat_features_idx)

    y_pred_te = model.predict(X_te)
    scores_te.append(r2_score(y_te, y_pred_te))

    y_pred_tr = model.predict(X_tr)
    scores_tr.append(r2_score(y_tr, y_pred_tr))

print("R² train CV mean:", np.mean(scores_tr).round(2))
print("R² test CV mean:", np.mean(scores_te).round(2))
print("R² train CV std :", np.std(scores_tr).round(2))
print("R² test CV std :", np.std(scores_te).round(2))

# ! Résultats :
# ! R² CV mean : 0.98 (train) et 0.33 (test)
# ! R² train CV std : 0.01 (train) et 0.29 (test)

# *************************************************
# *             Parte 4 :: GridSearch CV          *
# *************************************************

param_grid = {
    "depth": [4, 6, 8],
    "learning_rate": [0.02, 0.03, 0.05],
    "l2_leaf_reg": [3, 5, 10],
    "iterations": [2000, 3000, 4000],
}

gs = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    refit=True,
)

gs.fit(X_train, y_train, cat_features=cat_features_idx)

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
