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
import optuna

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
standard_features = [
    #    "Latitude",
    #    "Longitude",
    #    "building_age",
    #    "EnergyProfileScore",
    "dist_downtown"
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
    #    "BuildingType",
    "PrimaryPropertyGroup",
    #    "PrimaryPropertyType",
    "EnergyProfileGroup",
    #    "Neighborhood",
    "ListOfAllPropertyUseTypes",
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
# *        Parte 1 :: Optuna Catboost     *
# *****************************************

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)

def objective(trial):
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_state": seed,
        "verbose": 0,
        "depth": trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.05, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 12.0, log=True),
        "iterations": trial.suggest_int("iterations", 700, 1800),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
    }

    rmse_scores = []

    for train_idx, valid_idx in cv.split(X_train):
        X_tr = X_train.iloc[train_idx].copy()
        X_val = X_train.iloc[valid_idx].copy()
        y_tr = y_train.iloc[train_idx].copy()
        y_val = y_train.iloc[valid_idx].copy()

        # log target
        y_tr_log = np.log1p(y_tr)
        y_val_log = np.log1p(y_val)

        # indices des variables catégorielles dans le fold courant
        cat_features_idx_fold = [
            X_tr.columns.get_loc(col)
            for col in categorical_features
            if col in X_tr.columns
        ]

        model = CatBoostRegressor(**params)

        model.fit(
            X_tr,
            y_tr_log,
            cat_features=cat_features_idx_fold,
            eval_set=(X_val, y_val_log),
            use_best_model=True,
            early_stopping_rounds=100,
        )

        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)

        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

study = optuna.create_study(direction="minimize", study_name="catboost_rmse")
study.optimize(objective, n_trials=40, show_progress_bar=True)

print("Best trial:")
print("  RMSE CV :", study.best_value)
print("  Params  :", study.best_params)

best_params = study.best_params.copy()

final_model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    random_state=seed,
    verbose=0,
    **best_params
)

cat_features_idx = [
    X_train.columns.get_loc(col)
    for col in categorical_features
    if col in X_train.columns
]

y_train_log = np.log1p(y_train)

final_model.fit(
    X_train,
    y_train_log,
    cat_features=cat_features_idx
)

# prédictions
y_pred_train = np.expm1(final_model.predict(X_train))
y_pred_test = np.expm1(final_model.predict(X_test))

print(
    f"R² : {r2_score(y_train, y_pred_train):.3f} (train) et "
    f"{colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} "
    f"{r2_score(y_test, y_pred_test):.3f} "
    f"{colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred_test) ** 0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred_test):.4}")

# *****************************************
# *        Parte 2 :: Optuna Catboost     *
# *****************************************

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

X_rf = df[col_sel].copy()
y_rf = df["SiteEnergyUse(kBtu)"].copy()


def objective_rf(trial):
    rf_model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 300, 1000),
        max_depth=trial.suggest_int("max_depth", 6, 20),
        max_features=trial.suggest_float("max_features", 0.3, 0.7),
        min_samples_split=trial.suggest_int("min_samples_split", 5, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 10),
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("model", rf_model),
        ]
    )

    scores = []

    for train_idx, test_idx in cv.split(X_rf):
        X_train = X_rf.iloc[train_idx]
        X_test = X_rf.iloc[test_idx]
        y_train = y_rf.iloc[train_idx]
        y_test = y_rf.iloc[test_idx]

        model = clone(pipe)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        scores.append(r2_score(y_test, y_pred))

    return np.mean(scores)


study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=40)

print("Best score:", study_rf.best_value)
print("Best params:", study_rf.best_params)
