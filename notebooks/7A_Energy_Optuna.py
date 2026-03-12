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
numeric_features = [
    "Latitude",
    "Longitude",
    "NumberofBuildings",
    "NumberofFloors",
    "PropertyGFATotal",  # Je le garde et j'utiliserai le log_gfa ensuite
    "nb_property_uses",
    "building_age",
    "buildings_gfa_ratio",
    "parking_gfa_ratio",
    #    "gas_prop", suppression car leakage (calculé sur target)
    #    "elec_prop",
    #    "steam_prop",
]

categorical_features = ["BuildingType", "PrimaryPropertyGroup"]

col_sel = numeric_features + categorical_features
X = df[col_sel]
y = df["SiteEnergyUse(kBtu)"]  # df["log_SiteEnergyUse"]

X = X.copy()
X.columns = X.columns.astype(str)
numeric_features = [c for c in map(str, numeric_features) if c in X.columns]
categorical_features = [c for c in map(str, categorical_features) if c in X.columns]
X.info()

# 2) split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Scalling

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

seed = 42

# *****************************************
# *        Parte 1 :: Optuna Catboost     *
# *****************************************

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

X = df[col_sel].copy()
y = df["SiteEnergyUse(kBtu)"].copy()

X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())

for col in categorical_features:
    X[col] = X[col].astype(str).fillna("missing")

cat_features_idx = [X.columns.get_loc(col) for col in categorical_features]


def objective_catboost(trial):
    params = {
        "depth": trial.suggest_int("depth", 6, 9),
        "iterations": trial.suggest_int("iterations", 700, 2000),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_state": 42,
        "verbose": 0,
    }

    scores = []

    for train_idx, test_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, cat_features=cat_features_idx)

        y_pred = model.predict(X_test)
        scores.append(r2_score(y_test, y_pred))

    return np.mean(scores)


study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_catboost, n_trials=40)

print("Best score:", study_cat.best_value)
print("Best params:", study_cat.best_params)

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
