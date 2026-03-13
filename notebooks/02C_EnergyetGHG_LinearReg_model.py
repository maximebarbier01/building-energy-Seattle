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
y = df["SiteEnergyUse(kBtu)"].copy()

X = X.copy()
X.columns = X.columns.astype(str)
standard_features = [c for c in map(str, standard_features) if c in X.columns]
robust_features = [c for c in map(str, robust_features) if c in X.columns]
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

# 3) Scallling

standard_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

robust_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("outliers", ot.OutlierLogCapper(cap_factor=3.0, log_skew_threshold=1.0)),
        ("scaler", RobustScaler()),
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
        ("num_std", standard_pipeline, standard_features),
        ("num_rob", robust_pipeline, robust_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

# 3) Modelling

# Entraînement simple

targets = ["TotalGHGEmissions", "SiteEnergyUse(kBtu)"]

for target in targets:
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modele = LinearRegression()

    lr = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", modele),
        ]
    )

    lr.fit(X_train, y_train)

    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)

    print(f"\n===== Target : {target} =====")
    print(
        f"R² : {lr.score(X_train, y_train):.3f} (train) et "
        f"{colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK}"
        f"{lr.score(X_test, y_test):.3f}"
        f"{colorama.Style.RESET_ALL} (test)"
    )
    print(f"RMSE : {mean_squared_error(y_test, y_pred_test)**0.5:.4f}")
    print(f"MAE : {mean_absolute_error(y_test, y_pred_test):.4f}")

# ! R² : 0.741 (train) et  0.473  (test)
# ! RMSE : 8.819e+06
# ! MAE : 5.444e+06

# *************************************************
# *           Parte 2 :: Validation croisée       *
# *************************************************

# 2) split train/val

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=seed
)

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)

scores_te = []
scores_tr = []

for train_idx, test_idx in cv.split(X):
    X_tr = X.iloc[train_idx]
    X_te = X.iloc[test_idx]
    y_tr = y.iloc[train_idx]
    y_te = y.iloc[test_idx]

    model = modele

    model.fit(X_tr, y_tr)

    y_pred_te = model.predict(X_te)
    scores_te.append(r2_score(y_te, y_pred_te))

    y_pred_tr = model.predict(X_tr)
    scores_tr.append(r2_score(y_tr, y_pred_tr))

print("R² train CV mean:", np.mean(scores_tr).round(2))
print("R² test CV mean:", np.mean(scores_te).round(2))
print("R² train CV std :", np.std(scores_tr).round(2))
print("R² test CV std :", np.std(scores_te).round(2))

# ! Résultats :
# ! R² CV mean : 0.0 (train) et -0.01 (test)
# ! R² train CV std : 0.0 (train) et 0.01 (test)
