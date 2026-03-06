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
# *         Parte 1 :: modele de base     *
# *****************************************

lgb_model = lgb.LGBMRegressor(random_state=seed, n_jobs=-1)

lgb_pipe = Pipeline(steps=[("prep", preprocessor), ("model", lgb_model)])

lgb_pipe.fit(X_train, y_train)

y_pred = lgb_pipe.predict(X_test)

print(
    f"R² : {lgb_pipe.score(X_train, y_train):.3f} (train) et {lgb_pipe.score(X_test, y_test):.3f} (test)"
)
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

# ! 0.948 (train) et 0.594 (test)
# ! RMSE : 7.739e+06
# ! MAE : 3.674e+06

# **************************************************
# *         Parte 2 :: modele anti-overfitting     *
# **************************************************

lgb_model = lgb.LGBMRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5,
    random_state=seed,
    n_jobs=-1,
)

lgb_model_v2 = lgb.LGBMRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=20,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_samples=20,
    reg_lambda=10,
    random_state=42,
    n_jobs=-1,
)

lgb_model_v3 = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=50,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=3,
    random_state=42,
    n_jobs=-1,
)

lgb_pipe = Pipeline(steps=[("prep", preprocessor), ("model", lgb_model)])

lgb_pipe.fit(X_train, y_train)

y_pred = lgb_pipe.predict(X_test)

print(
    f"R² : {lgb_pipe.score(X_train, y_train):.3f} (train) et {colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} {lgb_pipe.score(X_test, y_test):.3f} {colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

# ! R² : 0.747 (train) et  0.638  (test)
# ! RMSE : 7.307e+06
# ! MAE : 3.707e+06

# *************************************************
# *           Parte 3 :: Validation croisée       *
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

    model = clone(lgb_pipe)
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
# ! R² CV mean : 0.75 (train) et 0.59 (test)
# ! R² train CV std : 0.01 (train) et 0.07 (test)

# *************************************************
# *             Parte 4 :: GridSearch CV          *
# *************************************************

# Grid Search CV

param_grid = {
    "model__n_estimators": [400, 600, 800],
    "model__max_depth": [10, 12, 14],
    "model__max_features": [0.4, 0.5, 0.6],
    "model__min_samples_split": [8, 10, 12],
    "model__min_samples_leaf": [3, 5, 7],
}

gs = GridSearchCV(
    estimator=rf_pipe, param_grid=param_grid, scoring="r2", cv=5, n_jobs=-1, refit=True
)

gs.fit(X_train, y_train)

best_model = gs.best_estimator_  # pipeline complet entraîné sur tout X_train

y_pred = best_model.predict(X_test)

print(
    f"R² : {best_model.score(X_train, y_train):.3f} (train) et {colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} {best_model.score(X_test, y_test):.3f} {colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

best_param = gs.best_params_

xgb_params = {k.replace("model__", ""): v for k, v in best_param.items()}

# ! R² : 0.787 (train) et  0.645  (test)
# ! RMSE : 7.24e+06
# ! MAE : 3.638e+06


# *************************************************
# *         Parte 5 :: RandomizedSearchCV         *
# *************************************************


param_dist = {
    "model__n_estimators": randint(400, 900),
    "model__max_depth": randint(8, 18),
    "model__max_features": uniform(0.35, 0.35),  # entre 0.35 et 0.70
    "model__min_samples_split": randint(6, 16),
    "model__min_samples_leaf": randint(2, 10),
}

random_search = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    n_iter=40,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_train, y_train)


best_model = random_search.best_estimator_  # pipeline complet entraîné sur tout X_train

y_pred = best_model.predict(X_test)

print(
    f"R² : {best_model.score(X_train, y_train):.3f} (train) et {colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} {best_model.score(X_test, y_test):.3f} {colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

best_param_random = random_search.best_params_

# ? Avec param_dist
# ! R² : 0.715 (train) et  0.160  (test)
# ! RMSE : 216.8
# ! MAE : 108.8
