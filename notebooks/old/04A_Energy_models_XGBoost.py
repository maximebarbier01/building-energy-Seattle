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

# 4) Modelling

# Entraînement simple
modele = XGBRegressor(random_state=57, n_jobs=-1)  # Paramètres par défaut

pipe = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", modele),
    ]
)

# Grid Search CV

param_grid = {
    "model__n_estimators": [400, 800],
    "model__max_depth": [4, 6],
    "model__learning_rate": [0.03, 0.07],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    refit=True,
)

gs.fit(X_train, y_train)

best_model = gs.best_estimator_  # pipeline complet entraîné sur tout X_train

y_pred = best_model.predict(X_test)

print(
    f"R² : {best_model.score(X_train, y_train):.3f} (train) et {colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} {best_model.score(X_test, y_test):.3f} {colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

# RandomizedSearchCV


best_param = random_search.best_params_

scores = cross_val_score(best_model, X, y, cv=5, scoring="r2")

scores.mean(), scores.std()

from sklearn.model_selection import RepeatedKFold, cross_val_score

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

scores = cross_val_score(best_model, X, y, cv=cv, scoring="r2", n_jobs=-1)

scores.mean(), scores.std()


# ! R² : 1.000 (train) et  0.438  (test)
# ! RMSE : 9.109e+06
# ! MAE : 4.236e+06
