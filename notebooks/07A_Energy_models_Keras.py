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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor

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
from sklearn.wrappers import KerasRegressor
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
y_log = y.apply(np.log1p)

X = X.copy()
X.columns = X.columns.astype(str)
standard_features = [c for c in map(str, standard_features) if c in X.columns]
robust_features = [c for c in map(str, robust_features) if c in X.columns]
categorical_features = [c for c in map(str, categorical_features) if c in X.columns]
X.info()

seed = 42

# 2) split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Scalling

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

# ****************************************************************
# *      Parte 1 :: Transformation des données pour Keras        *
# ****************************************************************

X_train_tf = preprocessor.fit_transform(X_train)
X_test_tf = preprocessor.transform(X_test)

# log target
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

print(X_train_tf.shape, X_test_tf.shape)


def build_tf_model(
    input_dim, hidden_units=64, n_hidden=2, dropout_rate=0.1, learning_rate=1e-3
):
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))

    for _ in range(n_hidden):
        model.add(layers.Dense(hidden_units, activation="swish"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="mae", metrics=["mse"])
    return model


# *****************************************
# *         Parte 1 :: modele de base     *
# *****************************************

keras.utils.set_random_seed(seed)

model = build_tf_model(
    input_dim=X_train_tf.shape[1],
    hidden_units=64,
    n_hidden=2,
    dropout_rate=0.1,
    learning_rate=1e-3,
)

early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

history = model.fit(
    X_train_tf,
    y_train_log,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0,
)

# prédictions en log
y_pred_train_log = model.predict(X_train_tf, verbose=0).ravel()
y_pred_test_log = model.predict(X_test_tf, verbose=0).ravel()

# retour échelle réelle
y_pred_train = np.expm1(y_pred_train_log)
y_pred_test = np.expm1(y_pred_test_log)

print(
    f"R² : {r2_score(y_train, y_pred_train):.3f} (train) et "
    f"{colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} "
    f"{r2_score(y_test, y_pred_test):.3f} "
    f"{colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred_test) ** 0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred_test):.4}")

# ! R² : 0.799 (train) et  0.635  (test)
# ! RMSE : 7.334e+06
# ! MAE : 3.408e+06

# *************************************************
# *             Parte 3 :: GridSearch CV          *
# *************************************************


def make_keras_model(
    meta, hidden_units=64, n_hidden=2, dropout_rate=0.1, learning_rate=1e-3
):
    input_dim = meta["n_features_in_"]

    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))

    for _ in range(n_hidden):
        model.add(layers.Dense(hidden_units, activation="swish"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="mae", metrics=["mse"])
    return model


keras.utils.set_random_seed(seed)

nn_reg = KerasRegressor(
    model=make_keras_model,
    verbose=0,
)

param_grid = {
    "model__hidden_units": [32, 64, 128],
    "model__n_hidden": [1, 2, 3],
    "model__dropout_rate": [0.0, 0.1, 0.2],
    "model__learning_rate": [1e-3, 5e-4],
    "batch_size": [16, 32],
    "epochs": [100, 200],
}

gs = GridSearchCV(
    estimator=nn_reg,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    refit=True,
)

gs.fit(X_train_tf, y_train_log)

print("Best params :", gs.best_params_)
print("Best CV score :", gs.best_score_)

best_nn = gs.best_estimator_

y_pred_test_log = best_nn.predict(X_test_tf).ravel()
y_pred_test = np.expm1(y_pred_test_log)

y_pred_train_log = best_nn.predict(X_train_tf).ravel()
y_pred_train = np.expm1(y_pred_train_log)

print(
    f"R² : {r2_score(y_train, y_pred_train):.3f} (train) et "
    f"{colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} "
    f"{r2_score(y_test, y_pred_test):.3f} "
    f"{colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred_test) ** 0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred_test):.4}")

# ! R² : 0.799 (train) et  0.635  (test)
# ! RMSE : 7.334e+06
# ! MAE : 3.408e+06

# *************************************************
# *         Parte 4 :: RandomizedSearchCV         *
# *************************************************


# *************************************************
# *           Parte 6 :: Validation croisée       *
# *************************************************

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)

scores = cross_validate(
    best_model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    },
    return_train_score=True,
    n_jobs=-1,
)

print("R² train CV mean :", scores["train_r2"].mean().round(3))
print("R² test  CV mean :", scores["test_r2"].mean().round(3))
print("R² test  CV std  :", scores["test_r2"].std().round(3))
print("RMSE test CV mean:", (-scores["test_rmse"].mean()).round(0))
print("MAE  test CV mean:", (-scores["test_mae"].mean()).round(0))

# ! Résultats :
# ! R² CV mean : 0.866 (train) et 0.633 (test)
# ! R² train CV std : 0.078 (test)
# ! RMSE test CV mean: 8198499.0
# ! MAE  test CV mean: 3273585.0
