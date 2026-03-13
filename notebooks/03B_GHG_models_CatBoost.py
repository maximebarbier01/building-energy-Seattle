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
df.head(2)

# *****************************************
# *        SELECTION DES FEATURES         *
# *****************************************

# 1) Preprocessing
standard_features = [
    "EnergyProfileScore",
    "dist_downtown",
    #    "nb_certifications",
    #    "is_mixed_use",
    #    "has_electricity",
    "has_steam",
    "has_gas",
]

robust_features = [
    #    "NumberofBuildings",
    "NumberofFloors",
    #    "PropertyGFATotal",
    "largest_use_ratio",
    "gfa_per_building",
    #    "gfa_per_floor",
    "log_gfa",
    "gas_prop",
    "elec_prop",
    "steam_prop",
]

categorical_features = [
    "PrimaryPropertyGroup",
    "PrimaryPropertyType",
    #    "CouncilDistrictCode",
    "EnergyProfileGroup",
    "ListOfAllPropertyUseTypes",
]

# *****************************************
# *          TARGET ET FEATURES           *
# *****************************************

col_sel = standard_features + robust_features + categorical_features

X = df[col_sel].copy()
y = df["TotalGHGEmissions"]

X = X.copy()
X.columns = X.columns.astype(str)
standard_features = [c for c in map(str, standard_features) if c in X.columns]
robust_features = [c for c in map(str, robust_features) if c in X.columns]
categorical_features = [c for c in map(str, categorical_features) if c in X.columns]
X.info()

# *****************************************
# *           TRAIN TEST SPLIT            *
# *****************************************

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

print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

# ! R² : 0.913 (train) et  0.779  (test)
# ! RMSE : 111.3
# ! MAE : 50.98

from sklearn.inspection import permutation_importance

result = permutation_importance(
    cat_model, X_test, y_test, n_repeats=10, random_state=42
)

importance = pd.Series(result.importances_mean, index=X_test.columns).sort_values(
    ascending=False
)

importance.tail(20)

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

cat_model_gs = TransformedTargetRegressor(
    regressor=regressor, func=np.log1p, inverse_func=np.expm1
)

gs = GridSearchCV(
    estimator=cat_model_gs,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    refit=True,
)

gs.fit(X_train, y_train, cat_features=cat_features_idx)

best_model_gs = gs.best_estimator_
y_pred = best_model_gs.predict(X_test)

print(gs.best_params_)
print(gs.best_score_)

print(
    f"R² : {best_model_gs.score(X_train, y_train):.3f} (train) et {colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} {best_model_gs.score(X_test, y_test):.3f} {colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

# ! Résultats :
# ! R² : 0.948 (train) et  0.751  (test)
# ! RMSE : 118.1
# ! MAE : 53.95

best_param = gs.best_params_

cat_params = {k.replace("model__", ""): v for k, v in best_param.items()}

# ? Best params
# {'regressor__depth': 4,
#  'regressor__iterations': 2000,
#  'regressor__l2_leaf_reg': 10,
#  'regressor__learning_rate': 0.05}

# *************************************************
# *         Parte 4 :: RandomazedSearch CV        *
# *************************************************

param_dist = {
    "regressor__depth": randint(3, 9),  # 3 à 8
    "regressor__learning_rate": uniform(0.01, 0.09),  # 0.01 à 0.10
    "regressor__l2_leaf_reg": randint(2, 15),  # 2 à 14
    "regressor__iterations": randint(500, 2500),  # 500 à 2499
    "regressor__bagging_temperature": uniform(0, 2),  # 0 à 2
    "regressor__random_strength": uniform(0, 2),  # 0 à 2
}

param_dist_v2 = {
    "regressor__depth": [3, 4, 5, 6],
    "regressor__learning_rate": [0.01, 0.02, 0.03, 0.05],
    "regressor__l2_leaf_reg": [3, 5, 7, 10, 15],
    "regressor__iterations": [500, 800, 1000, 1500],
    "regressor__bagging_temperature": [0, 0.5, 1, 2],
    "regressor__random_strength": [0, 0.5, 1, 2],
}

regressor = CatBoostRegressor(
    random_state=seed, loss_function="RMSE", eval_metric="RMSE", verbose=0
)

cat_model_rs = TransformedTargetRegressor(
    regressor=regressor, func=np.log1p, inverse_func=np.expm1
)

# 6) RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=cat_model_rs,
    param_distributions=param_dist,
    n_iter=40,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=seed,
    n_jobs=-1,
    refit=True,
    verbose=1,
)

# 7) Fit
random_search.fit(X_train, y_train, cat_features=cat_features_idx)

# 8) Meilleur modèle
best_model_rs = random_search.best_estimator_
y_pred = best_model_rs.predict(X_test)

print(
    f"R² : {best_model_rs.score(X_train, y_train):.3f} (train) et {colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} {best_model_rs.score(X_test, y_test):.3f} {colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")


#! R² : 0.930 (train) et  0.779  (test)
#! RMSE : 111.3
#! MAE : 54.21

best_param_random = random_search.best_params_

cat_random_params = {
    k.replace("regressor__", ""): v for k, v in best_param_random.items()
}

print("CatBoost params nettoyés :", cat_random_params)

# ? Params
# {'bagging_temperature': np.float64(0.5696809887549352),
#  'depth': 3,
#  'iterations': 1198,
#  'l2_leaf_reg': 2,
#  'learning_rate': np.float64(0.09789534602493875),
#  'random_strength': np.float64(0.8220740266364626)}

# *************************************************
# *                   BEST_MODELE                 *
# *************************************************

regressor = CatBoostRegressor(
    depth=3,
    bagging_temperature=float(0.09789534602493875),
    iterations=1200,
    l2_leaf_reg=2,
    learning_rate=float(0.09789534602493875),
    random_strength=float(0.8220740266364626),
    random_state=seed,
    loss_function="RMSE",
    eval_metric="RMSE",
    verbose=0,
)

best_cat_model = TransformedTargetRegressor(
    regressor=regressor, func=np.log1p, inverse_func=np.expm1
)

best_cat_model.fit(X_train, y_train, cat_features=cat_features_idx)

y_pred = best_cat_model.predict(X_test)

print(
    f"R² : {best_cat_model.score(X_train, y_train):.3f} (train) et {colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} {best_cat_model.score(X_test, y_test):.3f} {colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

# ! Résultats :
#! R² : 0.930 (train) et  0.779  (test)
#! RMSE : 111.3
#! MAE : 54.21


# *************************************************
# *           Parte 4 :: Validation croisée       *
# *************************************************

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

scores_tr = []
scores_te = []

rmse_tr = []
rmse_te = []

mae_tr = []
mae_te = []

for train_idx, test_idx in cv.split(X_train):
    X_tr = X_train.iloc[train_idx].copy()
    X_te = X_train.iloc[test_idx].copy()
    y_tr = y_train.iloc[train_idx].copy()
    y_te = y_train.iloc[test_idx].copy()

    # recalcul des colonnes catégorielles sur le fold courant
    cat_features_idx_fold = [
        X_tr.columns.get_loc(col) for col in categorical_features if col in X_tr.columns
    ]

    best_model.fit(X_tr, y_tr, cat_features=cat_features_idx_fold)

    y_pred_tr = best_model.predict(X_tr)
    y_pred_te = best_model.predict(X_te)

    scores_tr.append(r2_score(y_tr, y_pred_tr))
    scores_te.append(r2_score(y_te, y_pred_te))

    rmse_tr.append(mean_squared_error(y_tr, y_pred_tr) ** 0.5)
    rmse_te.append(mean_squared_error(y_te, y_pred_te) ** 0.5)

    mae_tr.append(mean_absolute_error(y_tr, y_pred_tr))
    mae_te.append(mean_absolute_error(y_te, y_pred_te))

print("R² train CV mean :", np.mean(scores_tr).round(3))
print("R² test  CV mean :", np.mean(scores_te).round(3))
print("R² train CV std  :", np.std(scores_tr).round(3))
print("R² test  CV std  :", np.std(scores_te).round(3))

print("RMSE train CV mean :", np.mean(rmse_tr).round(0))
print("RMSE test  CV mean :", np.mean(rmse_te).round(0))

print("MAE train CV mean :", np.mean(mae_tr).round(0))
print("MAE test  CV mean :", np.mean(mae_te).round(0))

# ! ===============================
# ! Conclusion – Modélisation TotalGHGEmissions
# ! ===============================
#
# La target TotalGHGEmissions a été modélisée à l'aide d'un modèle CatBoostRegressor.
# Afin de limiter les effets de skewness et l'influence des valeurs extrêmes,
# la variable cible a été transformée avec log1p.
#
# Les variables directement liées à la consommation énergétique totale
# (SiteEnergyUse, Electricity, NaturalGas, SteamUse) ont été retirées afin
# d'éviter toute fuite de données. Le modèle repose donc uniquement sur
# des variables structurelles du bâtiment, des ratios énergétiques et
# des variables catégorielles décrivant le type d'usage du bâtiment.
#
# Les performances obtenues sont les suivantes :
# - R² test : ~0.78
# - RMSE : ~111
# - MAE : ~54
#
# La validation croisée donne :
# - R² moyen : ~0.68
# - RMSE moyen : ~210
#
# L'écart entre les scores train et test reste modéré, ce qui suggère
# un modèle relativement bien généralisable malgré la taille limitée
# du dataset et l'hétérogénéité des types de bâtiments.
#
# Globalement, le modèle explique environ 78% de la variance des émissions
# de gaz à effet de serre, ce qui constitue une performance solide dans
# un contexte où les variables de consommation énergétique directe ont
# volontairement été exclues.
#
# Le modèle semble donc capturer correctement les relations entre les
# caractéristiques structurelles des bâtiments, leur usage et leurs
# émissions de GES.
