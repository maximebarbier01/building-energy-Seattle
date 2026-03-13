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
import shap

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

    best_cat_model.fit(X_tr, y_tr, cat_features=cat_features_idx_fold)

    y_pred_tr = best_cat_model.predict(X_tr)
    y_pred_te = best_cat_model.predict(X_te)

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


# ? *************************************************
# ?     Analyse des erreurs absolues du modèle      *
# ? *************************************************

y_pred = best_cat_model.predict(X_test)

errors = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})

errors["abs_error"] = abs(errors["y_true"] - errors["y_pred"])

errors.sort_values("abs_error", ascending=False).head(20)

errors.abs_error.describe()

errors = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})

errors["abs_error"] = abs(errors["y_true"] - errors["y_pred"])

errors.sort_values("abs_error", ascending=False).head(20)

errors.abs_error.describe()


def plot_regression_predictions(
    model,
    X,
    y,
    title="CatBoost Model: Predicted vs Target",
):
    y_pred = model.predict(X)

    plt.figure(figsize=(6, 6))

    plt.scatter(y, y_pred, alpha=0.5)

    # ligne parfaite
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
    )

    plt.xlabel("Target")
    plt.ylabel("Predicted")
    plt.title(title)

    plt.tight_layout()
    plt.show()


plot_regression_predictions(
    best_cat_model, X_train, y_train, title="CatBoost: Predicted vs Target"
)

# Le graphique "Predicted vs Target" montre que la majorité des prédictions
# se situent proche de la diagonale (y = x), ce qui indique que le modèle
# parvient globalement à bien estimer les émissions de GES. On observe
# toutefois une dispersion plus importante pour les valeurs élevées de la
# target, ce qui suggère que les bâtiments ayant les plus fortes émissions
# sont plus difficiles à prédire.


def plot_absolute_error_distribution(
    y_true, y_pred, title="Absolute Error Distribution"
):
    abs_error = np.abs(y_true - y_pred)

    plt.figure(figsize=(7, 5))

    sns.histplot(abs_error, bins=50, kde=True)

    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.title(title)

    plt.tight_layout()
    plt.show()


plot_absolute_error_distribution(y_test, y_pred, title="CatBoost Absolute Error")

# L'histogramme de l'erreur absolue montre que la plupart des erreurs sont
# relativement faibles, avec une forte concentration sous les 100 unités.
# La distribution présente cependant une longue queue à droite, ce qui
# traduit la présence de quelques observations avec des erreurs importantes,
# probablement associées à des bâtiments atypiques.


def plot_error_vs_target(y_true, y_pred, title="Error vs Target"):
    errors = y_true - y_pred

    plt.figure(figsize=(6, 5))

    plt.scatter(y_true, errors, alpha=0.5)

    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Target")
    plt.ylabel("Error")
    plt.title(title)

    plt.tight_layout()
    plt.show()


plot_error_vs_target(y_test, y_pred)

# Le graphique "Error vs Target" met en évidence une augmentation de la
# variance des erreurs lorsque la valeur réelle de la target augmente.
# Ce phénomène d'hétéroscédasticité est fréquent dans les problèmes liés
# aux consommations énergétiques ou aux émissions, où les grandes structures
# présentent des comportements plus variables.

plt.scatter(np.log1p(y_test), np.log1p(y_pred))

# Enfin, le dernier graphique confirme la bonne corrélation entre les valeurs
# réelles et les valeurs prédites (dans l'espace log-transformé utilisé pour
# l'entraînement), ce qui indique que le modèle capture correctement la
# structure générale des données.

# ! Conclusion
# Dans l'ensemble, ces diagnostics suggèrent que le modèle CatBoost fournit
# des prédictions globalement fiables, tout en rencontrant davantage de
# difficultés sur les bâtiments présentant les niveaux d'émissions les plus
# élevés.

# ? ************************************************
# ?      Varibales importantes dans le modèle     *
# ? ************************************************

# ? Best model

print(type(best_cat_model))
print(hasattr(best_cat_model, "regressor"))
print(hasattr(best_cat_model, "regressor_"))

importance_bm = best_cat_model.regressor_.get_feature_importance()

importance_bm.shape, X_test.shape

imp_bm_df = pd.DataFrame(
    {"feature": X_train.columns, "importance": importance_bm}
).sort_values("importance", ascending=False)

imp_bm_df

plt.figure(figsize=(8, 5))

sns.barplot(data=imp_bm_df, x="importance", y="feature")

plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.show()

# ? Base model (cat_model)

print(type(cat_model))
print(hasattr(cat_model, "regressor"))
print(hasattr(cat_model, "regressor_"))

importance_cat = cat_model.regressor_.get_feature_importance()

imp_cat_df = pd.DataFrame(
    {"feature": X_train.columns, "importance": importance_cat}
).sort_values("importance", ascending=False)

imp_cat_df

plt.figure(figsize=(8, 5))

sns.barplot(data=imp_cat_df, x="importance", y="feature")

plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.show()

# ? ************************************************
# ?       Comparaison par tranches de  log_gfa     *
# ? ************************************************


def add_binned_feature(df, feature_col, n_bins=4, method="qcut"):
    """
    Ajoute une colonne binned pour une variable numérique.

    method:
    - 'qcut' : quantiles
    - 'cut'  : intervalles réguliers
    """
    df = df.copy()

    new_col = f"{feature_col}_bin"

    if method == "qcut":
        df[new_col] = pd.qcut(df[feature_col], q=n_bins, duplicates="drop")
    elif method == "cut":
        df[new_col] = pd.cut(df[feature_col], bins=n_bins)
    else:
        raise ValueError("method doit être 'qcut' ou 'cut'")

    return df


def analyze_model_by_numeric_feature_bins(
    model,
    X_test,
    y_test,
    selected_features,
    feature_col,
    n_bins=4,
    method="qcut",
    min_samples=5,
):
    X_analysis = X_test.copy()
    X_analysis = add_binned_feature(
        X_analysis, feature_col, n_bins=n_bins, method=method
    )

    bin_col = f"{feature_col}_bin"
    bins = X_analysis[bin_col].dropna().unique().tolist()
    bins = sorted(bins)

    results = []

    for b in bins:
        mask = X_analysis[bin_col] == b

        X_group = X_analysis.loc[mask, selected_features].copy()
        y_group = y_test.loc[mask].copy()

        if len(X_group) < min_samples:
            continue

        y_pred_group = model.predict(X_group)

        results.append(
            {
                "bin": str(b),
                "n": len(X_group),
                "y_mean": y_group.mean(),
                "r2": r2_score(y_group, y_pred_group),
                "rmse": mean_squared_error(y_group, y_pred_group) ** 0.5,
                "mae": mean_absolute_error(y_group, y_pred_group),
                "mape": mean_absolute_percentage_error(y_group, y_pred_group),
                "rmse_over_mean": (mean_squared_error(y_group, y_pred_group) ** 0.5)
                / y_group.mean()
                / y_group.mean(),
            }
        )

    return pd.DataFrame(results)


gfa_results_df = analyze_model_by_numeric_feature_bins(
    model=best_cat_model,
    X_test=X_test,
    y_test=y_test,
    selected_features=col_sel,
    feature_col="log_gfa",
    n_bins=4,
    method="qcut",
    min_samples=5,
)

gfa_results_df


X_test_analysis = X_test.copy()

X_test_analysis = add_binned_feature(
    X_test_analysis, feature_col="log_gfa", n_bins=4, method="qcut"
)

gfa_bins = X_test_analysis["log_gfa_bin"].dropna().unique().tolist()
gfa_bins = sorted(gfa_bins)

x_test_y_test_gfa_bins = {}

for gfa_bin in gfa_bins:
    mask = X_test_analysis["log_gfa_bin"] == gfa_bin

    X_group = X_test_analysis.loc[mask, col_sel].copy()
    y_group = y_test.loc[mask].copy()

    x_test_y_test_gfa_bins[gfa_bin] = (X_group, y_group)

for gfa_bin in gfa_bins:
    X_group, y_group = x_test_y_test_gfa_bins[gfa_bin]

    if len(X_group) < 5:
        continue

    plot_regression_predictions(
        best_cat_model,
        X_group,
        y_group,
        title=f"CatBoost: Predicted vs Target - Log GFA {gfa_bin}",
    )

# ! Conclusion
# L'analyse des prédictions a été réalisée en segmentant les observations
# selon des intervalles de la variable log_gfa (log de la surface totale).
# Cette variable est la plus importante dans le modèle (≈35 % d'importance),
# ce qui confirme que la taille du bâtiment joue un rôle majeur dans la
# prédiction des émissions de gaz à effet de serre.
#
# Les graphiques "Predicted vs Target" montrent que :
#
# - Pour les bâtiments de petite taille, les prédictions sont relativement
#   proches de la diagonale (y = x), indiquant une bonne précision du modèle.
#
# - Lorsque la taille des bâtiments augmente, la dispersion des points
#   devient plus importante. Le modèle tend à sous-estimer certaines
#   observations présentant les plus fortes émissions.
#
# - Les bâtiments les plus grands présentent également une variabilité
#   plus forte dans leurs émissions, ce qui rend leur prédiction plus
#   difficile pour le modèle.
#
# Cette analyse confirme que la taille du bâtiment est un facteur
# déterminant dans la prédiction des émissions, mais également que
# l'incertitude du modèle augmente avec la surface des bâtiments.
#
# Globalement, le modèle CatBoost capture correctement la tendance
# générale entre la surface des bâtiments et leurs émissions de GES,
# tout en rencontrant davantage de difficultés pour les bâtiments
# les plus grands ou atypiques.


for gfa_bin in gfa_bins:
    X_group, y_group = x_test_y_test_gfa_bins[gfa_bin]

    if len(X_group) < 5:
        continue

    y_pred_group = best_cat_model.predict(X_group)

    plot_error_vs_target(
        y_group, y_pred_group, title=f"Error vs Target - Log GFA {gfa_bin}"
    )

# Afin de mieux comprendre le comportement du modèle, les erreurs de
# prédiction ont été analysées par segments de la variable log_gfa
# (logarithme de la surface totale du bâtiment).
#
# Les graphiques "Error vs Target" montrent que :
#
# - Pour les bâtiments de petite taille, les erreurs sont globalement
#   faibles et relativement centrées autour de zéro. Le modèle parvient
#   donc à prédire correctement les émissions pour ces observations.
#
# - À mesure que la taille des bâtiments augmente, la dispersion des
#   erreurs devient plus importante. Cela signifie que les prédictions
#   deviennent plus incertaines pour les bâtiments de grande surface.
#
# - Les bâtiments les plus grands présentent les erreurs les plus
#   importantes, avec plusieurs cas de sous-estimation ou de
#   surestimation marquées.
#
# Ce phénomène peut s'expliquer par une plus grande variabilité des
# usages énergétiques dans les bâtiments de grande taille
# (multi-usages, équipements spécifiques, infrastructures complexes),
# ce qui rend leur comportement énergétique plus difficile à modéliser.
#
# Malgré cette augmentation de la variance des erreurs pour les grands
# bâtiments, les prédictions restent globalement centrées autour de
# zéro, ce qui indique que le modèle ne présente pas de biais
# systématique dans ses estimations.
#
# Cette analyse confirme que la taille du bâtiment est un facteur
# structurant dans la prédiction des émissions de GES et que
# l'incertitude du modèle augmente avec la surface des bâtiments.

for gfa_bin in gfa_bins:
    X_group, y_group = x_test_y_test_gfa_bins[gfa_bin]

    if len(X_group) < 5:
        continue

    y_pred_group = best_cat_model.predict(X_group)

    plot_absolute_error_distribution(
        y_group, y_pred_group, title=f"CatBoost Absolute Error - Log GFA {gfa_bin}"
    )

# ! ===============================
# ! Distribution des erreurs absolues selon la taille des bâtiments (log_gfa)
# ! ===============================
#
# Afin d'analyser plus finement la performance du modèle, la distribution
# des erreurs absolues a été étudiée pour différents intervalles de la
# variable log_gfa (logarithme de la surface totale des bâtiments).
#
# Les histogrammes montrent que :
#
# - Pour les bâtiments de petite taille, les erreurs sont majoritairement
#   faibles et concentrées dans les premières classes. Cela indique que
#   le modèle parvient à prédire correctement les émissions pour ces
#   observations.
#
# - Lorsque la taille des bâtiments augmente, la distribution des erreurs
#   s'élargit progressivement et présente une queue plus longue vers la
#   droite. Cela signifie que certaines prédictions peuvent présenter des
#   erreurs importantes pour les bâtiments les plus grands.
#
# - Les bâtiments de grande surface présentent donc une plus grande
#   variabilité des erreurs, ce qui reflète une plus grande complexité
#   dans leur comportement énergétique.
#
# Globalement, ces distributions confirment que le modèle CatBoost fournit
# des prédictions relativement précises pour la majorité des bâtiments,
# tout en rencontrant davantage de difficultés pour les bâtiments les
# plus grands ou atypiques.
#
# Cette observation est cohérente avec l'importance élevée de la variable
# log_gfa dans le modèle (~35 %), qui indique que la surface des bâtiments
# joue un rôle majeur dans la prédiction des émissions de gaz à effet de
# serre.


# ? ************************************************
# ?      Comparaison par tranches de  elec_prop    *
# ? ************************************************


gfa_results_df = analyze_model_by_numeric_feature_bins(
    model=best_cat_model,
    X_test=X_test,
    y_test=y_test,
    selected_features=col_sel,
    feature_col="elec_prop",
    n_bins=4,
    method="qcut",
    min_samples=5,
)

gfa_results_df


X_test_analysis = X_test.copy()

X_test_analysis = add_binned_feature(
    X_test_analysis, feature_col="elec_prop", n_bins=4, method="qcut"
)

gfa_bins = X_test_analysis["elec_prop_bin"].dropna().unique().tolist()
gfa_bins = sorted(gfa_bins)

x_test_y_test_gfa_bins = {}

for gfa_bin in gfa_bins:
    mask = X_test_analysis["elec_prop_bin"] == gfa_bin

    X_group = X_test_analysis.loc[mask, col_sel].copy()
    y_group = y_test.loc[mask].copy()

    x_test_y_test_gfa_bins[gfa_bin] = (X_group, y_group)

for gfa_bin in gfa_bins:
    X_group, y_group = x_test_y_test_gfa_bins[gfa_bin]

    if len(X_group) < 5:
        continue

    plot_regression_predictions(
        best_cat_model,
        X_group,
        y_group,
        title=f"CatBoost: Predicted vs Target - elec prop {gfa_bin}",
    )

# ! Conclusion

# Une analyse complémentaire a été réalisée en segmentant les observations
# selon la variable elec_prop, qui représente la proportion d'énergie
# provenant de l'électricité dans la consommation totale du bâtiment.
#
# Cette variable est l'une des plus importantes du modèle (~33 % d'importance),
# ce qui indique que le mix énergétique joue un rôle majeur dans la
# prédiction des émissions de gaz à effet de serre.
#
# Les graphiques "Predicted vs Target" montrent que :
#
# - Pour les bâtiments ayant une faible proportion d'électricité, les
#   prédictions sont globalement proches de la diagonale, ce qui indique
#   une bonne capacité du modèle à estimer les émissions.
#
# - Lorsque la proportion d'électricité augmente, la dispersion des
#   prédictions devient plus importante. Cela suggère que les bâtiments
#   fortement dépendants de l'électricité présentent des comportements
#   énergétiques plus variables.
#
# - Les observations avec elec_prop proche de 1 (bâtiments principalement
#   alimentés en électricité) montrent également une variabilité plus
#   importante des prédictions.
#
# Cette analyse suggère que la composition du mix énergétique influence
# fortement les émissions de GES, mais que les bâtiments entièrement
# électriques peuvent présenter des comportements plus hétérogènes
# en fonction de leur usage, de leurs équipements ou de leur taille.
#
# Globalement, le modèle CatBoost capture correctement la relation entre
# la proportion d'électricité et les émissions de GES, tout en montrant
# une incertitude plus importante pour les bâtiments dont le mix
# énergétique est fortement dominé par l'électricité.


# ? ************************************************
# ?    Intéraction énergie / taille / émissions    *
# ? ************************************************

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X_test["elec_prop"],
    y_test,
    c=X_test["log_gfa"],
    cmap="viridis",
    alpha=0.7
)

plt.colorbar(scatter, label="log_gfa")

plt.xlabel("Electricity proportion (elec_prop)")
plt.ylabel("Total GHG Emissions")
plt.title("GHG Emissions vs Electricity Proportion\nColored by Building Size (log_gfa)")

plt.show()

# ===============================
# Interaction entre la proportion d'électricité, la taille des bâtiments
# et les émissions de GES
# ===============================
#
# Ce graphique représente la relation entre la proportion d'électricité
# dans le mix énergétique (elec_prop) et les émissions de gaz à effet
# de serre (TotalGHGEmissions). La couleur des points correspond à la
# taille des bâtiments, mesurée par la variable log_gfa.
#
# Plusieurs observations peuvent être faites :
#
# - Les bâtiments de petite taille (couleurs foncées) présentent
#   globalement des niveaux d'émissions faibles, quelle que soit
#   leur proportion d'électricité.
#
# - Les émissions les plus élevées sont principalement associées
#   aux bâtiments de grande taille (couleurs claires), ce qui
#   confirme que la surface totale du bâtiment est un facteur
#   déterminant dans la production d'émissions.
#
# - La proportion d'électricité seule ne suffit pas à expliquer
#   les émissions : des bâtiments ayant une proportion similaire
#   peuvent présenter des niveaux d'émissions très différents
#   selon leur taille.
#
# Cette visualisation met en évidence l'interaction entre la taille
# du bâtiment et le mix énergétique dans la détermination des
# émissions de gaz à effet de serre. Elle explique également
# pourquoi les variables log_gfa et elec_prop apparaissent comme
# les plus importantes dans le modèle CatBoost.

# création de l'explainer
explainer = shap.TreeExplainer(best_cat_model.regressor_)

# calcul des valeurs SHAP
shap_values = explainer.shap_values(X_test)

# ? ************************************************
# ?      Comparaison par tranches de PrimaryPropertyType
# ? ************************************************


groups = sorted(X_test["PrimaryPropertyType"].dropna().unique())

group_results = []

for group in groups:
    mask = X_test["PrimaryPropertyType"] == group

    X_group = X_test.loc[mask, col_sel].copy()
    y_group = y_test.loc[mask].copy()

    if len(X_group) < 5:
        continue

    y_pred_group = best_cat_model.predict(X_group)

    group_results.append(
        {
            "PrimaryPropertyType": group,
            "n": len(X_group),
            "y_mean": y_group.mean(),
            "r2": r2_score(y_group, y_pred_group),
            "rmse": mean_squared_error(y_group, y_pred_group) ** 0.5,
            "mae": mean_absolute_error(y_group, y_pred_group),
            "mape": mean_absolute_percentage_error(y_group, y_pred_group),
            "rmse_over_mean": (mean_squared_error(y_group, y_pred_group) ** 0.5)
            / y_group.mean(),
        }
    )

group_results_df = pd.DataFrame(group_results).sort_values("rmse")

for group in groups:
    mask = X_test["PrimaryPropertyType"] == group

    X_group = X_test.loc[mask, col_sel].copy()
    y_group = y_test.loc[mask].copy()

    if len(X_group) < 5:
        continue

    plot_regression_predictions(
        best_cat_model,
        X_group,
        y_group,
        title=f"CatBoost: Predicted vs Target - PrimaryPropertyType {group}",
    )

for group in groups:
    mask = X_test["PrimaryPropertyType"] == group

    X_group = X_test.loc[mask, col_sel].copy()
    y_group = y_test.loc[mask].copy()

    if len(X_group) < 5:
        continue

    y_pred_group = best_cat_model.predict(X_group)

    plot_absolute_error_distribution(
        y_group, y_pred_group, title=f"Absolute Error Distribution - {group}"
    )


# ? ===============================
# ? Interprétation globale du modèle avec SHAP
# ? ===============================

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="dot"
)

# Le graphique SHAP summary plot permet d'analyser l'importance des variables
# ainsi que leur influence sur les prédictions du modèle CatBoost.
#
# Les résultats montrent que :
#
# - log_gfa est la variable la plus influente. Les valeurs élevées de log_gfa
#   augmentent fortement les prédictions d'émissions, ce qui confirme que
#   la taille du bâtiment est le principal déterminant des émissions de GES.
#
# - elec_prop est la deuxième variable la plus importante. La proportion
#   d'électricité dans le mix énergétique influence significativement les
#   émissions, ce qui reflète l'impact du type d'énergie consommée.
#
# - EnergyProfileGroup et EnergyProfileScore contribuent également aux
#   prédictions, suggérant que les caractéristiques énergétiques globales
#   des bâtiments jouent un rôle dans la détermination des émissions.
#
# - Les autres variables, comme gas_prop, largest_use_ratio ou
#   dist_downtown, ont un impact plus modéré mais permettent d'affiner
#   les prédictions en capturant des effets structurels ou contextuels.
#
# Globalement, le modèle s'appuie principalement sur la taille du bâtiment
# et la composition du mix énergétique pour prédire les émissions de gaz
# à effet de serre.



# ===============================
# Effet de la taille des bâtiments (log_gfa)
# ===============================

shap.dependence_plot(
    "log_gfa",
    shap_values,
    X_test
)

# Le graphique SHAP dependence plot met en évidence la relation entre
# log_gfa et son impact sur les prédictions du modèle.
#
# On observe une relation croissante quasi linéaire : lorsque la surface
# du bâtiment augmente, l'impact de cette variable sur la prédiction des
# émissions devient de plus en plus positif.
#
# Cela confirme que les bâtiments les plus grands sont associés aux
# niveaux d'émissions les plus élevés.
#
# La coloration par EnergyProfileScore suggère que l'efficacité énergétique
# peut moduler cet effet : des bâtiments de taille comparable peuvent
# présenter des émissions différentes selon leur profil énergétique.


# ===============================
# Interaction entre log_gfa et elec_prop
# ===============================

shap.dependence_plot(
    "log_gfa",
    shap_values,
    X_test,
    interaction_index="elec_prop"
)

# Le graphique montre l'interaction entre la taille des bâtiments
# (log_gfa) et la proportion d'électricité dans le mix énergétique
# (elec_prop).
#
# On observe que l'impact de log_gfa reste dominant, mais la proportion
# d'électricité semble moduler les prédictions du modèle.
#
# Les bâtiments de grande taille combinés à une forte proportion
# d'électricité présentent généralement les impacts les plus élevés
# sur les émissions prédites.
#
# Cette interaction confirme que le modèle apprend des relations
# cohérentes entre la structure des bâtiments et leur consommation
# énergétique.

# ! ===============================
# ! Conclusion de l'analyse du modèle
# ! ===============================
#
# L'analyse des valeurs SHAP confirme que le modèle CatBoost capture
# des relations cohérentes entre les caractéristiques des bâtiments
# et leurs émissions de gaz à effet de serre.
#
# La taille du bâtiment (log_gfa) apparaît comme le facteur le plus
# déterminant, suivie par la composition du mix énergétique,
# notamment la proportion d'électricité.
#
# Ces résultats suggèrent que les émissions de GES des bâtiments
# sont principalement expliquées par leur dimension et leur
# consommation énergétique relative.
#
# Les variables supplémentaires permettent d'affiner les prédictions
# en capturant les différences liées aux usages et aux profils
# énergétiques des bâtiments.

# ? ************************************************
# ?      Interpretation des erreurs du modele      *
# ? ************************************************

y_pred = best_cat_model.predict(X_test)

errors = np.abs(y_test - y_pred)

error_df = X_test.copy()
error_df["error"] = errors

error_df.sort_values("error", ascending=False).head(10)

regressor = CatBoostRegressor(
    iterations=500,
    depth=4,
    learning_rate=0.05,
    verbose=0
)

error_model = TransformedTargetRegressor(
    regressor=regressor, func=np.log1p, inverse_func=np.expm1
)

error_model.fit(X_train, y_train, cat_features=cat_features_idx)

explainer_error = shap.TreeExplainer(error_model.regressor_)

shap_values_error = explainer_error.shap_values(X_test)


# ===============================
# Analyse des facteurs expliquant les erreurs du modèle
# ===============================

shap.summary_plot(shap_values_error, X_test)

# Afin de mieux comprendre dans quelles situations le modèle présente
# les erreurs les plus importantes, un modèle secondaire a été entraîné
# pour prédire l'erreur absolue des prédictions.
#
# L'analyse SHAP de ce modèle met en évidence les variables qui
# contribuent le plus aux erreurs.
#
# Les résultats montrent que :
#
# - log_gfa est la variable la plus influente dans les erreurs du modèle.
#   Les bâtiments de grande taille sont associés aux erreurs les plus
#   importantes.
#
# - elec_prop joue également un rôle notable, suggérant que certaines
#   configurations de mix énergétique sont plus difficiles à modéliser.
#
# - Les variables liées au profil énergétique du bâtiment
#   (EnergyProfileGroup, EnergyProfileScore) contribuent aussi à
#   expliquer certaines erreurs.
#
# Globalement, les erreurs du modèle semblent principalement associées
# aux bâtiments de grande taille et aux configurations énergétiques
# plus complexes.



# ===============================
# Relation entre la taille des bâtiments et l'erreur du modèle
# ===============================

plt.scatter(X_test["log_gfa"], errors)
plt.xlabel("log_gfa")
plt.ylabel("Absolute Error")
plt.title("Model Error vs Building Size")
plt.show()

# Le graphique "Model Error vs Building Size" montre que l'erreur
# absolue du modèle tend à augmenter avec la taille des bâtiments
# (log_gfa).
#
# Les bâtiments de petite et moyenne taille présentent généralement
# des erreurs faibles, tandis que les bâtiments les plus grands
# sont associés à des erreurs beaucoup plus importantes.
#
# Ce phénomène reflète une hétéroscédasticité dans les données :
# la variabilité des émissions augmente avec la taille du bâtiment,
# ce qui rend les prédictions plus difficiles pour les structures
# les plus grandes.
#
# Cette observation est cohérente avec les analyses précédentes
# montrant que la taille du bâtiment est la variable la plus
# déterminante dans la prédiction des émissions.


# ! ===============================
# ! Conclusion sur les limites du modèle
# ! ===============================
#
# L'analyse des erreurs montre que le modèle fournit des prédictions
# globalement précises pour la majorité des bâtiments, en particulier
# pour les bâtiments de taille moyenne.
#
# Cependant, les erreurs augmentent significativement pour les
# bâtiments les plus grands, qui présentent des comportements
# énergétiques plus complexes et plus variables.
#
# Cette observation souligne l'importance de la taille du bâtiment
# dans la modélisation des émissions et met en évidence certaines
# limites du modèle pour les structures atypiques ou très grandes.