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
df.shape

# *****************************************
# *        SELECTION DES FEATURES         *
# *****************************************

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

# ! R² : 0.830 (train) et  0.717  (test)
# ! RMSE : 6.783e+06
# ! MAE : 3.102e+06

# *************************************************
# *      Parte 2 :: modele avec early stopping    *
# *************************************************

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=seed
)

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
# ! R² : 0.802 (train) et  0.697  (test)
# ! RMSE : 6.684e+06
# ! MAE : 2.965e+06

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
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print(
    f"R² : {best_model.score(X_train, y_train):.3f} (train) et {colorama.Style.BRIGHT}{colorama.Back.CYAN}{colorama.Fore.BLACK} {best_model.score(X_test, y_test):.3f} {colorama.Style.RESET_ALL} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

best_param_random = random_search.best_params_

cat_random_params = {
    k.replace("regressor__", ""): v for k, v in best_param_random.items()
}

print("CatBoost params nettoyés :", cat_random_params)

#! R² : 0.846 (train) et  0.290  (test)
#! RMSE : 199.4
#! MAE : 2.97e+06

# ? Params
# {'bagging_temperature': np.float64(0.8207658460712595),
# 'depth': 5,
# 'iterations': 1040,
# 'l2_leaf_reg': 5,
# 'learning_rate': np.float64(0.09485682135021828),
# 'random_strength': np.float64(1.197730932977072)}

# *************************************************
# *                   BEST_MODELE                 *
# *************************************************

regressor = CatBoostRegressor(
    bagging_temperature=float(0.8207658460712595),
    depth=5,
    iterations=1040,
    l2_leaf_reg=5,
    learning_rate=float(0.09485682135021828),
    random_strength=float(1.197730932977072),
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
    f"R² : {best_cat_model.score(X_train, y_train):.3f} (train) et "
    f"{best_cat_model.score(X_test, y_test):.3f} (test)"
)
print(f"RMSE : {mean_squared_error(y_test, y_pred) ** 0.5:.4}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4}")

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

    regressor = CatBoostRegressor(
        random_state=seed,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=0,
    )

    model = TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1
    )

    model.fit(X_tr, y_tr, cat_features=cat_features_idx_fold)

    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)

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

# ? Le modèle CatBoost standard avec transformation logarithmique de la cible présente
# ? la meilleure capacité de généralisation selon le coefficient de détermination,
# ? avec un R² test de 0.717 contre 0.702 pour la version optimisée par RandomizedSearchCV.
# ? En revanche, le modèle optimisé obtient des erreurs absolues plus faibles,
# ? avec un RMSE de 6.63e+06 et un MAE de 2.97e+06, contre respectivement 6.783e+06 et 3.102e+06 pour le modèle standard.

# *************************************************
# *      Varibales importantes dans le modèle     *
# *************************************************

# ? Best model

print(type(best_model))
print(hasattr(best_model, "regressor"))
print(hasattr(best_model, "regressor_"))

importance_bm = best_model.regressor_.get_feature_importance()

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

# *************************************************
# *     Analyse des erreurs absolues du modèle    *
# *************************************************

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
    cat_model, X_test, y_test, title="CatBoost: Predicted vs Target"
)

plot_regression_predictions(
    best_model, X_train, y_train, title="CatBoost: Predicted vs Target"
)


def plot_regression_error(
    model, X, y, xlim=None, title="Distribution of Regression Errors"
):
    y_pred = model.predict(X)

    errors = y - y_pred

    plt.figure(figsize=(7, 5))

    sns.histplot(errors, bins=50, kde=True)

    plt.axvline(0, color="red", linestyle="--")

    if xlim is not None:
        plt.xlim(xlim)

    plt.xlabel("Error (y_true - y_pred)")
    plt.ylabel("Frequency")
    plt.title(title)

    plt.tight_layout()
    plt.show()


plot_regression_error(
    cat_model, X_test, y_test, xlim=(-1e7, 1e7), title="CatBoost Error Distribution"
)


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

plt.scatter(np.log1p(y_test), np.log1p(y_pred))

# *************************************************
# *     Comparaison par PrimaryPropertyGroup      *
# *************************************************

model = cat_model
y_pred = model.predict(X_test)


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


X_test_analysis = X_test.copy()

X_test_analysis = add_binned_feature(
    X_test_analysis, feature_col="PropertyGFATotal", n_bins=4, method="qcut"
)

print(X_test_analysis["PropertyGFATotal_bin"].value_counts(dropna=False))

gfa_bins = X_test_analysis["PropertyGFATotal_bin"].dropna().unique().tolist()
gfa_bins = sorted(gfa_bins)

print(gfa_bins)

x_test_y_test_gfa_bins = {}

for gfa_bin in gfa_bins:
    mask = X_test_analysis["PropertyGFATotal_bin"] == gfa_bin

    X_group = X_test_analysis.loc[mask, col_sel].copy()
    y_group = y_test.loc[mask].copy()

    x_test_y_test_gfa_bins[gfa_bin] = (X_group, y_group)

for gfa_bin in gfa_bins:
    X_group, y_group = x_test_y_test_gfa_bins[gfa_bin]

    if len(X_group) < 5:
        continue

    plot_regression_predictions(
        model,
        X_group,
        y_group,
        title=f"CatBoost: Predicted vs Target - PropertyGFATotal {gfa_bin}",
    )

for gfa_bin in gfa_bins:
    X_group, y_group = x_test_y_test_gfa_bins[gfa_bin]

    if len(X_group) < 5:
        continue

    plot_regression_error(
        model,
        X_group,
        y_group,
        xlim=(-1e7, 1e7),
        title=f"CatBoost Error Distribution - PropertyGFATotal {gfa_bin}",
    )

gfa_results = []

for gfa_bin in gfa_bins:
    X_group, y_group = x_test_y_test_gfa_bins[gfa_bin]

    if len(X_group) < 5:
        continue

    y_pred_group = model.predict(X_group)

    gfa_results.append(
        {
            "PropertyGFATotal_bin": str(gfa_bin),
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

gfa_results_df = pd.DataFrame(gfa_results)
gfa_results_df


# ? RMSE
plt.figure(figsize=(10, 5))
sns.barplot(data=gfa_results_df, x="rmse", y="PropertyGFATotal_bin")
plt.title("RMSE by PropertyGFATotal bin")
plt.xlabel("RMSE")
plt.ylabel("PropertyGFATotal bin")
plt.tight_layout()
plt.show()

# ? MAE
plt.figure(figsize=(10, 5))
sns.barplot(data=gfa_results_df, x="mae", y="PropertyGFATotal_bin")
plt.title("MAE by PropertyGFATotal bin")
plt.xlabel("MAE")
plt.ylabel("PropertyGFATotal bin")
plt.tight_layout()
plt.show()

# ? R²
plt.figure(figsize=(10, 5))
sns.barplot(data=gfa_results_df, x="r2", y="PropertyGFATotal_bin")
plt.title("R² by PropertyGFATotal bin")
plt.xlabel("R²")
plt.ylabel("PropertyGFATotal bin")
plt.tight_layout()
plt.show()

# *************************************************
# *          Fonction pour comparaison            *
# *************************************************


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
    model=best_model,
    X_test=X_test,
    y_test=y_test,
    selected_features=col_sel,
    feature_col="PropertyGFATotal",
    n_bins=4,
    method="qcut",
    min_samples=5,
)

gfa_results_df

# *************************************************
# *     Comparaison par PrimaryPropertyGroup      *
# *************************************************

groups = sorted(X_test["PrimaryPropertyGroup"].dropna().unique())

group_results = []

for group in groups:
    mask = X_test["PrimaryPropertyGroup"] == group

    X_group = X_test.loc[mask, col_sel].copy()
    y_group = y_test.loc[mask].copy()

    if len(X_group) < 5:
        continue

    y_pred_group = best_model.predict(X_group)

    group_results.append(
        {
            "PrimaryPropertyGroup": group,
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
group_results_df.PrimaryPropertyGroup

for group in groups:
    mask = X_test["PrimaryPropertyGroup"] == group

    X_group = X_test.loc[mask, col_sel].copy()
    y_group = y_test.loc[mask].copy()

    if len(X_group) < 5:
        continue

    plot_regression_predictions(
        best_model, X_group, y_group, title=f"Predicted vs Target - {group}"
    )


# ? R²
plt.figure(figsize=(10, 5))
sns.barplot(data=group_results_df, x="r2", y="PrimaryPropertyGroup")
plt.title("R² by PrimaryPropertyGroup bin")
plt.xlabel("R²")
plt.ylabel("Primary Profil Group")
plt.tight_layout()
plt.show()

# ? RMSE
plt.figure(figsize=(10, 5))
sns.barplot(data=group_results_df, x="rmse", y="PrimaryPropertyGroup")
plt.title("R² by PrimaryPropertyGroup bin")
plt.xlabel("R²")
plt.ylabel("Primary Profil Group")
plt.tight_layout()
plt.show()

df.PrimaryPropertyGroup.value_counts()

# *************************************************
# *      Comparaison par EnergyProfileGroup       *
# *************************************************

groups = sorted(X_test["EnergyProfileGroup"].dropna().unique())

group_results = []

for group in groups:
    mask = X_test["EnergyProfileGroup"] == group

    X_group = X_test.loc[mask, col_sel].copy()
    y_group = y_test.loc[mask].copy()

    if len(X_group) < 5:
        continue

    y_pred_group = best_model.predict(X_group)

    group_results.append(
        {
            "EnergyProfileGroup": group,
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
    mask = X_test["EnergyProfileGroup"] == group

    X_group = X_test.loc[mask, col_sel].copy()
    y_group = y_test.loc[mask].copy()

    if len(X_group) < 5:
        continue

    plot_regression_predictions(
        best_model, X_group, y_group, title=f"Predicted vs Target - {group}"
    )


for group in groups:
    mask = X_test["EnergyProfileGroup"] == group

    X_group = X_test.loc[mask, col_sel].copy()
    y_group = y_test.loc[mask].copy()

    if len(X_group) < 5:
        continue

    plot_regression_error(
        best_model,
        X_group,
        y_group,
        xlim=(-1e7, 1e7),
        title=f"Distribution of Regression Errors - {group}",
    )

for group in groups:
    mask = X_test["EnergyProfileGroup"] == group

    X_group = X_test.loc[mask, col_sel].copy()
    y_group = y_test.loc[mask].copy()

    if len(X_group) < 5:
        continue

    y_pred_group = best_model.predict(X_group)

    plot_absolute_error_distribution(
        y_group, y_pred_group, title=f"Absolute Error Distribution - {group}"
    )
