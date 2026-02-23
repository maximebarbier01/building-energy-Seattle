import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# =========================
# 1) Différence de moyennes
# =========================

def cohens_d(x, y):
    """Cohen's d (2 groupes indépendants)."""
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / s_pooled

def hedges_g(x, y):
    """Hedges' g (Cohen's d corrigé petits échantillons)."""
    x = np.asarray(x); y = np.asarray(y)
    d = cohens_d(x, y)
    nx, ny = len(x), len(y)
    df = nx + ny - 2
    # Correction de biais (J)
    J = 1 - (3 / (4 * df - 1))
    return J * d


# =========================
# 2) Corrélations
# =========================

def pearson_r(x, y):
    """r de Pearson + p-value (r est une taille d'effet)."""
    r, p = stats.pearsonr(x, y)
    return r, p

def spearman_r(x, y):
    """rho de Spearman + p-value (rho est une taille d'effet)."""
    rho, p = stats.spearmanr(x, y)
    return rho, p

def r_squared_from_r(r):
    """r² : proportion de variance expliquée (en corrélation simple)."""
    return float(r) ** 2


# =========================
# 3) ANOVA : η², η² partiel, ω²
# =========================

def anova_effect_sizes(dataframe, dv, factor, typ=2):
    """
    Calcule eta², eta² partiel, omega² pour une ANOVA à 1 facteur:
    dv ~ C(factor)

    - dataframe : pd.DataFrame
    - dv : variable dépendante (numérique) (str)
    - factor : facteur catégoriel (str)
    - typ : type de sommes des carrés (2 recommandé en pratique)
    """
    df = dataframe.copy()
    model = ols(f"{dv} ~ C({factor})", data=df).fit()
    aov = sm.stats.anova_lm(model, typ=typ)

    # Récupère les lignes
    effect_row = aov.index[0]          # "C(factor)"
    error_row = aov.index[-1]          # "Residual"
    ss_effect = aov.loc[effect_row, "sum_sq"]
    df_effect = aov.loc[effect_row, "df"]
    ss_error  = aov.loc[error_row, "sum_sq"]
    df_error  = aov.loc[error_row, "df"]
    ms_error  = ss_error / df_error # pyright: ignore[reportOperatorIssue]
    ss_total  = ss_effect + ss_error

    eta2 = ss_effect / ss_total
    partial_eta2 = ss_effect / (ss_effect + ss_error)

    # Omega squared (ω²) pour ANOVA à 1 facteur
    omega2 = (ss_effect - df_effect * ms_error) / (ss_total + ms_error)

    return {
        "eta2": float(eta2),
        "partial_eta2": float(partial_eta2),
        "omega2": float(omega2),
        "anova_table": aov
    }


# =========================
# 4) Régression : R², R² ajusté
# =========================

def r2_and_adj_r2_from_model(y, X):
    """
    R² et R² ajusté via statsmodels (régression linéaire).
    - y : array-like (n,)
    - X : array-like (n, p) (sans constante)
    """
    X = sm.add_constant(X)  # ajoute l'intercept
    model = sm.OLS(y, X).fit()
    return float(model.rsquared), float(model.rsquared_adj), model

# Variante formule (si tu as déjà R², n et p)
def adjusted_r2(r2, n, p):
    """
    R² ajusté via formule.
    - n : nb d'observations
    - p : nb de variables explicatives (sans la constante)
    """
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


# =========================
# 5) Catégorielles (Chi²) : Cramér's V
# =========================

def cramers_v(x, y):
    """
    Cramér's V entre deux variables catégorielles.
    x, y : array-like de catégories
    """
    table = pd.crosstab(x, y)
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    r, k = table.shape
    V = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    return float(V), float(p), table
