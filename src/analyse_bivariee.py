import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kruskal, chi2_contingency

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _is_categorical(s: pd.Series, max_unique_ratio: float = 0.05, max_unique: int = 50) -> bool:
    """
    Heuristique: object/category => catégoriel.
    Int avec peu de valeurs distinctes => catégoriel (ex: code district).
    """
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_bool_dtype(s):
        return True
    # Cas des ints "codes"
    if pd.api.types.is_integer_dtype(s):
        nunique = s.nunique(dropna=True)
        ratio = nunique / max(len(s.dropna()), 1)
        return (nunique <= max_unique) or (ratio <= max_unique_ratio)
    return False

def _interpret_p(p: float, alpha: float = 0.05) -> str:
    if np.isnan(p):
        return "p-value = NaN (test invalide: données manquantes/groupes vides/valeurs constantes)."
    if p < 0.001:
        return "Différence/association statistiquement significative (p < 0.001)."
    if p < alpha:
        return f"Différence/association statistiquement significative (p < {alpha})."
    return f"Pas de preuve statistique d’association (p ≥ {alpha})."

def _interpret_eta2(eta2: float) -> str:
    # Règles de pouce (Cohen-ish)
    if np.isnan(eta2):
        return "Taille d’effet non calculable."
    if eta2 < 0.01:
        return "Taille d’effet négligeable (η² < 0.01)."
    if eta2 < 0.06:
        return "Petit effet (η² ~ 0.01–0.06)."
    if eta2 < 0.14:
        return "Effet moyen (η² ~ 0.06–0.14)."
    return "Grand effet (η² ≥ 0.14)."

def _interpret_corr(r: float) -> str:
    ar = abs(r)
    if ar < 0.1:
        return "Corrélation négligeable."
    if ar < 0.3:
        return "Faible corrélation."
    if ar < 0.5:
        return "Corrélation modérée."
    return "Forte corrélation."

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    cm = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(cm)
    n = cm.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = cm.shape
    phi2 = chi2 / n
    denom = min(k - 1, r - 1)
    return np.sqrt(phi2 / denom) if denom > 0 else np.nan

def association_report(
    df: pd.DataFrame,
    var_x: str,
    var_y: str,
    alpha: float = 0.05,
    corr_method_num_num: str = "spearman",
    dropna: bool = True,
    min_group_size: int = 3
) -> dict:
    """
    Retourne un dict avec test, p-value, taille d'effet et interprétation.
    - num vs num: Spearman (par défaut) ou Pearson
    - cat vs num: Kruskal-Wallis + eta_squared
    - cat vs cat: Chi2 + Cramér's V
    """
    x = df[var_x]
    y = df[var_y]

    # Drop NA en paire pour num-num, sinon on gère par groupe pour cat-num
    if dropna:
        d = df[[var_x, var_y]].dropna()
        x_clean, y_clean = d[var_x], d[var_y]
    else:
        x_clean, y_clean = x, y

    x_cat = _is_categorical(x_clean)
    y_cat = _is_categorical(y_clean)
    x_num = _is_numeric(x_clean)
    y_num = _is_numeric(y_clean)

    result = {
        "var_x": var_x,
        "var_y": var_y,
        "test": None,
        "p_value": np.nan,
        "effect_size": np.nan,
        "effect_size_name": None,
        "interpretation": ""
    }

    # --- NUM vs NUM ---
    if x_num and y_num and (not x_cat) and (not y_cat):
        method = corr_method_num_num.lower()
        if method == "pearson":
            r, p = pearsonr(x_clean, y_clean)
            test_name = "Pearson correlation"
        else:
            r, p = spearmanr(x_clean, y_clean, nan_policy="omit")
            test_name = "Spearman correlation"

        result.update({
            "test": test_name,
            "p_value": float(p),
            "effect_size": float(r),
            "effect_size_name": "r" if method == "pearson" else "rho"
        })

        interp = _interpret_p(p, alpha) + " " + _interpret_corr(r)
        result["interpretation"] = interp
        return result

    # --- CAT vs NUM (dans un sens ou l’autre) ---
    # On force: cat = group, num = outcome
    if (x_cat and y_num) or (y_cat and x_num):
        if x_cat and y_num:
            group_var, value_var = var_x, var_y
        else:
            group_var, value_var = var_y, var_x

        # Construire les groupes en dropna sur la valeur
        groups = []
        k = 0
        N = 0
        for name, g in df.groupby(group_var):
            vals = g[value_var].dropna().values
            if len(vals) >= min_group_size:
                groups.append(vals)
                k += 1
                N += len(vals)

        if k < 2:
            result["test"] = "Kruskal-Wallis"
            result["interpretation"] = (
                "Impossible de lancer Kruskal-Wallis : moins de 2 groupes valides "
                f"(min_group_size={min_group_size})."
            )
            return result

        H, p = kruskal(*groups)

        # eta squared pour Kruskal (approx) : (H - k + 1) / (N - k)
        eta2 = (H - k + 1) / (N - k) if (N - k) > 0 else np.nan
        eta2 = max(0.0, float(eta2)) if not np.isnan(eta2) else np.nan

        result.update({
            "test": f"Kruskal-Wallis ({group_var} → {value_var})",
            "p_value": float(p),
            "effect_size": eta2,
            "effect_size_name": "eta_squared"
        })

        result["interpretation"] = _interpret_p(p, alpha) + " " + _interpret_eta2(eta2)
        return result

    # --- CAT vs CAT ---
    if x_cat and y_cat:
        cm = pd.crosstab(x_clean, y_clean)
        if cm.size == 0:
            result["test"] = "Chi-square"
            result["interpretation"] = "Table de contingence vide (pas assez de données)."
            return result

        chi2, p, _, _ = chi2_contingency(cm)
        v = _cramers_v(x_clean, y_clean)

        result.update({
            "test": "Chi-square independence",
            "p_value": float(p),
            "effect_size": float(v),
            "effect_size_name": "cramers_v"
        })

        # Interprétation simple pour V (seuils proches de ceux de r)
        result["interpretation"] = _interpret_p(p, alpha) + " " + _interpret_corr(v)
        return result

    # Fallback
    result["interpretation"] = (
        "Types de variables difficiles à déterminer automatiquement. "
        "Vérifie les dtypes / le nombre de valeurs uniques."
    )
    return result
