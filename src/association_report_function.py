import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kruskal, chi2_contingency
from typing import Dict, Optional, Literal


def _safe_float(x) -> float:
    """Convertit en float de manière sécurisée."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _is_constant(s: pd.Series) -> bool:
    """Vérifie si une série est constante (une seule valeur unique)."""
    s_clean = s.dropna()
    return len(s_clean) == 0 or s_clean.nunique() <= 1


def _is_categorical(s: pd.Series, max_unique: int = 50) -> bool:
    """
    Détermine si une série doit être traitée comme catégorielle.
    
    Critères :
    - Type categorical, bool, object ou string
    - Entiers avec peu de valeurs uniques
    """
    # Types explicitement catégoriels
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == "bool":
        return True
    
    # Object et string (anciens et nouveaux types pandas)
    dtype_name = s.dtype.name
    if dtype_name in ("object", "str", "string"):
        return True
    
    # Vérification supplémentaire pour les variantes de string
    dtype_str = str(s.dtype).lower()
    if "string" in dtype_str or dtype_str == "str":
        return True
    
    # Entiers avec peu de modalités
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) <= max_unique
    
    return False


def _interpret_p_value(p: float, alpha: float = 0.05) -> str:
    """Interprète la p-value."""
    if np.isnan(p):
        return "⚠️ p-value non calculable (données insuffisantes ou problème)"
    if p < 0.001:
        return "✓ Très significatif (p < 0.001)"
    if p < alpha:
        return f"✓ Significatif (p < {alpha})"
    return f"✗ Non significatif (p ≥ {alpha})"


def _interpret_correlation(r: float) -> str:
    """Interprète la force d'une corrélation (Pearson, Spearman, Cramér's V)."""
    if np.isnan(r):
        return "Effet non calculable"
    
    abs_r = abs(r)
    if abs_r < 0.1:
        return "Effet négligeable"
    elif abs_r < 0.3:
        return "Effet faible"
    elif abs_r < 0.5:
        return "Effet modéré"
    else:
        return "Effet fort"


def _interpret_eta_squared(eta2: float) -> str:
    """Interprète eta² (taille d'effet pour Kruskal-Wallis)."""
    if np.isnan(eta2):
        return "Effet non calculable"
    
    if eta2 < 0.01:
        return "Effet négligeable (η² < 0.01)"
    elif eta2 < 0.06:
        return "Petit effet (0.01 ≤ η² < 0.06)"
    elif eta2 < 0.14:
        return "Effet moyen (0.06 ≤ η² < 0.14)"
    else:
        return "Grand effet (η² ≥ 0.14)"


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calcule le V de Cramér pour mesurer l'association entre variables catégorielles.
    Formule : V = sqrt(χ² / (n × min(r-1, c-1)))
    """
    try:
        contingency = pd.crosstab(x, y)
        
        if contingency.size == 0:
            return np.nan
        
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        r, c = contingency.shape
        
        denom = min(r - 1, c - 1)
        
        if n == 0 or denom <= 0:
            return np.nan
        
        v = np.sqrt((chi2 / n) / denom)
        return float(v)
    
    except Exception:
        return np.nan


def analyze_association(
    df: pd.DataFrame,
    var_x: str,
    var_y: str,
    alpha: float = 0.05,
    corr_method: Literal["pearson", "spearman"] = "spearman",
    max_unique_categorical: int = 50,
    min_group_size: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Analyse l'association entre deux variables avec sélection automatique du test.
    
    Tests utilisés :
    - Num × Num → Pearson ou Spearman
    - Cat × Num → Kruskal-Wallis (+ eta²)
    - Cat × Cat → Chi² (+ V de Cramér)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données
    var_x, var_y : str
        Noms des variables à analyser
    alpha : float, default=0.05
        Seuil de significativité
    corr_method : {'pearson', 'spearman'}, default='spearman'
        Méthode de corrélation pour variables numériques
    max_unique_categorical : int, default=50
        Nombre max de valeurs uniques pour considérer un entier comme catégoriel
    min_group_size : int, default=3
        Taille minimale d'un groupe pour Kruskal-Wallis
    verbose : bool, default=True
        Afficher un résumé textuel
    
    Returns
    -------
    dict
        Dictionnaire avec les résultats du test
    """
    
    # === VÉRIFICATIONS INITIALES ===
    if var_x not in df.columns:
        raise KeyError(f"La variable '{var_x}' n'existe pas dans le DataFrame")
    if var_y not in df.columns:
        raise KeyError(f"La variable '{var_y}' n'existe pas dans le DataFrame")
    
    result = {
        "var_x": var_x,
        "var_y": var_y,
        "dtype_x": str(df[var_x].dtype),
        "dtype_y": str(df[var_y].dtype),
        "test": None,
        "p_value": np.nan,
        "effect_size": np.nan,
        "effect_size_name": None,
        "n_total": len(df),
        "n_used": 0,
        "n_missing": df[[var_x, var_y]].isna().any(axis=1).sum(),
        "k_groups": None,
        "interpretation_p": "",
        "interpretation_effect": "",
    }
    
    # Suppression des valeurs manquantes
    data = df[[var_x, var_y]].dropna()
    
    if data.empty:
        result["interpretation_p"] = "❌ Aucune donnée valide (toutes les lignes ont des NaN)"
        if verbose:
            print(f"\n{'='*60}\n⚠️  ANALYSE IMPOSSIBLE\n{'='*60}")
            print(f"Variables : {var_x} × {var_y}")
            print(f"Raison : Aucune observation sans valeur manquante")
        return result
    
    result["n_used"] = len(data)
    
    # === DÉTECTION DU TYPE DE VARIABLE ===
    x_series = data[var_x]
    y_series = data[var_y]
    
    x_is_cat = _is_categorical(x_series, max_unique_categorical)
    y_is_cat = _is_categorical(y_series, max_unique_categorical)
    
    x_is_num = pd.api.types.is_numeric_dtype(x_series) and not x_is_cat
    y_is_num = pd.api.types.is_numeric_dtype(y_series) and not y_is_cat
    
    # === CAS 1 : NUMÉRIQUE × NUMÉRIQUE ===
    if x_is_num and y_is_num:
        if len(data) < 3:
            result["test"] = f"Corrélation ({corr_method})"
            result["interpretation_p"] = "❌ Pas assez de données (n < 3)"
            if verbose:
                print(f"\n{'='*60}\n⚠️  DONNÉES INSUFFISANTES\n{'='*60}")
                print(f"Variables : {var_x} (num) × {var_y} (num)")
                print(f"Observations utilisées : {len(data)}")
            return result
        
        if _is_constant(x_series) or _is_constant(y_series):
            result["test"] = f"Corrélation ({corr_method})"
            result["interpretation_p"] = "❌ Variable constante détectée"
            result["interpretation_effect"] = "Corrélation non définie"
            if verbose:
                print(f"\n{'='*60}\n⚠️  VARIABLE CONSTANTE\n{'='*60}")
                print(f"Variables : {var_x} × {var_y}")
                const_var = var_x if _is_constant(x_series) else var_y
                print(f"Variable constante : {const_var}")
            return result
        
        # Calcul de la corrélation
        if corr_method.lower() == "pearson":
            r, p = pearsonr(x_series, y_series)
            result["test"] = "Corrélation de Pearson"
            result["effect_size_name"] = "r"
        else:
            r, p = spearmanr(x_series, y_series)
            result["test"] = "Corrélation de Spearman"
            result["effect_size_name"] = "rho"
        
        result["p_value"] = round(_safe_float(p),4)
        result["effect_size"] = round(_safe_float(r),4)
        result["interpretation_p"] = _interpret_p_value(result["p_value"], alpha)
        result["interpretation_effect"] = _interpret_correlation(result["effect_size"])
    
    # === CAS 2 : CATÉGORIEL × NUMÉRIQUE ===
    elif (x_is_cat and y_is_num) or (y_is_cat and x_is_num):
        # Identifier quelle variable est le groupe
        if x_is_cat:
            group_var, value_var = var_x, var_y
        else:
            group_var, value_var = var_y, var_x
        
        # Préparation des groupes
        valid_groups = []
        group_names = []
        
        for name, group_df in data.groupby(group_var):
            values = group_df[value_var].dropna().values
            
            # Vérifier taille et variance non nulle
            if len(values) >= min_group_size and np.std(values) > 0: # type: ignore
                valid_groups.append(values)
                group_names.append(name)
        
        k = len(valid_groups)
        result["k_groups"] = k
        
        if k < 2:
            result["test"] = "Kruskal-Wallis"
            result["interpretation_p"] = f"❌ Moins de 2 groupes valides (k={k})"
            if verbose:
                print(f"\n{'='*60}\n⚠️  GROUPES INSUFFISANTS\n{'='*60}")
                print(f"Variables : {group_var} (cat) → {value_var} (num)")
                print(f"Groupes valides : {k}/2 minimum requis")
            return result
        
        # Test de Kruskal-Wallis
        H, p = kruskal(*valid_groups)
        
        # Calcul de eta² (epsilon²)
        N = sum(len(g) for g in valid_groups)
        result["n_used"] = N
        
        eta2 = np.nan
        if (N - k) > 0 and not np.isnan(H):
            eta2 = (H - k + 1) / (N - k)
            eta2 = float(np.clip(eta2, 0.0, 1.0))
        
        result["test"] = f"Kruskal-Wallis ({group_var} → {value_var})"
        result["p_value"] = round(_safe_float(p),4)
        result["effect_size"] = round(_safe_float(eta2),4)
        result["effect_size_name"] = "eta²"
        result["interpretation_p"] = _interpret_p_value(result["p_value"], alpha)
        result["interpretation_effect"] = _interpret_eta_squared(result["effect_size"])
    
    # === CAS 3 : CATÉGORIEL × CATÉGORIEL ===
    elif x_is_cat and y_is_cat:
        # Table de contingence
        contingency = pd.crosstab(x_series, y_series)
        
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            result["test"] = "Chi² d'indépendance"
            result["interpretation_p"] = "❌ Table insuffisante (minimum 2×2 requis)"
            if verbose:
                print(f"\n{'='*60}\n⚠️  TABLE DE CONTINGENCE INSUFFISANTE\n{'='*60}")
                print(f"Variables : {var_x} × {var_y}")
                print(f"Dimensions : {contingency.shape[0]}×{contingency.shape[1]}")
            return result
        
        # Test du Chi²
        chi2, p, dof, expected = chi2_contingency(contingency)
        v = _cramers_v(x_series, y_series)
        
        result["test"] = "Chi² d'indépendance"
        result["p_value"] = round(_safe_float(p),4)
        result["effect_size"] = round(_safe_float(v),4)
        result["effect_size_name"] = "V de Cramér"
        result["interpretation_p"] = _interpret_p_value(result["p_value"], alpha)
        result["interpretation_effect"] = _interpret_correlation(result["effect_size"])
    
    # === CAS NON GÉRÉ ===
    else:
        result["interpretation_p"] = "❌ Type de variables non pris en charge"
        result["interpretation_effect"] = "Vérifier les types de données"
        if verbose:
            print(f"\n{'='*60}\n⚠️  TYPE NON GÉRÉ\n{'='*60}")
            print(f"{var_x} : {result['dtype_x']}")
            print(f"{var_y} : {result['dtype_y']}")
        return result
    
    # === AFFICHAGE RÉSUMÉ ===
    if verbose:
        print(f"\n{'='*60}")
        print(f"📊  ANALYSE D'ASSOCIATION")
        print(f"{'='*60}")
        print(f"Variables     : {var_x} × {var_y}")
        print(f"Test          : {result['test']}")
        print(f"Observations  : {result['n_used']}/{result['n_total']} ({result['n_missing']} manquantes)")
        
        if result['k_groups'] is not None:
            print(f"Groupes       : {result['k_groups']}")
        
        print(f"\n📈 RÉSULTATS")
        print(f"{'-'*60}")
        print(f"p-value       : {result['p_value']:.4f}")
        
        if result['effect_size_name']:
            print(f"{result['effect_size_name']:13} : {result['effect_size']:.3f}")
        
        print(f"\n💡 INTERPRÉTATION")
        print(f"{'-'*60}")
        print(f"{result['interpretation_p']}")
        print(f"{result['interpretation_effect']}")
        print(f"{'='*60}\n")
    
    return result