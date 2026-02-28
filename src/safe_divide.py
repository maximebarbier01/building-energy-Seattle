import numpy as np
import pandas as pd


# ---------------------------
# Helpers
# ---------------------------
def safe_divide(num: pd.Series, den: pd.Series, eps: float = 1e-6) -> pd.Series:
    """Division robuste: évite den ~= 0 et garde NaN si den est NaN."""
    den_adj = den.copy()
    den_adj = den_adj.where(den_adj.abs() > eps, np.nan)
    return num / den_adj


def iqr_bounds(s: pd.Series, factor: float = 1.5):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return lower, upper


def cap_iqr(df: pd.DataFrame, cols, factor: float = 1.5) -> pd.DataFrame:
    """Winsorisation IQR colonne par colonne."""
    out = df.copy()
    for c in cols:
        s = out[c]
        if s.dropna().empty:
            continue
        low, up = iqr_bounds(s, factor=factor)
        out[c] = s.clip(low, up)
    return out


def outlier_ratio_iqr(df: pd.DataFrame, cols, factor: float = 1.5) -> pd.Series:
    """Ratio d'observations outliers selon IQR."""
    ratios = {}
    for c in cols:
        s = df[c]
        s_nonan = s.dropna()
        if s_nonan.empty:
            ratios[c] = 0.0
            continue
        low, up = iqr_bounds(s_nonan, factor=factor)
        ratios[c] = ((s_nonan < low) | (s_nonan > up)).mean() * 100
    return pd.Series(ratios).sort_values(ascending=False)


def is_binary_series(s: pd.Series) -> bool:
    vals = pd.Series(s.dropna().unique())
    if len(vals) <= 2:
        return set(vals.astype(float)) <= {0.0, 1.0}
    return False


# ---------------------------
# Main turnkey function
# ---------------------------
def handle_outliers_turnkey(
    df: pd.DataFrame,
    ratio_defs: dict | None = None,
    log_keywords: tuple[str, ...] = ("kBtu", "GFA", "GHG", "Emissions", "EUI"),
    log_skew_threshold: float = 1.0,
    cap_factor: float = 3.0,  # plus doux que 1.5 (souvent mieux en énergie)
    eps: float = 1e-6,
    verbose: bool = True,
):
    """
    Traite les outliers de façon robuste:
    - (optionnel) recalcule des ratios en safe_divide
    - log1p sur variables positives très asymétriques
    - cap IQR (winsorisation) sur numériques hors binaires
    Retourne: df_clean, report(dict)
    """
    df2 = df.copy()

    # 1) Numériques
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()

    # 2) (Optionnel) ratios: ratio_defs = {"parking_gfa_ratio": ("parking_gfa", "PropertyGFATotal"), ...}
    if ratio_defs:
        for new_col, (num_col, den_col) in ratio_defs.items():
            if num_col in df2.columns and den_col in df2.columns:
                df2[new_col] = safe_divide(df2[num_col], df2[den_col], eps=eps)

        # refresh num cols
        num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()

    # 3) Exclure binaires du traitement outliers (pas de sens de cap/log)
    binary_cols = [c for c in num_cols if is_binary_series(df2[c])]
    cont_cols = [c for c in num_cols if c not in binary_cols]

    # 4) Rapport outliers AVANT
    ratio_before = outlier_ratio_iqr(df2, cont_cols, factor=1.5)

    # 5) Log1p sur variables positives, très skewed
    #    - cible par mots-clés (kBtu, GFA, etc.)
    #    - + condition: skewness > seuil et min >= 0
    log_cols = []
    for c in cont_cols:
        s = df2[c]
        if s.dropna().empty:
            continue
        if (s.min(skipna=True) >= 0) and (abs(s.dropna().skew()) >= log_skew_threshold):
            # petit filtre "métier" basé sur nom de variable (optionnel mais pratique)
            if any(k.lower() in c.lower() for k in log_keywords):
                log_cols.append(c)

    for c in log_cols:
        df2[c] = np.log1p(df2[c])

    # 6) Cap (winsorisation) IQR doux sur toutes les continues (après log)
    df3 = cap_iqr(df2, cont_cols, factor=cap_factor)

    # 7) Rapport outliers APRES
    ratio_after = outlier_ratio_iqr(df3, cont_cols, factor=1.5)

    report = {
        "n_rows": len(df),
        "n_numeric_cols": len(num_cols),
        "binary_cols": binary_cols,
        "log_cols_applied": log_cols,
        "outlier_ratio_before_pct_top": ratio_before.head(15),
        "outlier_ratio_after_pct_top": ratio_after.head(15),
    }

    if verbose:
        print("✅ Outlier handling done")
        print(f"- Numeric cols: {len(num_cols)} (binary excluded: {len(binary_cols)})")
        print(f"- log1p applied on: {len(log_cols)} columns")
        print("\nTop outlier ratios BEFORE (IQR 1.5):")
        display(ratio_before.head(15).to_frame("outlier_pct"))
        print("\nTop outlier ratios AFTER (IQR 1.5):")
        display(ratio_after.head(15).to_frame("outlier_pct"))

    return df3, report
