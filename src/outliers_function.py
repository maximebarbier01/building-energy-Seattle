import pandas as pd

def detect_outliers_iqr(df, factor=1.5):
    """
    Detect outliers using the IQR method for each numerical column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing numerical variables.
    factor : float, optional (default=1.5)
        Multiplicative factor for the IQR rule.

    Returns
    -------
    outliers_dict : dict
        Dictionary with column names as keys and DataFrames of outliers as values.
    outliers_ratio : pandas.Series
        Proportion of outliers per column.
    """

    outliers_dict = {}
    outliers_ratio = {}

    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        mask = (df[col] < Q1 - factor * IQR) | (df[col] > Q3 + factor * IQR)
        outliers_dict[col] = df[mask]
        outliers_ratio[col] = mask.mean()*100

    return outliers_dict, pd.Series(outliers_ratio)

import numpy as np
import matplotlib.pyplot as plt

def plot_lorenz(series, ax=None, label=None):
    s = series.dropna().sort_values()
    
    cum_values = s.cumsum()
    cum_share = cum_values / cum_values.iloc[-1]
    cum_pop = np.arange(1, len(s)+1) / len(s)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    
    ax.plot(cum_pop, cum_share, label=label)
    ax.plot([0,1], [0,1], linestyle="--", color="grey")
    
    ax.set_xlabel("Part cumulée des bâtiments")
    ax.set_ylabel("Part cumulée de la variable")
    ax.set_title("Courbe de Lorenz")
    
    if label:
        ax.legend()
    
    return ax
