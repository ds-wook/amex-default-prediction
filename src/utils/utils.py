from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import figure


def plot_importance(
    importances: np.ndarray,
    features_names: List[str],
    PLOT_TOP_N: int = 20,
    figsize: Tuple[int, int] = (10, 10),
) -> figure:
    importance_df = pd.DataFrame(data=importances, columns=features_names)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale("log")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Importance")
    sns.boxplot(data=sorted_importance_df[plot_cols], orient="h", ax=ax)

    return plt
