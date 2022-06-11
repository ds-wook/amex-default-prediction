# %%
import numpy as np
import pandas as pd

# %%
train_label = pd.read_csv("../input/amex-default-prediction/train_labels.csv")
train_label.head()
# %%
transformer_oof = np.load("../res/models/transformer_oof.npy")
transformer_oof.shape
# %%
def amex_metric(y_true: np.array, y_pred: np.array) -> float:
    indices = np.argsort(y_pred)[::-1]
    target = y_true[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / np.sum(target)

    weighted_target = target * weight
    lorentz = (weighted_target / weighted_target.sum()).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos
    gini_max = 10 * n_neg * (n_pos + 20 * n_neg - 19) / (n_pos + 20 * n_neg)

    g = gini / gini_max

    return 0.5 * (g + d)
# %%
amex_metric(train_label.target, transformer_oof)
# %%
