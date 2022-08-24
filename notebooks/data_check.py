# %%
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# %%
preds1 = pd.read_csv("../output/5fold_best_boosting_gradient.csv")
preds2 = pd.read_csv("../output/lb_overfitting.csv")
# preds3 = pd.read_csv("../output/10fold_stacking_tabnet.csv")
# preds4 = pd.read_csv("../output/keras-cnn_sub.csv")
preds2.head()
# %%
preds1["prediction"] = 0.7 * preds1["prediction"] + 0.3 * preds2["prediction"]
preds1.head()

# %%
preds1.to_csv("../output/final_ensemble_submit_ver1.csv", index=False)
# %%


def amex_metric(y_true, y_pred) -> float:
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])
    gini = [0, 0]

    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


# %%
oof_scores = pd.read_csv("../res/models/oof_5fold_best_boosting_gradient.csv")
print(amex_metric(oof_scores["target"], oof_scores["prediction"]))
catboost_oof_scores = pd.read_csv(
    "../res/models/oof_5fold_tabnet_bruteforce_features_seed3407.csv"
)
oof_scores["prediction"] = (
    oof_scores["prediction"] * 0.999996 + catboost_oof_scores["prediction"] * 0.000004
)
print(amex_metric(oof_scores["target"], oof_scores["prediction"]))
# %%
np.exp(-10)
# %%
