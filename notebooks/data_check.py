# %%
import numpy as np
import pandas as pd
from scipy.stats import rankdata

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
oof_scores = pd.read_csv("../res/models/oof_5fold_boosting_gradient.csv")
stacking_scores = pd.read_csv("../res/models/oof_5fold_stacking_gradient.csv")
print(amex_metric(oof_scores["target"], oof_scores["prediction"]))
print(amex_metric(stacking_scores["target"], stacking_scores["prediction"]))
# %%
import optuna


def objective(trial):
    w1 = trial.suggest_uniform("w1", 0, 1)
    oof_preds = oof_scores["prediction"] * (1 - w1) + stacking_scores["prediction"] * w1
    return amex_metric(oof_scores["target"], oof_preds)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_params)
# %%
oof_preds = pd.DataFrame()
oof_preds["prediction"] = (
    oof_scores["prediction"] * (1-0.03221084221328546) + stacking_scores["prediction"] * 0.03221084221328546
)
print(amex_metric(oof_scores["target"], oof_preds["prediction"]))
# %%
preds1 = pd.read_csv("../output/5fold_boosting_gradient.csv")
preds2 = pd.read_csv("../output/5fold_stacking_gradient.csv")
preds3 = pd.read_csv("../output/keras-cnn_sub.csv")
preds4 = pd.read_csv("../output/lb_overfitting.csv")
preds1.head()
# %%
preds1["prediction"] = 0.999 * preds1["prediction"] + 0.001 * preds2["prediction"]
preds1.head()

# %%
preds1.to_csv("../output/final_ensemble_submit_ver1.csv", index=False)

# %%
preds1["prediction"] = 0.9999 * preds1["prediction"] + 0.0001 * preds3["prediction"]
preds1.head()
# %%
preds1.to_csv("../output/final_ensemble_submit_ver2.csv", index=False)
# %%
preds1
# %%
