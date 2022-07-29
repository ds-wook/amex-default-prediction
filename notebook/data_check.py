# %%
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# %%
preds1 = pd.read_csv("../output/5fold_lightgbm_statement_features_seed88.csv")
preds2 = pd.read_csv("../output/5fold_lightgbm_statement_features_seed94.csv")
preds3 = pd.read_csv("../output/5fold_lightgbm_statement_features_seed99.csv")
# %%
preds1["prediction"] = (
    preds1["prediction"] + preds2["prediction"] + preds3["prediction"]
) / 3
preds1.head()

# %%
preds1.to_csv("../output/5fold_lightgbm_statement_features_ensemble.csv", index=False)
# %%
ensemble_preds1 = pd.read_csv("../output/overfitting_lb.csv")
ensemble_preds1["prediction"] = (
    ensemble_preds1["prediction"] * 0.8 + preds3["prediction"] * 0.2
)
ensemble_preds1.head()
# %%
ensemble_preds1.to_csv("../output/overfitting_lb_test.csv", index=False)
# %%


rankdata(preds1["prediction"])
# %%
rankdata(preds2["prediction"])
# %%
