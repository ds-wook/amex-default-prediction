# %%
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# %%
preds1 = pd.read_csv("../output/5fold_lightgbm_lag_features_ensemble.csv")
preds2 = pd.read_csv("../output/10fold_xgboost_stacking.csv")
preds3 = pd.read_csv("../output/10fold_tabnet_stacking.csv")
# %%
preds1["prediction"] = (
    0.97 * preds1["prediction"]
    + 0.02 * preds2["prediction"]
    + 0.01 * preds3["prediction"]
)
preds1.head()

# %%
preds1.to_csv("../output/5fold_lightgbm_final_ensemble.csv", index=False)
# %%
df = preds1.copy()
df.head()
# %%
df["prediction2"] = preds2.prediction
df["prediction3"] = preds3.prediction
df["prediction4"] = preds4.prediction
# %%
df.corr()
# %%
