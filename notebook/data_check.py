# %%
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# %%
preds1 = pd.read_csv("../output/5fold_lightgbm_lag_features_ensemble.csv")
preds2 = pd.read_csv("../output/mean_submission.csv")
# %%
preds1["prediction"] = 0.9 * preds1["prediction"] + 0.1 * preds2["prediction"]
preds1.head()

# %%
preds1.to_csv("../output/5fold_lightgbm_final_ensemble.csv", index=False)
# %%
