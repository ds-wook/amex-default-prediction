# %%
import gc

import numpy as np
import pandas as pd

# %%
ensemble_preds = pd.read_csv("../output/ensemble_blend_preds_with_lag.csv")
ensemble_preds.head()
# %%
lgbm = pd.read_csv("../output/5fold_lightgbm_time_diff_features.csv")
lgbm.head()
# %%
ensemble_preds["prediction"] = (
    0.3 * ensemble_preds["prediction"] + 0.7 * lgbm["prediction"]
)
ensemble_preds.head()
# %%
ensemble_preds.to_csv("../output/ensemble_preds_with_lgbm.csv", index=False)

# %%
