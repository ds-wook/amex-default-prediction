# %%
import gc

import numpy as np
import pandas as pd

# %%
ensemble_preds = pd.read_csv("../output/ensemble_blend_preds_with_lag.csv")
ensemble_preds.head()
# %%
ensemble_preds["prediction"] = np.round(ensemble_preds["prediction"], 2)
ensemble_preds.head()
# %%
lgbm = pd.read_csv("../output/5fold_lightgbm_lag_features.csv")
lgbm.head()
# %%
ensemble_preds["prediction"] = (
    0.7 * ensemble_preds["prediction"] + 0.3 * lgbm["prediction"]
)
ensemble_preds.head()
# %%
ensemble_preds.to_csv("../output/ensemble_blend_preds_with_2lag.csv", index=False)
