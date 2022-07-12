# %%
import gc

import numpy as np
import pandas as pd

# %%
ensemble_preds = pd.read_csv("../output/test_lgbm_baseline_5fold_seed42.csv")
ensemble_preds.head()
# %%
lgbm = pd.read_csv("../output/gradient_ensemble.csv")
lgbm.head()
# %%
ensemble_preds["prediction"] = (
    0.7 * ensemble_preds["prediction"] + 0.3 * lgbm["prediction"]
)
ensemble_preds.head()
# %%
ensemble_preds.to_csv("../output/ensemble_gradient_lightgbm.csv", index=False)

# %%
