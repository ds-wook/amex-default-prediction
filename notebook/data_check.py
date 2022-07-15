# %%
import gc

import numpy as np
import pandas as pd

# %%
ensemble_preds = pd.read_csv("../output/test_lgbm_baseline_5fold_seed42.csv")
ensemble_preds.head()
# %%
lgbm = pd.read_csv("../output/5fold_lightgbm_rate_features_seed52.csv")
lgbm.head()
# %%
ensemble_preds["prediction"] = (
    0.7 * ensemble_preds["prediction"] + 0.3 * lgbm["prediction"]
)
ensemble_preds.head()
# %%
ensemble_preds.to_csv("../output/ensemble_gradient_lightgbm.csv", index=False)

# %%
train = pd.read_parquet("../input/amex-data-parquet/train.parquet")
train.head()
# %%
