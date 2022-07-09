# %%
import numpy as np
import pandas as pd

# %%
# %%
ensemble_preds = pd.read_csv("../output/ensemble_blend_preds_with_lag.csv")
ensemble_preds.head()
# %%
lgbm = pd.read_csv("../output/test_lgbm_baseline_5fold_seed_blend.csv")
lgbm.head()
# %%
ensemble_preds["prediction"] = (
    0.2 * ensemble_preds["prediction"] + 0.8 * lgbm["prediction"]
)
ensemble_preds.head()
# %%
ensemble_preds.to_csv("../output/ensemble_blend_preds.csv", index=False)

# %%
