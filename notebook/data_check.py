# %%
import numpy as np
import pandas as pd

# %%
ensemble_preds = pd.read_csv("../output/ensemble_blend_preds.csv")
ensemble_preds.head()
# %%
lightgbm_dart = pd.read_csv("../output/test_lgbm_baseline_5fold_seed42.csv")
lightgbm_dart.head()
# %%
ensemble_preds["prediction"] = (
    ensemble_preds["prediction"] * 0.7 + lightgbm_dart["prediction"] * 0.3
)
ensemble_preds.head()
# %%
ensemble_preds.to_csv("../output/ensemble_blend_preds.csv", index=False)
# %%
