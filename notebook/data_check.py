# %%
import gc

import numpy as np
import pandas as pd

# %%
preds1 = pd.read_csv("../output/5fold_lightgbm_trick_features_seed22.csv")
preds2 = pd.read_csv("../output/5fold_lightgbm_sdist_features_seed42.csv")
preds3 = pd.read_csv("../output/5fold_lightgbm_trick_features_seed94.csv")
# %%
preds1["prediction"] = (
    preds1["prediction"] + preds2["prediction"] + preds3["prediction"]
) / 3
preds1.head()

# %%
ensemble_preds1 = pd.read_csv("../output/stacking_ensemble.csv")
ensemble_preds2 = pd.read_csv("../output/10fold_stacking_tabnet.csv")
# %%
ensemble_preds1["prediction"] = (
    0.4 * preds1["prediction"]
    + 0.2 * ensemble_preds1["prediction"]
    + 0.4 * ensemble_preds2["prediction"]
)
ensemble_preds1.to_csv("../output/ensemble_mean_preds.csv", index=False)
# %%
