# %%
import gc

import numpy as np
import pandas as pd

# %%
preds1 = pd.read_csv("../output/5fold_lightgbm_trick_features_seed22.csv")
preds2 = pd.read_csv("../output/5fold_lightgbm_rate_features_seed52.csv")
preds3 = pd.read_csv("../output/5fold_lightgbm_trick_features_seed94.csv")
# %%
preds1["prediction"] = (
    preds1["prediction"]
    + preds2["prediction"]
    + preds3["prediction"]
) / 3
preds1.head()
# %%
preds1.to_csv("../output/ensemble_gradient_lightgbm.csv", index=False)

# %%
ensemble_preds = pd.read_csv("../output/overfitting_lb.csv")
preds1["prediction"] = (preds1["prediction"] * 0.2 + ensemble_preds["prediction"] * 0.8)
preds1.head()
# %%
preds1.to_csv("../output/overfitting_lb.csv", index=False)
# %%
