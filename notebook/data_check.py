# %%
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
import numpy as np

ensemble_preds1 = pd.read_csv("../output/overfitting_lb.csv")
ensemble_preds1["prediction"] = np.clip(ensemble_preds1["prediction"], 0, 1)
ensemble_preds1.head()
# %%
ensemble_preds1.to_csv("../output/overfitting_lb_test.csv", index=False)
# %%
