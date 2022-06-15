# %%
import numpy as np
import pandas as pd

# %%
ensemble_preds = pd.read_csv("../output/ensemble_blend_preds.csv")
catboost = pd.read_csv("../output/catboost_preds.csv")
ensemble_preds.head()
# %%
catboost.head()
# %%
ensemble_preds["prediction"] = (
    ensemble_preds["prediction"] * 0.9 + catboost["prediction"] * 0.1
)
# %%
ensemble_preds.head()
# %%
ensemble_preds.to_csv("../output/ensemble_blend_preds_test.csv", index=False)
# %%
