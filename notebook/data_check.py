# %%
import numpy as np
import pandas as pd
from amex.data.dataset import load_train_dataset, load_test_dataset
# %%
lgbm_preds = pd.read_csv("../output/gradient_ensemble.csv")

lgbm_preds.head()
# %%
ensemble = pd.read_csv("../output/ensemble_blend_preds.csv")
ensemble.head()
# %%
ensemble["prediction"] = (lgbm_preds["prediction"] * 0.4 + ensemble["prediction"] * 0.6)
ensemble.head()
# %%
ensemble.to_csv("../output/ensemble_blend_preds2.csv", index=False)
# %%
