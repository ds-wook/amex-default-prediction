# %%
import numpy as np
import pandas as pd

# %%
ensemble = pd.read_csv("../output/ensemble_blend_preds.csv")
xgb = pd.read_csv("../output/5fold_xgboost_version2.csv")

blend = ensemble.copy()

blend = pd.merge(ensemble, xgb, on="customer_ID")
blend.rename(
    columns={"prediction_x": "ensemble_preds", "prediction_y": "xgb_preds"},
    inplace=True,
)
blend.head()
# %%
blend["prediction"] = blend["ensemble_preds"] * 0.7 + blend["xgb_preds"] * 0.3
blend[["customer_ID", "prediction"]].to_csv(
    "../output/ensemble_xgb_preds.csv", index=False
)
# %%
