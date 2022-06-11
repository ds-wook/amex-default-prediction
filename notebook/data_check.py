# %%
import numpy as np
import pandas as pd

# %%
submission = pd.read_csv("../output/sample_submission.csv")
ensemble_blend_preds = pd.read_csv("../output/ensemble_blend_preds.csv")
lgbm_pay = pd.read_csv("../output/10fold_lightgbm_pay_features.csv")
# %%

submission["prediction"] = (
    ensemble_blend_preds["prediction"] * 0.7 + lgbm_pay["prediction"] * 0.3
)
submission.to_csv("../output/simple_blend.csv", index=False)

# %%
