# %%
import pickle

import pandas as pd

# %%
train = pd.read_feather("../input/amex-default-prediction/train_data.ftr")
# %%
model_path = "../res/models/10fold_lightgbm_avg_models.pkl"
with open(model_path, "rb") as output:
    model_result = pickle.load(output)

model_result
# %%
train.columns
# %%
train.shape
# %%
