# %%
import pandas as pd

# %%
train = pd.read_parquet("../input/amex-default-prediction/train_meta_features.parquet")
train.head()
# %%
train["preds_mean"]
# %%
