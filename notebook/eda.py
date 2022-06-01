# %%
import numpy as np
import pandas as pd

# %%
train = pd.read_feather("../input/amex-default-prediction/train_data.ftr")
test = pd.read_feather("../input/amex-default-prediction/test_data.ftr")

# %%
train.head()
# %%
train = (
    train.groupby("customer_ID")
    .tail(1)
    .set_index("customer_ID", drop=True)
    .sort_index()
    .drop(["S_2"], axis="columns")
)

train.head()
# %%
train.columns
# %%