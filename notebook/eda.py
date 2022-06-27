# %%
import numpy as np
import pandas as pd

# %%
path = "../input/amex-data-parquet/"
train = pd.read_parquet(path + "train.parquet")
train.head()
# %%
train.shape
# %%
train.groupby("customer_ID").tail(1)
# %%
train.drop_duplicates(subset=["customer_ID"], keep="last").isna().sum()

# %%
train.drop_duplicates(subset=["customer_ID"], keep="last")[train.drop_duplicates(subset=["customer_ID"], keep="last")]