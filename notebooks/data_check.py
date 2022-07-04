# %%
import gc

import numpy as np
import pandas as pd


# %%
train = pd.read_pickle("../input/amex-time-features/train_time_features.pkl", compression="gzip")


train.shape
# %%

train.head()
# %%
train["D_59_diff_last"].head()
# %%
