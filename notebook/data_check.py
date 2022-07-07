# %%
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
train_sample = pd.read_pickle("../input/amex-lag-features/train_lag_features_0.pkl", compression="gzip")
train_sample["D_39_diff_last"].head()
# %%
