# %%
import gc

import numpy as np
import pandas as pd


# %%
train = pd.read_parquet("../input/amex-data-parquet/train.parquet")

train.shape
# %%

# %%
len(time_features)
# %%
set(train.columns.to_list()) - set(time_features)
# %%
