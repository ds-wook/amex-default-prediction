# %%
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# %%
train = pd.read_parquet("../input/amex-data-parquet/train.parquet")
target = pd.read_csv("../input/amex-default-prediction/train_labels.csv")
train.head()
# %%
data = pd.merge(train, target, on="customer_ID")
data.head()
# %%
data[data["target"] == 1].head()
# %%
test = pd.read_parquet("../input/amex-data-parquet/test.parquet")
test.shape
# %%
