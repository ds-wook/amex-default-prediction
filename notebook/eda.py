# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
path = "../input/amex-data-parquet/"
train = pd.read_parquet(path + "train.parquet")
train.head()
# %%
group = train.groupby(["customer_ID"]).size().to_frame("size").to_numpy()
group.shape
# %%
np.unique(group)
# %%
group = train.groupby(["customer_ID"]).size()
group
# %%
group.unique()
# %%
