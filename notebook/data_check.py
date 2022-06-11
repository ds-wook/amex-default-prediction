# %%
import numpy as np
import pandas as pd

# %%
train = pd.read_pickle(
    "../input/amex-pay-features/train_pay_features.pkl", compression="gzip"
)
print(train.shape)
# %%
train.head()
# %%
