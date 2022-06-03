# %%
import numpy as np
import pandas as pd

# %%
test = pd.read_pickle(
    "../input/amex-default-prediction/test_agg.pkl", compression="gzip"
)

# %%
test_sample = pd.read_pickle(
    "../input/amex-agg-data-f32/test_agg_f32_part_0.pkl", compression="gzip"
)
# %%
test["B_30_last"].head()
# %%
test_sample["B_30_last"].head()
# %%
