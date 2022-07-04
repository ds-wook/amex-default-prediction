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
train["B_30"].head()
# %%
train["B_2"].head()

# %%

sns.lineplot(x="S_2", y="B_2", data=train)
plt.xticks(rotation=45)
plt.show()

# %%

train["B_2_Bin"] = train["B_2"].map(
    lambda x: 0 if 0.7 < x < 0.9 else np.nan if np.isnan(x) else 1
)
train["B_2_Bin"].head()
# %%

train["B_2_Bin"].isna().sum()
# %%
train["B_2"].isna().sum()
# %%
