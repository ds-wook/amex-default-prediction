# %%
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
train_credit = pd.read_csv("../input/amex-default-prediction/train_credit.csv")
test_credit = pd.read_csv("../input/amex-default-prediction/test_credit.csv")

train_proba = pd.read_csv("../input/amex-default-prediction/train_credit_proba.csv")
test_proba = pd.read_csv("../input/amex-default-prediction/test_credit_proba.csv")

# %%
for num in tqdm(range(10)):
    test_sample = pd.read_pickle(
        f"../input/amex-agg-data-f32/test_agg_f32_part_{num}.pkl", compression="gzip"
    )
    test_sample["CS"] = test_credit.iloc[num * 100000 : (num + 1) * 100000]["CS"].copy()
    test_sample["CP"] = test_proba.iloc[num * 100000 : (num + 1) * 100000][
        "credit_proba"
    ].copy()
    test_sample.to_pickle(
        f"../input/amex-credit-data/test_agg_f32_credit_{num}.pkl", compression="gzip"
    )

# %%
train = pd.read_pickle(
    "../input/amex-agg-data-f32/train_agg_f32.pkl", compression="gzip"
)

train["CS"] = train_credit["CS"].copy()
train["CP"] = train_proba["credit_proba"].copy()
train.to_pickle("../input/amex-credit-data/train_agg_credit.pkl", compression="gzip")

# %%
test_sample.shape

# %%

ensemble = pd.read_csv("../output/ensemble_test.csv")
tabnet = pd.read_csv("../output/10fold_tabnet.csv")

# %%
ensemble["prediction"] = (
    ensemble["prediction"] * 0.9 + tabnet["prediction"] * 0.1
)
ensemble.to_csv("../output/ensemble_blend_preds.csv", index=False)
# %%
