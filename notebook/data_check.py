# %%
import pandas as pd
from tqdm import tqdm

train_credit = pd.read_csv("../input/amex-default-prediction/train_credit.csv")
test_credit = pd.read_csv("../input/amex-default-prediction/test_credit.csv")

# %%

for num in tqdm(range(10)):
    test_sample = pd.read_pickle(
        f"../input/amex-agg-data-f32/test_agg_f32_part_{num}.pkl", compression="gzip"
    )
    test_sample["CS"] = test_credit.iloc[num * 100000 : (num + 1) * 100000]["CS"].copy()
    test_sample.to_pickle(
        f"../input/amex-agg-data-f32/test_agg_f32_credit_{num}.pkl", compression="gzip"
    )

# %%
train = pd.read_pickle(
    "../input/amex-agg-data-f32/train_agg_f32.pkl", compression="gzip"
)

train["CS"] = train_credit["CS"].copy()
train.to_pickle(
    "../input/amex-agg-data-f32/train_agg_f32_credit.pkl", compression="gzip"
)

# %%
test_sample.shape
# %%
test = pd.read_feather("../input/amex-default-prediction/test_data.ftr")

cid = test.pop("customer_ID")
test["customer_ID"] = cid.str[-16:].apply(lambda x: int(x, 16))
