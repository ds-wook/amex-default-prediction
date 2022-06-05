# %%
import numpy as np
import pandas as pd

# %%


def split_test_dataset(num: int) -> None:
    """
    Split test dataset
    Args:
        config: config file
    """
    test = pd.read_pickle(
        f"../input/amex-agg-data-f32/test_agg_f32_part_{num}.pkl", compression="gzip"
    )

    test_credit = pd.read_csv("../input/amex-default-prediction/test_credit.csv")

    test_credit_sample = test_credit.iloc[num : (num + 1) * 100000]
    test["CS"] = test_credit_sample["CS"]
    test.to_pickle(
        f"../input/amex-agg-data-f32/test_agg_f32_part_credit_{num}.pkl",
        compression="gzip",
    )


# %%
for num in range(10):
    split_test_dataset(num)
# %%
train = pd.read_pickle(
    "../input/amex-agg-data-f32/train_agg_f32.pkl", compression="gzip"
)

# %%
train_credit = pd.read_csv("../input/amex-default-prediction/train_credit.csv")

# %%
train
# %%
train_credit
# %%
train["CS"] = train_credit["CS"].copy()
# %%
train.to_pickle(
    "../input/amex-agg-data-f32/train_agg_f32_credit.pkl",
    compression="gzip",
)

# %%
