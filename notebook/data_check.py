# %%
import gc

import pandas as pd
from tqdm import tqdm


# %%
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # FEATURE ENGINEERING FROM
    # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created

    all_cols = [c for c in list(df.columns) if c not in ["customer_ID", "S_2"]]
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    num_features = [col for col in all_cols if col not in cat_features]
    df_num_agg = df.groupby("customer_ID")[num_features].agg(
        ["mean", "std", "min", "max", "last"]
    )
    df_num_agg.columns = ["_".join(x) for x in df_num_agg.columns]

    df_cat_agg = df.groupby("customer_ID")[cat_features].agg(
        ["last", "nunique", "count"]
    )
    df_cat_agg.columns = ["_".join(x) for x in df_cat_agg.columns]

    df = pd.concat([df_num_agg, df_cat_agg], axis=1)
    del df_num_agg, df_cat_agg
    gc.collect()

    print("shape after engineering", df.shape)

    return df


# %%
train = pd.read_parquet("../input/amex-data-parquet/train.parquet")
# %%
def add_after_pay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that are only available after payment.
    Args:
        df: DataFrame with customer_ID as index.
    Returns:
        DataFrame with customer_ID as index and additional features.
    """
    df = df.copy()
    before_cols = ["B_11", "B_14", "B_17", "D_39", "D_131", "S_16", "S_23"]
    after_cols = ["P_2", "P_3"]

    # after pay features
    after_pay_features = []
    for b_col in before_cols:
        for a_col in after_cols:
            if b_col in df.columns:
                df[f"{b_col}_{a_col}"] = df[b_col] - df[a_col]
                after_pay_features.append(f"{b_col}_{a_col}")

    df_after_agg = df.groupby("customer_ID")[after_pay_features].agg(["mean", "std"])
    df_after_agg.columns = ["_".join(x) for x in df_after_agg.columns]

    return df_after_agg


# %%
train_after_agg = add_after_pay_features(train)
train_after_agg.head()

# %%
train.head()
# %%
test_sample = pd.read_pickle(
    "../input/amex-pay-features/test_pay_features_part_9.pkl", compression="gzip"
)
test_sample.tail()
# %%
test_sample = pd.read_pickle(
    "../input/amex-agg-data-f32/test_agg_f32_part_9.pkl", compression="gzip"
)
test_sample.set_index("customer_ID").tail()
# %%
test_sample.index
# %%
test_sample.info()
# %%
"float" in test_sample["P_2_mean"].dtype.name
# %%
test_sample["P_2_mean"].dtype.name
# %%
train = pd.read_pickle("../input/amex-pay-features/train_pay_features.pkl", compression="gzip")
train.head()
# %%
