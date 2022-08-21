import gc
from pathlib import Path
from typing import List, NoReturn, Tuple

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

tqdm.pandas()


def split_dataset(a: np.ndarray, n: int) -> Tuple[np.ndarray]:
    """
    Split array into n parts
    Args:
        a: array
        n: number of parts
    Returns:
        array of tuple
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def get_difference(df: pd.DataFrame, num_features: List[str]) -> pd.DataFrame:
    """
    Create diff feature
    Args:
        df: dataframe
        num_features: list of numerical features
    Returns:
        dataframe
    """

    df_diff = (
        df.loc[:, num_features + ["customer_ID"]]
        .groupby(["customer_ID"])
        .progress_apply(
            lambda x: np.diff(x.values[-2:, :], axis=0).squeeze().astype(np.float32)
        )
    )
    cols = [col + "_diff1" for col in df[num_features].columns]
    df_diff = pd.DataFrame(df_diff.values.tolist(), columns=cols, index=df_diff.index)
    df_diff.reset_index(inplace=True)

    return df_diff


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features
    Args:
        df: dataframe
    Returns:
        dataframe
    """
    # drop nan >=75% features
    drop_features = (
        "D_42, D_49, D_66, D_73, D_76, R_9, B_29, D_87, D_88, D_106, R_26, D_108, "
        "D_110, D_111, B_39, B_42, D_132, D_134, D_135, D_136, D_137, D_138, D_142"
    )
    drop_features = drop_features.split(", ")
    df = df.drop(columns=drop_features)

    # FEATURE ENGINEERING FROM
    # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created
    df["S_2"] = pd.to_datetime(df["S_2"])
    df["SDist"] = df[["customer_ID", "S_2"]].groupby(
        "customer_ID"
    ).diff() / np.timedelta64(1, "D")
    df["SDist"] = df["SDist"].fillna(30.53)

    all_cols = [c for c in list(df.columns) if c not in ["customer_ID", "S_2"]]
    # cat_features = (
    #     "B_30, B_38, D_114, D_116, D_117, D_120, D_126, D_63, D_64, D_66, D_68"
    # )
    cat_features = (
        "B_30, B_38, D_114, D_116, D_117, D_120, D_126, D_63, D_64, D_68"
    )
    cat_features = cat_features.split(", ")
    num_features = [col for col in all_cols if col not in cat_features]

    # Get the difference
    df_diff_agg = get_difference(df, num_features)

    df_num_agg = df.groupby("customer_ID")[num_features].agg(
        ["first", "mean", "std", "min", "max", "last"]
    )
    df_num_agg.columns = ["_".join(x) for x in df_num_agg.columns]
    df_num_agg.reset_index(inplace=True)

    df_cat_agg = df.groupby("customer_ID")[cat_features].agg(
        ["last", "nunique", "count"]
    )
    df_cat_agg.columns = ["_".join(x) for x in df_cat_agg.columns]
    df_cat_agg.reset_index(inplace=True)

    # add last statement date, statements count and "new customer" category (LT=0.5)
    df_date_agg = df.groupby("customer_ID")[["S_2", "B_3", "D_104"]].agg(
        ["last", "count"]
    )
    df_date_agg.columns = ["_".join(x) for x in df_date_agg.columns]
    df_date_agg.rename(columns={"S_2_count": "LT", "S_2_last": "S_2"}, inplace=True)
    df_date_agg.loc[(df_date_agg.B_3_last.isnull()) & (df_date_agg.LT == 1), "LT"] = 0.5
    df_date_agg.loc[
        (df_date_agg.D_104_last.isnull()) & (df_date_agg.LT == 1), "LT"
    ] = 0.5
    df_date_agg.drop(
        ["B_3_last", "D_104_last", "B_3_count", "D_104_count"], axis=1, inplace=True
    )
    df_date_agg.reset_index(inplace=True)

    # Transform int64 columns to int32
    cols = list(df_num_agg.dtypes[df_num_agg.dtypes == "float64"].index)
    df_num_agg.loc[:, cols] = df_num_agg.loc[:, cols].progress_apply(
        lambda x: x.astype(np.float32)
    )

    # Transform int64 columns to int32
    cols = list(df_cat_agg.dtypes[df_cat_agg.dtypes == "int64"].index)
    df_cat_agg.loc[:, cols] = df_cat_agg.loc[:, cols].progress_apply(
        lambda x: x.astype(np.int32)
    )

    df = (
        df_num_agg.merge(df_cat_agg, how="inner", on="customer_ID")
        .merge(df_diff_agg, how="inner", on="customer_ID")
        .merge(df_date_agg, how="inner", on="customer_ID")
    )

    del df_num_agg, df_cat_agg, df_diff_agg, df_date_agg
    gc.collect()

    return df


@hydra.main(config_path="../config/", config_name="data", version_base="1.2.0")
def _main(cfg: DictConfig) -> NoReturn:
    path = Path(get_original_cwd())
    train = pd.read_parquet(path / "input/amex-data-parquet/train.parquet")
    label = pd.read_csv(path / "input/amex-default-prediction/train_labels.csv")
    test = pd.read_parquet(path / "input/amex-data-parquet/test.parquet")
    path = Path(get_original_cwd()) / cfg.dataset.path

    # build train features
    train = build_features(train)
    train = pd.merge(train, label, on="customer_ID")

    print(train.shape)

    train.to_parquet(path / f"{cfg.dataset.train}.parquet")

    del train
    gc.collect()

    # build test features
    split_ids = split_dataset(test.customer_ID.unique(), cfg.dataset.num_test)

    for (i, ids) in enumerate(split_ids):
        test_sample = test[test.customer_ID.isin(ids)]
        test_agg = build_features(test_sample)
        print(i, test_agg.shape)

        test_agg.to_parquet(path / f"{cfg.dataset.test}_{i}.parquet")

        del test_agg
        gc.collect()

    del test
    gc.collect()


if __name__ == "__main__":
    _main()
