import gc
import warnings
from pathlib import Path
from typing import List, NoReturn, Tuple

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

warnings.filterwarnings("ignore")
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
        .progress_apply(lambda x: np.diff(x.values[-2:, :], axis=0).squeeze().astype(np.float32))
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
    # FEATURE ENGINEERING FROM

    all_cols = [c for c in list(df.columns) if c not in ["customer_ID", "S_2"]]
    cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
    num_features = [col for col in all_cols if col not in cat_features + ["preds"]]

    # Get the difference
    df_diff_agg = get_difference(df, num_features)

    num_features = [col for col in all_cols if col not in cat_features]
    df_num_agg = df.groupby("customer_ID")[num_features].agg(["first", "mean", "std", "min", "max", "last"])
    df_num_agg.columns = ["_".join(x) for x in df_num_agg.columns]
    df_num_agg.reset_index(inplace=True)

    df_cat_agg = df.groupby("customer_ID")[cat_features].agg(["count", "first", "last", "nunique"])
    df_cat_agg.columns = ["_".join(x) for x in df_cat_agg.columns]
    df_cat_agg.reset_index(inplace=True)

    # Transform int64 columns to int32
    cols = list(df_num_agg.dtypes[df_num_agg.dtypes == "float64"].index)
    df_num_agg.loc[:, cols] = df_num_agg.loc[:, cols].progress_apply(lambda x: x.astype(np.float32))

    # Transform int64 columns to int32
    cols = list(df_cat_agg.dtypes[df_cat_agg.dtypes == "int64"].index)
    df_cat_agg.loc[:, cols] = df_cat_agg.loc[:, cols].progress_apply(lambda x: x.astype(np.int32))

    df = df_num_agg.merge(df_cat_agg, how="inner", on="customer_ID").merge(df_diff_agg, how="inner", on="customer_ID")

    del df_num_agg, df_cat_agg, df_diff_agg
    gc.collect()

    return df


@hydra.main(config_path="../config/", config_name="data", version_base="1.2.0")
def _main(cfg: DictConfig) -> NoReturn:
    path = Path(get_original_cwd())
    train = pd.read_parquet(path / "input/amex-default-prediction/train_meta.parquet")
    label = pd.read_csv(path / "input/amex-default-prediction/train_labels.csv")
    test = pd.read_parquet(path / "input/amex-default-prediction/test_meta.parquet")
    path = Path(get_original_cwd()) / cfg.dataset.path

    # build train features
    train = build_features(train)
    train = pd.merge(train, label, how="left", on="customer_ID")

    print(train.shape)

    train.to_parquet(path / f"{cfg.dataset.train}.parquet")

    del train
    gc.collect()

    # build test features
    split_ids = split_dataset(test.customer_ID.unique(), cfg.dataset.num)

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
