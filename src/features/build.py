import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def categorize_train(train: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.dataset.encoder

    le_encoder = LabelEncoder()

    for cat_feature in tqdm(config.dataset.cat_features):
        train[cat_feature] = le_encoder.fit_transform(train[cat_feature])

        with open(path / f"{cat_feature}.pkl", "wb") as f:
            pickle.dump(le_encoder, f)

    return train


def categorize_test(test: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    path = Path(get_original_cwd()) / config.dataset.encoder

    for cat_feature in tqdm(config.dataset.cat_features):
        le_encoder = pickle.load(open(path / f"{cat_feature}.pkl", "rb"))
        test[cat_feature] = le_encoder.transform(test[cat_feature])
        gc.collect()

    return test


def create_features(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Create average features
    Args:
        df: dataframe
        config: config file
    Returns:
        dataframe
    """
    cid = pd.Categorical(df.pop("customer_ID"), ordered=True)
    last = cid != np.roll(cid, -1)  # mask for last statement of every customer

    if config.dataset.is_train:
        target = df.loc[last, config.dataset.target]
        gc.collect()

    df_avg = (
        df[config.dataset.features_avg]
        .groupby(cid)
        .mean()
        .rename(columns={f: f"{f}_avg" for f in config.dataset.features_avg})
    )
    gc.collect()

    df_max = (
        df.groupby(cid)
        .max()[config.dataset.features_max]
        .rename(columns={f: f"{f}_max" for f in config.dataset.features_max})
    )
    gc.collect()

    df_min = (
        df.groupby(cid)
        .min()[config.dataset.features_min]
        .rename(columns={f: f"{f}_min" for f in config.dataset.features_min})
    )
    gc.collect()

    df_last = (
        df.loc[last, config.dataset.features_last]
        .rename(columns={f: f"{f}_last" for f in config.dataset.features_last})
        .set_index(np.asarray(cid[last]))
    )
    gc.collect()

    df_categorical = df_last[config.dataset.features_categorical].astype(object)
    features_not_cat = [
        f for f in df_last.columns if f not in config.dataset.features_categorical
    ]

    df_categorical = (
        categorize_train(df_categorical, config)
        if config.dataset.is_train
        else categorize_test(df_categorical, config)
    )

    df = pd.concat(
        [df_last[features_not_cat], df_categorical, df_avg, df_min, df_max], axis=1
    )
    del df_avg, df_max, df_min, df_last, df_categorical, cid, last, features_not_cat

    if config.dataset.is_train:
        return df, target

    return df


def add_features(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Add features

    Args:
        df: Dataset
        config: config file
    Return:
       feature engineered Dataset
    """
    df_num_agg = df.groupby("customer_ID")[config.dataset.num_features].agg(
        ["mean", "std", "min", "max", "last"]
    )
    gc.collect()

    df_num_agg.columns = ["_".join(x) for x in df_num_agg.columns]

    df_cat_agg = df.groupby("customer_ID")[config.dataset.cat_features].agg(
        ["count", "last", "nunique"]
    )
    gc.collect()

    df_cat_agg.columns = ["_".join(x) for x in df_cat_agg.columns]

    if config.dataset.is_train:
        df_target = (
            df.groupby("customer_ID")
            .tail(1)
            .set_index("customer_ID", drop=True)
            .sort_index()["target"]
        )
        gc.collect()
        df = pd.concat([df_num_agg, df_cat_agg, df_target], axis=1)
        del df_num_agg, df_cat_agg, df_target
    else:
        df = pd.concat([df_num_agg, df_cat_agg], axis=1)
        del df_num_agg, df_cat_agg

    return df


def make_trick(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create nan feature
    Args:
        df: dataframe
    Returns:
        dataframe
    """

    for col in df.columns:
        if df[col].dtype == "float16":
            df[col] = df[col].astype("float32").round(decimals=2).astype("float16")
        gc.collect()

    return df
