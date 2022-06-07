import gc
import pickle
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def create_categorical_train(train: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
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


def create_categorical_test(test: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
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


def create_features(
    df: pd.DataFrame, config: DictConfig
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
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
        create_categorical_train(df_categorical, config)
        if config.dataset.is_train
        else create_categorical_test(df_categorical, config)
    )

    df = pd.concat(
        [df_last[features_not_cat], df_categorical, df_avg, df_min, df_max], axis=1
    )
    del df_avg, df_max, df_min, df_last, df_categorical, cid, last, features_not_cat

    if config.dataset.is_train:
        return df, target

    return df


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
