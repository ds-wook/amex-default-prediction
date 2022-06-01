import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def categorical_train_encoding(train: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    path = Path(get_original_cwd())

    le_encoder = LabelEncoder()

    for categorical_feature in tqdm(config.dataset.features_categorical):
        train[categorical_feature] = le_encoder.fit_transform(
            train[categorical_feature]
        )

    with open(path / config.dataset.encoder, "wb") as f:
        pickle.dump(le_encoder, f)

    return train


def categorical_test_encoding(test: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Categorical encoding
    Args:
        df: dataframe
        cat_col: list of categorical columns
    Returns:
        dataframe
    """
    path = Path(get_original_cwd())

    le_encoder = pickle.load(open(path / config.dataset.encoder, "rb"))

    for categorical_feature in tqdm(config.dataset.features_categorical):
        test[categorical_feature] = le_encoder.transform(test[categorical_feature])

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
        categorical_train_encoding(df_categorical, config)
        if config.dataset.is_train
        else categorical_test_encoding(df_categorical, config)
    )

    df = pd.concat(
        [df_last[features_not_cat], df_categorical, df_avg, df_min, df_max], axis=1
    )
    del df_avg, df_max, df_min, df_last, df_categorical, cid, last, features_not_cat

    if config.dataset.is_train:
        return df, target

    return df
