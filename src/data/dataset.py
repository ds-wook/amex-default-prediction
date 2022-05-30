import gc
import logging
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from features.build import (
    categorical_test_encoding,
    categorical_train_encoding,
    create_avg_features,
)

warnings.filterwarnings("ignore")


def load_train_dataset(config: DictConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load train dataset
    Args:
        config: config
    Returns:
        train_x: train dataset
        train_y: train target
    """
    path = Path(get_original_cwd()) / config.dataset.path
    logging.info("Loading dataset...")

    train = pd.read_feather(path / config.dataset.train)
    train_x, train_y = create_avg_features(train, config)

    logging.info(f"train: {train_x.shape}, target: {train_y.shape}")

    return train_x, train_y


def load_cat_train_dataset(config: DictConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load train dataset
    Args:
        config: config
    Returns:
        train_x: train dataset
        train_y: train target
    """
    path = Path(get_original_cwd()) / config.dataset.path
    logging.info("Loading dataset...")

    train = pd.read_pickle(path / config.dataset.train, compression="gzip")
    train = categorical_train_encoding(train, config.dataset.cat_features)
    train_x = train.drop(columns=config.dataset.target)
    train_y = train[config.dataset.target]

    logging.info(f"train: {train_x.shape}, target: {train_y.shape}")

    return train_x, train_y


def load_test_dataset(config: DictConfig) -> pd.DataFrame:
    """
    Load train dataset
    Args:
        config: config
    Returns:
        test_x: test dataset
    """
    path = Path(get_original_cwd()) / config.dataset.path

    logging.info("Loading dataset...")
    test = pd.read_feather(path / config.dataset.test)
    gc.collect()

    cid = pd.Categorical(test.pop("customer_ID"), ordered=True)
    last = cid != np.roll(cid, -1)  # mask for last statement of every customer
    df_avg = (
        test[config.dataset.features_avg]
        .groupby(cid)
        .mean()
        .rename(columns={f: f"{f}_avg" for f in config.dataset.features_avg})
    )
    gc.collect()

    test = (
        test.loc[last, config.dataset.features_last]
        .rename(columns={f: f"{f}_last" for f in config.dataset.features_last})
        .set_index(np.asarray(cid[last]))
    )
    gc.collect()

    test_x = pd.concat([test, df_avg], axis=1)
    del df_avg, cid, last, test

    logging.info(f"test: {test_x.shape}")

    return test_x


def load_cat_test_dataset(config: DictConfig) -> pd.DataFrame:
    """
    Load train dataset
    Args:
        config: config
    Returns:
        test_x: test dataset
    """
    path = Path(get_original_cwd()) / config.dataset.path
    logging.info("Loading dataset...")
    train = pd.read_pickle(path / config.dataset.train, compression="gzip")
    test = pd.read_pickle(path / config.dataset.test, compression="gzip")
    test_x = categorical_test_encoding(train, test, config.dataset.cat_features)
    del train
    logging.info(f"test: {test_x.shape}")

    return test_x
