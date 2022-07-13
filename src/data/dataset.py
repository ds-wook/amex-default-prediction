import logging
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

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
    logging.info("Loading train dataset...")

    train = pd.read_pickle(
        path / f"{config.dataset.train}.{config.dataset.type}", compression="gzip"
    )
    train_y = train[config.dataset.target]
    train_x = train.drop(columns=[config.dataset.drop_features, config.dataset.target])

    logging.info(f"train: {train_x.shape}, target: {train_y.shape}")

    return train_x, train_y


def load_train_dataset_parquet(config: DictConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load train dataset
    Args:
        config: config
    Returns:
        train_x: train dataset
        train_y: train target
    """
    path = Path(get_original_cwd()) / config.dataset.path
    logging.info("Loading train dataset...")

    train = pd.read_parquet(path / f"{config.dataset.train}.{config.dataset.type}")
    train_y = train[config.dataset.target]
    train_x = train.drop(columns=[config.dataset.drop_features, config.dataset.target])

    logging.info(f"train: {train_x.shape}, target: {train_y.shape}")

    return train_x, train_y


def load_test_dataset(config: DictConfig, num: int = 0) -> pd.DataFrame:
    """
    Load train dataset
    Args:
        config: config
    Returns:
        test_x: test dataset
    """
    path = Path(get_original_cwd()) / config.dataset.path
    logging.info("Loading test dataset...")
    test = pd.read_pickle(
        path / f"{config.dataset.test}_{num}.{config.dataset.type}", compression="gzip"
    )
    test_x = test.drop(columns=[config.dataset.drop_features])
    logging.info(f"test: {test_x.shape}")

    return test_x


def load_test_dataset_parquet(config: DictConfig, num: int = 0) -> pd.DataFrame:
    """
    Load train dataset
    Args:
        config: config
    Returns:
        test_x: test dataset
    """
    path = Path(get_original_cwd()) / config.dataset.path
    logging.info("Loading test dataset...")
    test = pd.read_parquet(path / f"{config.dataset.test}_{num}.{config.dataset.type}")
    test_x = test.drop(columns=[config.dataset.drop_features])
    logging.info(f"test: {test_x.shape}")

    return test_x


# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
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
