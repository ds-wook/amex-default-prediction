import logging
import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from features.build import make_trick
from utils.utils import reduce_mem_usage

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

    train = pd.read_pickle(path / config.dataset.train, compression="gzip")
    train_x = train.drop(columns=[config.dataset.drop_features, config.dataset.target])
    train_y = train[config.dataset.target]

    train_x = make_trick(train_x)
    train_x = reduce_mem_usage(train_x)
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
    test = pd.read_pickle(path / f"{config.dataset.test}_{num}.pkl", compression="gzip")
    test_x = test.drop(columns=[config.dataset.drop_features])
    test_x = make_trick(test_x)
    test_x = reduce_mem_usage(test_x)

    logging.info(f"test: {test_x.shape}")

    return test_x


def split_test_dataset(config: DictConfig) -> None:
    """
    Split test dataset
    Args:
        config: config file
    """
    path = Path(get_original_cwd()) / config.dataset.path
    logging.info("Loading test dataset...")
    test = pd.read_pickle(path / config.dataset.test, compression="gzip")

    for i in range(10):
        test.iloc[i : (i + 1) * 100000].to_pickle(
            path / f"part_test_{i}.pkl", compression="gzip"
        )
