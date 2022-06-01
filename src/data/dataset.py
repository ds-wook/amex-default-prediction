import logging
import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from features.build import categorize_test, categorize_train, make_nan_feature


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
    train = categorize_train(train, config)
    train = make_nan_feature(train)
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
    logging.info("Loading test dataset...")
    test = pd.read_pickle(path / config.dataset.test, compression="gzip")
    test = make_nan_feature(test)
    test_x = categorize_test(test, config)
    logging.info(f"test: {test_x.shape}")

    return test_x
