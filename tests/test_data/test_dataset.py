import logging
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from test_features.test_build import (
    add_diff_features,
    add_trick_features,
    add_rate_features,
    add_customized_features,
    create_categorical_train,
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
    logging.info("Loading train dataset...")

    train = pd.read_parquet(path / f"{config.dataset.train}.parquet")
    train_y = train[config.dataset.target]
    train_x = train.drop(columns=[*config.dataset.drop_features, config.dataset.target])
    train_x = add_trick_features(train_x)
    train_x = add_diff_features(train_x)
    train_x = add_rate_features(train_x)
    train_x = add_customized_features(train_x)
    train_x = create_categorical_train(train_x, config)
    logging.info(f"train: {train_x.shape}, target: {train_y.shape}")

    return train_x, train_y


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
