import logging
import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pandas import DataFrame

from features.build import categorical_encoding

warnings.filterwarnings("ignore")


def load_dataset(config: DictConfig) -> Tuple[DataFrame, pd.Series, pd.DataFrame]:
    """
    Load train dataset
    Args:
        config: config
    Returns:
        train_x: train dataset
        train_y: train target
        test_x: test dataset
    """
    path = Path(get_original_cwd()) / config.dataset.path

    logging.info("Loading dataset...")
    train = pd.read_feather(path / config.dataset.train)
    test = pd.read_feather(path / config.dataset.test)
    train = (
        train.groupby("customer_ID")
        .tail(1)
        .set_index("customer_ID", drop=True)
        .sort_index()
        .drop(["S_2"], axis="columns")
    )

    test = (
        test.groupby("customer_ID")
        .tail(1)
        .set_index("customer_ID", drop=True)
        .sort_index()
        .drop(["S_2"], axis="columns")
    )

    train_y = train[config.dataset.target]
    train = train.drop(columns=config.dataset.target)
    train_x, test_x = categorical_encoding(
        train, test, cat_col=config.dataset.cat_features
    )

    return train_x, train_y, test_x
