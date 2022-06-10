from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


class StackingDataLoder:
    def __init__(
        self,
        config: DictConfig,
        train_data: np.ndarray,
        test_data: np.ndarray,
    ) -> None:
        self.config = config
        self.train_data = train_data
        self.test_data = test_data

    def make_train_dataset(self):
        path = Path(get_original_cwd()) / self.config.dataset.path
        train_label = pd.read_csv(path / self.config.dataset.train_label)
        train_data = pd.DataFrame(
            self.train_data,
            columns=[f"oof_preds_{i + 1}" for i in range(self.train_data.shape[1])],
        )
        train = pd.concat([train_label, train_data], axis=1)

        return train

    def make_test_dataset(self):
        test = pd.DataFrame(
            self.test_data,
            columns=[f"preds_{i + 1}" for i in range(self.test_data.shape[1])],
        )
        return test
