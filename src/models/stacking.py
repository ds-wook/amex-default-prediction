from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


class StackingDataLoder:
    def __init__(
        self,
        config: DictConfig,
        train_data: List[np.ndarray],
        test_data: List[np.ndarray],
    ) -> None:
        self.config = config
        self.train_data = train_data
        self.test_data = test_data

    def make_train_dataset(self):
        path = Path(get_original_cwd()) / self.config.dataset.path
        train_label = pd.read_csv(path / self.config.dataset.train_label)
        train_data = pd.DataFrame(
            self.train_data,
            columns=[f"oof_preds_{i}" for i in range(len(self.train_data))],
        )
        train = pd.concat([train_data, train_label], axis=1)

        return train

    def make_test_dataset(self):
        test = pd.DataFrame(
            self.test_data, columns=[f"preds_{i}" for i in range(len(self.test_data))]
        )
        return test
