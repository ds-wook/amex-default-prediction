from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression

from models.base import BaseModel


class AmexStackingDataset:
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
            columns=[f"preds_{i + 1}" for i in range(self.train_data.shape[1])],
        )
        train = pd.concat([train_label, train_data], axis=1)

        return train

    def make_test_dataset(self):
        test = pd.DataFrame(
            self.test_data,
            columns=[f"preds_{i + 1}" for i in range(self.test_data.shape[1])],
        )
        return test


class StackingLogisticRegression(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _train_train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> LogisticRegression:
        """
        Train a logistic regression model
        """
        model = LogisticRegression()

        model.fit(X_train, y_train)

        return model
