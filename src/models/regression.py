from lib2to3.pytree import Base
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.linear_model import BayesianRidge

from models.base import BaseModel


class BayesianRidgeTrainer(BaseModel):
    def __init__(self, config: DictConfig, metric: Callable, search: bool = False):
        super().__init__(config, metric, search)

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: Optional[pd.DataFrame] = None, y_valid: Optional[pd.Series] = None,) -> BayesianRidge:
        """
        load train model
        """
        model = BayesianRidge()
        
