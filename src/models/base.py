from __future__ import annotations

import gc
import logging
import pickle
import warnings
from abc import ABCMeta, abstractclassmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, NoReturn

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: Dict[str, Any]
    scores: Dict[str, Dict[str, float]]


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config: DictConfig, metric: Callable[[np.ndarray, np.ndarray], float], search: bool = False):
        self.config = config
        self.metric = metric
        self.search = search
        self._max_score = 0.0
        self._num_fold_iter = 0
        self.result = None

    @abstractclassmethod
    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> NoReturn:
        """
        Trains the model.
        """
        raise NotImplementedError

    def save_model(self):
        """
        Save model
        """
        model_path = Path(get_original_cwd()) / self.config.model.path / self.config.model.result

        with open(model_path, "wb") as output:
            pickle.dump(self.result, output, pickle.HIGHEST_PROTOCOL)

    def train(self, train_x: pd.DataFrame, train_y: pd.Series) -> ModelResult:
        """
        Train data
        Args:
            train_x: train dataset
            train_y: target dataset
        Return:
            Model Result
        """
        models = dict()
        scores = dict()
        folds = self.config.model.fold
        seed = self.config.dataset.seed

        str_kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = str_kf.split(train_x, train_y)
        oof_preds = np.zeros(len(train_x))

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            # save dart parameters
            self._max_score = 0.0
            self._num_fold_iter = fold

            # split train and validation data
            X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

            # model
            model = self._train(X_train, y_train, X_valid, y_valid)
            models[f"fold_{fold}"] = model

            # validation
            oof_preds[valid_idx] = (
                model.predict(X_valid)
                if isinstance(model, lgb.Booster)
                else model.predict(xgb.DMatrix(X_valid))
                if isinstance(model, xgb.Booster)
                else model.predict_proba(X_valid)[:, 1]
            )

            # score
            score = self.metric(y_valid, oof_preds[valid_idx])

            scores[f"fold_{fold}"] = score

            if not self.search:
                logging.info(f"Fold {fold}: {score}")

            del X_train, X_valid, y_train, y_valid, model
            gc.collect()

        oof_score = self.metric(train_y, oof_preds)
        logging.info(f"OOF Score: {oof_score}")
        logging.info(f"CV means: {np.mean(list(scores.values()))}")
        logging.info(f"CV std: {np.std(list(scores.values()))}")

        self.result = ModelResult(
            oof_preds=oof_preds,
            models=models,
            scores={"oof_score": oof_score, "KFold_scores": scores},
        )

        return self.result
