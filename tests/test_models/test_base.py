import gc
import logging
import pickle
import warnings
from abc import ABCMeta, abstractclassmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, NoReturn, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from test_evaluation.test_evaluate import amex_metric
from test_models.test_callbacks import CallbackEnv

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: Dict[str, Any]
    scores: Dict[str, Dict[str, float]]


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config: DictConfig, search: bool = False) -> NoReturn:
        self.config = config
        self.search = search
        self.__max_score = 0.0
        self.__num_fold_iter = 0
        self.result = None

    @abstractclassmethod
    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> NoReturn:
        """
        Trains the model.
        """
        raise NotImplementedError

    def save_dart_model(self) -> Callable[[CallbackEnv], NoReturn]:
        def callback(env: CallbackEnv) -> NoReturn:
            iteration = env.iteration
            score = (
                env.evaluation_result_list[1][2]
                if self.config.model.loss.is_customized
                else env.evaluation_result_list[3][2]
            )
            if self.__max_score < score:
                self.__max_score = score
                logging.info(
                    f"High Score: iteration {iteration}, score={self.__max_score}"
                )

                env.model.save_model(
                    Path(get_original_cwd())
                    / self.config.model.path
                    / f"{self.config.model.result}_fold{self.__num_fold_iter}.lgb"
                )

        callback.order = 0
        return callback

    def save_model(self) -> NoReturn:
        """
        Save model
        """
        model_path = (
            Path(get_original_cwd()) / self.config.model.path / self.config.model.name
        )

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
            self.__max_score = 0.0
            self.__num_fold_iter = fold

            # split train and validation data
            X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

            # model
            model = self._train(X_train, y_train, X_valid, y_valid)
            model = (
                lgb.Booster(
                    model_file=Path(get_original_cwd())
                    / self.config.model.path
                    / f"{self.config.model.result}_fold{self.__num_fold_iter}.lgb"
                )
                if isinstance(model, lgb.Booster)
                else model
            )
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
            score = amex_metric(y_valid, oof_preds[valid_idx])

            scores[f"fold_{fold}"] = score

            if not self.search:
                logging.info(f"Best Score: {self.__max_score}")
                logging.info(f"Fold {fold}: {score}")

            del X_train, X_valid, y_train, y_valid, model
            gc.collect()

        oof_score = amex_metric(train_y, oof_preds)
        logging.info(f"OOF Score: {oof_score}")
        logging.info(f"CV means: {np.mean(list(scores.values()))}")
        logging.info(f"CV std: {np.std(list(scores.values()))}")

        self.result = ModelResult(
            oof_preds=oof_preds,
            models=models,
            scores={"oof_score": oof_score, "KFold_scores": scores},
        )

        return self.result
