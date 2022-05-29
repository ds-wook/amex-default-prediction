import gc
import logging
import pickle
import warnings
from abc import ABCMeta, abstractclassmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold


warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: Dict[str, Any]
    scores: Dict[str, float]


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config: DictConfig, metric: Callable, search: bool = False):
        self.config = config
        self.metric = metric
        self.search = search
        self.result = None

    @abstractclassmethod
    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ):
        """
        Trains the model.
        """
        raise NotImplementedError

    def save_model(self):
        """
        Save model
        """
        model_path = Path(get_original_cwd()) / self.config.model.path

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

        str_kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        splits = str_kf.split(train_x, train_y)
        oof_preds = np.zeros((train_x.shape[0],))

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            # split train and validation data
            X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]
            wandb.init(
                entity=self.config.experiment.entity,
                project=self.config.experiment.project,
                name=self.config.experiment.name + f"_fold_{fold}",
                reinit=True,
            )
            # model
            model = self._train(
                X_train,
                y_train,
                X_valid,
                y_valid,
            )

            models[f"fold_{fold}"] = model

            # validation
            oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
            score = self.metric(
                pd.DataFrame({"target": y_valid.to_numpy()}),
                pd.Series(oof_preds[valid_idx], name="prediction"),
            )
            scores[f"fold_{fold}"] = score

            if not self.search:
                logging.info(f"Fold {fold}: {score}")

            gc.collect()

            del X_train, X_valid, y_train, y_valid
            # Close run for that fold
            wandb.join()

        oof_score = self.metric(
            pd.DataFrame({"target": train_y.to_numpy()}),
            pd.Series(oof_preds, name="prediction"),
        )
        logging.info(f"OOF Score: {oof_score}")

        self.result = ModelResult(
            oof_preds=oof_preds,
            models=models,
            scores={"oof_score": oof_score, "KFold_scores": scores},
        )

        return self.result
