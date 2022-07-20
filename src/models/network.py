import gc
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold

from models.base import BaseModel, ModelResult


class TabNetTrainer(BaseModel):
    def __init__(
        self,
        params: Optional[Dict[str, Any]],
        cat_idxs: Optional[List[int]] = None,
        cat_dims: Optional[List[int]] = None,
        search: bool = False,
        **kwargs,
    ):
        self.params = params
        self.search = search
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        super().__init__(**kwargs)

    def _get_default_params(self) -> Dict[str, Any]:
        return {
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=2e-2),
            "scheduler_params": {"step_size": 50, "gamma": 0.9},
            "scheduler_fn": torch.optim.lr_scheduler.StepLR,
            "mask_type": "entmax",  # "sparsemax"
        }

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> TabNetClassifier:
        """method train"""
        model = TabNetClassifier(**self._get_default_params())

        model.fit(
            X_train=X_train.values,
            y_train=y_train.values,
            max_epochs=100,
            patience=2,
            batch_size=1024,
            virtual_batch_size=256,
            num_workers=1,
            drop_last=False,
            eval_set=[
                (X_train.to_numpy(), y_train.to_numpy()),
                (X_valid.to_numpy(), y_valid.to_numpy()),
            ],
            eval_name=["train", "val"],
            eval_metric=["logloss"],
        )

        return model

    def train(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        thershold: float = 0.4,
        verbose: Union[bool, int] = False,
    ) -> ModelResult:
        """
        Train data
            Parameter:
                train_x: train dataset
                train_y: target dataset
                groups: group fold parameters
                params: lightgbm' parameters
                verbose: log lightgbm' training
            Return:
                True: Finish Training
        """

        models = dict()
        scores = dict()

        str_kf = StratifiedKFold(n_splits=self.fold, shuffle=True, random_state=42)
        splits = str_kf.split(train_x, train_y)

        oof_preds = np.zeros(train_x.shape[0])

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

            # model
            model = self._train(X_train, y_train, X_valid, y_valid)
            models[f"fold_{fold}"] = model

            # validation
            oof_preds[valid_idx] = model.predict_proba(X_valid.to_numpy())[:, 1]

            score = self.metric(
                pd.DataFrame({"target": y_valid.to_numpy()}),
                pd.Series(oof_preds[valid_idx], name="prediction"),
            )
            print(f"Fold{fold} score: {score}")
            scores[f"fold_{fold}"] = score
            gc.collect()

            del X_train, X_valid, y_train, y_valid

        oof_score = self.metric(
            pd.DataFrame({"target": train_y.to_numpy()}),
            pd.Series(oof_preds, name="prediction"),
        )
        logging.info(f"OOF Score: {oof_score}")

        self.result = ModelResult(
            oof_preds=oof_preds,
            models=models,
            preds=None,
            scores={"oof_score": oof_score, "KFold_scores": scores},
        )

        return self.result
