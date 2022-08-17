import logging
import warnings
from pathlib import Path
from typing import Callable, NoReturn, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from test_evaluation.test_evaluate import lgb_amex_metric
from test_models.test_base import BaseModel
from test_models.test_callbacks import CallbackEnv

warnings.filterwarnings("ignore")


class LightGBMTrainer(BaseModel):
    def __init__(self, **kwargs) -> NoReturn:
        super().__init__(**kwargs)

    def _save_dart_model(self) -> Callable[[CallbackEnv], NoReturn]:
        def callback(env: CallbackEnv) -> NoReturn:
            iteration = env.iteration
            score = (
                env.evaluation_result_list[1][2]
                if self.config.model.loss.is_customized
                else env.evaluation_result_list[3][2]
            )
            if self._max_score < score:
                self._max_score = score
                logging.info(
                    f"High Score: iteration {iteration}, score={self._max_score}"
                )
                env.model.save_model(
                    Path(get_original_cwd())
                    / self.config.model.path
                    / f"{self.config.model.result}_fold{self._num_fold_iter}.lgb"
                )

        callback.order = 0
        return callback

    def _weighted_logloss(
        self, preds: np.ndarray, dtrain: lgb.Dataset
    ) -> Tuple[float, float]:
        """
        weighted logloss for dart
        Args:
            preds: prediction
            dtrain: lgb.Dataset
            mult_no4prec: weight for no4prec
            max_weights: max weight for no4prec
        Returns:
            gradient, hessian
        """
        eps = 1e-16
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))

        # top 4%
        labels_mat = np.transpose(np.array([np.arange(len(labels)), labels, preds]))
        pos_ord = labels_mat[:, 2].argsort()[::-1]
        labels_mat = labels_mat[pos_ord]
        weights_4perc = np.where(labels_mat[:, 1] == 0, 20, 1)
        top4 = np.cumsum(weights_4perc) <= int(0.04 * np.sum(weights_4perc))
        top4 = top4[labels_mat[:, 0].argsort()]

        weights = (
            1
            + np.exp(
                -self.config.model.loss.mult_no4prec
                * np.linspace(self.config.model.loss.max_weights - 1, 0, len(top4))
            )[labels_mat[:, 0].argsort()]
        )
        # Set to one weights of positive labels in top 4perc
        weights[top4 & (labels == 1.0)] = 1.0
        # Set to one weights of negative labels
        weights[(labels == 0.0)] = 1.0

        grad = preds * (1 + weights * labels - labels) - (weights * labels)
        hess = np.maximum(
            ((2 * preds - 3 * preds**2) * (1 + weights * labels - labels)), eps
        )

        return grad, hess

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> lgb.Booster:
        """
        load train model
        """
        train_set = lgb.Dataset(
            data=X_train,
            label=y_train,
            categorical_feature=self.config.features.cat_features,
        )
        valid_set = lgb.Dataset(
            data=X_valid,
            label=y_valid,
            categorical_feature=self.config.features.cat_features,
        )

        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=dict(self.config.model.params),
            verbose_eval=self.config.model.verbose,
            num_boost_round=self.config.model.num_boost_round,
            feval=lgb_amex_metric,
            fobj=self._weighted_logloss
            if self.config.model.loss.is_customized
            else None,
            callbacks=[self._save_dart_model()],
        )

        model = lgb.Booster(
            model_file=Path(get_original_cwd())
            / self.config.model.path
            / f"{self.config.model.result}_fold{self._num_fold_iter}.lgb"
        )

        return model
