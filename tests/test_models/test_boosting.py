import warnings
from typing import NoReturn, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from test_evaluation.test_evaluate import lgb_amex_metric
from test_models.test_base import BaseModel


warnings.filterwarnings("ignore")


class LightGBMTrainer(BaseModel):
    def __init__(self, **kwargs) -> NoReturn:
        super().__init__(**kwargs)

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
        hess = preds * (1 - preds) * (1 + weights * labels - labels)

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
            callbacks=[self.save_dart_model()],
        )

        return model
