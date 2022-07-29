from typing import Callable

import lightgbm as lgb
import pandas as pd
from omegaconf import DictConfig
from optuna.trial import FrozenTrial
from sklearn.model_selection import train_test_split

from evaluation.evaluate import lgb_amex_metric
from tuning.base import BaseTuner


class LightGBMTuner(BaseTuner):
    def __init__(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        config: DictConfig,
        metric: Callable,
    ):
        self.train_x = train_x
        self.train_y = train_y
        super().__init__(config, metric)

    def _objective(self, trial: FrozenTrial) -> float:
        """
        Objective function
        Args:
            trial: trial object
            config: config object
        Returns:
            metric score
        """
        # trial parameters
        params = {
            "objective": "binary",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": trial.suggest_int("seed", **self.config.tuning.params.seed),
            "learning_rate": trial.suggest_float(
                "learning_rate", **self.config.tuning.params.learning_rate
            ),
            "lambda_l1": trial.suggest_loguniform(
                "lambda_l1", **self.config.tuning.params.lambda_l1
            ),
            "lambda_l2": trial.suggest_loguniform(
                "lambda_l2", **self.config.tuning.params.lambda_l2
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", **self.config.tuning.params.num_leaves
            ),
            "feature_fraction": trial.suggest_uniform(
                "feature_fraction", **self.config.tuning.params.feature_fraction
            ),
            "bagging_fraction": trial.suggest_uniform(
                "bagging_fraction", **self.config.tuning.params.bagging_fraction
            ),
            "bagging_freq": trial.suggest_int(
                "bagging_freq", **self.config.tuning.params.bagging_freq
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_child_samples", **self.config.tuning.params.min_child_samples
            ),
        }

        # train
        X_train, X_valid, y_train, y_valid = train_test_split(
            self.train_x,
            self.train_y,
            random_state=self.config.dataset.seed,
            stratify=self.train_y,
        )

        train_set = lgb.Dataset(
            X_train,
            y_train,
            categorical_feature=self.config.dataset.cat_features,
        )
        valid_set = lgb.Dataset(
            X_valid,
            y_valid,
            categorical_feature=self.config.dataset.cat_features,
        )

        model = lgb.train(
            params=params,
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            verbose_eval=self.config.model.verbose,
            num_boost_round=self.config.model.num_boost_round,
            feval=lgb_amex_metric,
        )

        y_preds = model.predict(X_valid)
        score = self.metric(y_valid, y_preds)

        return score
