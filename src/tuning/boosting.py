from typing import Optional

import pandas as pd
from lightgbm import LGBMClassifier
from neptune.new import Run
from omegaconf import DictConfig
from optuna.trial import Trial
from sklearn.model_selection import train_test_split

from evaluation.evaluate import lgb_amex_metric
from tuning.base import BaseTuner


class LightGBMTuner(BaseTuner):
    def __init__(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        config: DictConfig,
        run: Optional[Run] = None,
    ):
        self.train_x = train_x
        self.train_y = train_y
        super().__init__(config, run)

    def _objective(self, trial: Trial, config: DictConfig) -> float:
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
            "max_depth": trial.suggest_int("max_depth", *config.search.max_depth),
            "subsample": trial.suggest_float("subsample", *config.search.subsample),
            "gamma": trial.suggest_float("gamma", *config.search.gamma),
            "reg_alpha": trial.suggest_float("reg_alpha", *config.search.reg_alpha),
            "reg_lambda": trial.suggest_float("reg_lambda", *config.search.reg_lambda),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *config.search.colsample_bytree
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *config.search.min_child_weight
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", *config.search.learning_rate
            ),
        }

        # train
        X_train, X_valid, y_train, y_valid = train_test_split(
            self.train_x,
            self.train_y,
            random_state=self.config.model.seed,
            stratify=self.train_y,
        )

        model = LGBMClassifier(random_state=self.config.model.seed, **params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric=lgb_amex_metric,
            early_stopping_rounds=self.config.model.early_stopping_rounds,
            verbose=self.config.model.verbose,
        )

        y_preds = model.predict_proba(X_valid)[:, 1]
        score = self.metric(
            pd.DataFrame({"target": y_valid.to_numpy()}),
            pd.Series(y_preds, name="prediction"),
        )

        return score