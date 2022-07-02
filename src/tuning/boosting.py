import pandas as pd
from lightgbm import LGBMClassifier
from omegaconf import DictConfig
from optuna.trial import Trial
from sklearn.model_selection import train_test_split

from amex.evaluation.evaluate import lgb_amex_metric
from amex.tuning.base import BaseTuner


class LightGBMTuner(BaseTuner):
    def __init__(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        config: DictConfig,
    ):
        self.train_x = train_x
        self.train_y = train_y
        super().__init__(config)

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
            "reg_alpha": trial.suggest_float("reg_alpha", *config.search.reg_alpha),
            "reg_lambda": trial.suggest_float("reg_lambda", *config.search.reg_lambda),
            "num_leaves": trial.suggest_int("num_leaves", *config.search.num_leaves),
            "max_bin": trial.suggest_int("max_bin", *config.search.max_bin),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *config.search.colsample_bytree
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", *config.search.min_child_weight
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", *config.search.learning_rate
            ),
            "n_estimators": trial.suggest_float(
                "n_estimators", *config.search.n_estimators
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
