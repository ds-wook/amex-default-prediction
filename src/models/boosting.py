import pickle
import warnings
from typing import Optional

import lightgbm as lgb
import pandas as pd
import wandb.catboost as wandb_cb
import wandb.lightgbm as wandb_lgb
import wandb.xgboost as wandb_xgb
from catboost import CatBoostClassifier, Pool
from lightgbm import Booster
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

from evaluation.evaluate import CatBoostEvalMetricAmex, lgb_amex_metric, xgb_amex_metric
from models.base import BaseModel

warnings.filterwarnings("ignore")


class DartEarlyStopping:
    def __init__(self, data_name: str, monitor_metric: str, stopping_round: int):
        self.data_name = data_name
        self.monitor_metric = monitor_metric
        self.stopping_round = stopping_round
        self.best_score = None
        self.best_model = None
        self.best_score_list = []
        self.best_iter = 0

    def _is_higher_score(self, metric_score: float, is_higher_better: float) -> bool:
        if self.best_score is None:
            return True
        return (
            (self.best_score < metric_score)
            if is_higher_better
            else (self.best_score > metric_score)
        )

    def _deepcopy(self, x):
        return pickle.loads(pickle.dumps(x))

    def __call__(self, env):
        evals = env.evaluation_result_list
        for data, metric, score, is_higher_better in evals:
            if data != self.data_name or metric != self.monitor_metric:
                continue
            if not self._is_higher_score(score, is_higher_better):
                if env.iteration - self.best_iter > self.stopping_round:
                    eval_result_str = "\t".join(
                        [
                            lgb.callback._format_eval_result(x)
                            for x in self.best_score_list
                        ]
                    )
                    lgb.basic._log_info(
                        f"Early stopping, best iteration is:\n[{self.best_iter+1}]\t{eval_result_str}"
                    )
                    lgb.basic._log_info(
                        'You can get best model by "DartEarlyStopping.best_model"'
                    )
                    raise lgb.callback.EarlyStopException(
                        self.best_iter, self.best_score_list
                    )
                return

            self.best_model = self._deepcopy(env.model)
            self.best_iter = env.iteration
            self.best_score_list = evals
            self.best_score = score
            return
        raise ValueError("monitoring metric not found")


class LightGBMTrainer(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> Booster:
        """
        load train model
        """
        train_set = lgb.Dataset(
            X_train, y_train, categorical_feature=self.config.dataset.cat_features
        )
        valid_set = lgb.Dataset(
            X_valid, y_valid, categorical_feature=self.config.dataset.cat_features
        )

        es = DartEarlyStopping("valid_1", "amex", stopping_round=500)

        model = lgb.train(
            params=dict(self.config.models.params),
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            verbose_eval=self.config.models.verbose,
            num_boost_round=self.config.models.num_boost_round,
            callbacks=[wandb_lgb.wandb_callback(), es],
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            feval=lgb_amex_metric,
        )

        wandb_lgb.log_summary(model)

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> CatBoostClassifier:
        """
        load train model
        """
        train_data = Pool(
            data=X_train, label=y_train, cat_features=self.config.dataset.cat_features
        )
        valid_data = Pool(
            data=X_valid, label=y_valid, cat_features=self.config.dataset.cat_features
        )

        model = CatBoostClassifier(
            random_state=self.config.models.seed,
            cat_features=self.config.dataset.cat_features,
            eval_metric=CatBoostEvalMetricAmex(),
            **self.config.models.params,
        )
        model.fit(
            train_data,
            eval_set=valid_data,
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            verbose=self.config.models.verbose,
            callbacks=[wandb_cb.WandbCallback()],
        )

        return model


class XGBoostTrainer(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> XGBClassifier:
        """
        load train model
        """

        model = XGBClassifier(
            random_state=self.config.models.seed, **self.config.models.params
        )

        model.fit(
            X_train,
            y_train,
            eval_metric=xgb_amex_metric,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=self.config.models.early_stopping_rounds,
            verbose=self.config.models.verbose,
            callbacks=[wandb_xgb.wandb_callback()],
        )

        return model


class HistGradientBoostingTrainer(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> HistGradientBoostingClassifier:
        model = HistGradientBoostingClassifier(**self.config.models.params)
        model.fit(X_train, y_train)
        return model
