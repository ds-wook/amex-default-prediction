import warnings
from typing import NoReturn, Optional

import lightgbm as lgb
import pandas as pd
import wandb.catboost as wandb_cb
import wandb.lightgbm as wandb_lgb
import wandb.xgboost as wandb_xgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from lightgbm import Booster

from evaluation.evaluate import CatBoostEvalMetricAmex, lgb_amex_metric, xgb_amex_metric
from models.base import BaseModel

warnings.filterwarnings("ignore")


class LightGBMTrainer(BaseModel):
    def __init__(self, **kwargs) -> NoReturn:
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
            params=dict(self.config.model.params),
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            verbose_eval=self.config.model.verbose,
            num_boost_round=self.config.model.num_boost_round,
            callbacks=[wandb_lgb.wandb_callback()],
            feval=lgb_amex_metric,
        )

        wandb_lgb.log_summary(model)

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, **kwargs) -> NoReturn:
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
            data=X_train, label=y_train, cat_features=self.config.features.cat_features
        )
        valid_data = Pool(
            data=X_valid, label=y_valid, cat_features=self.config.features.cat_features
        )

        model = CatBoostClassifier(
            random_state=self.config.model.seed,
            cat_features=self.config.dataset.cat_features,
            eval_metric=CatBoostEvalMetricAmex(),
            **self.config.model.params,
        )
        model.fit(
            train_data,
            eval_set=valid_data,
            early_stopping_rounds=self.config.model.early_stopping_rounds,
            verbose=self.config.model.verbose,
            callbacks=[wandb_cb.WandbCallback()],
        )

        return model


class XGBoostTrainer(BaseModel):
    def __init__(self, **kwargs) -> NoReturn:
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
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
        watchlist = [(dtrain, "train"), (dvalid, "eval")]

        model = xgb.train(
            dict(self.config.model.params),
            dtrain=dtrain,
            evals=watchlist,
            feval=xgb_amex_metric,
            maximize=True,
            callbacks=[wandb_xgb.WandbCallback()],
            num_boost_round=self.config.model.num_boost_round,
            early_stopping_rounds=self.config.model.early_stopping_rounds,
            verbose_eval=self.config.model.verbose,
        )

        return model
