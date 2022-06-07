import warnings

import pandas as pd
import wandb.catboost as wandb_cb
import wandb.lightgbm as wandb_lgb
import wandb.xgboost as wandb_xgb
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from evaluation.evaluate import CatBoostEvalMetricAmex, lgb_amex_metric, xgb_amex_metric
from models.base import BaseModel

warnings.filterwarnings("ignore")


class LightGBMTrainer(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> LGBMClassifier:
        """
        load train model
        """

        model = LGBMClassifier(
            random_state=self.config.model.seed, **self.config.model.params
        )

        if self.config.model.params.boosting_type == "dart":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric=lgb_amex_metric,
                verbose=self.config.model.verbose,
                callbacks=[wandb_lgb.wandb_callback()],
            )

        else:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric=lgb_amex_metric,
                early_stopping_rounds=self.config.model.early_stopping_rounds,
                verbose=self.config.model.verbose,
                callbacks=[wandb_lgb.wandb_callback()],
            )

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> XGBClassifier:
        """
        load train model
        """

        model = XGBClassifier(
            random_state=self.config.model.seed, **self.config.model.params
        )

        if self.config.model.params.booster == "dart":
            model.fit(
                X_train,
                y_train,
                eval_metric=xgb_amex_metric,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                verbose=self.config.model.verbose,
                callbacks=[wandb_xgb.wandb_callback()],
            )

        else:
            model.fit(
                X_train,
                y_train,
                eval_metric=xgb_amex_metric,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                early_stopping_rounds=self.config.model.early_stopping_rounds,
                verbose=self.config.model.verbose,
                callbacks=[wandb_xgb.wandb_callback()],
            )

        return model
