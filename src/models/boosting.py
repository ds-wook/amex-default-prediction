import warnings
from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb.catboost as wandb_cb
import wandb.lightgbm as wandb_lgb
import wandb.xgboost as wandb_xgb
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

from evaluation.evaluate import CatBoostEvalMetricAmex, lgb_amex_metric, xgb_amex_metric
from models.base import BaseModel

warnings.filterwarnings("ignore")


class LightGBMTrainer(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
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
            random_state=self.config.model.seed, **self.config.model.params
        )

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
        model = HistGradientBoostingClassifier(**self.config.model.params)
        model.fit(X_train, y_train)
        return model


def get_splits_gain(
    tree_num: int = 0,
    parent: int = -1,
    tree: Optional[LGBMClassifier] = None,
    lev: int = 0,
    node_name: Optional[str] = None,
    split_gain: Optional[float] = None,
    reclimit: int = 50000,
) -> Union[Iterable[Tuple[int, float]], Iterable[Tuple[int, float]]]:
    if tree is None:
        raise Exception("No tree present to analyze!")
    for k, v in tree.items():
        if type(v) != dict and k in ["split_feature"]:
            old_parent = parent
            parent = v
            tag = k
            yield tree_num, tag, old_parent, parent, lev, node_name, split_gain
        elif isinstance(v, dict):
            if v.get("split_gain") is None:
                continue
            else:
                tree = v
                lev_inc = lev + 1
                node_name = k
                split_gain = v["split_gain"]
                for result in get_splits_gain(
                    tree_num, parent, tree, lev_inc, node_name, split_gain
                ):
                    yield result
        else:
            continue


def plot_feat_interaction(model: LGBMClassifier) -> None:
    dumped_model = model.booster_.dump_model()
    tree_info = []
    for j in range(0, len(dumped_model["tree_info"])):
        for i in get_splits_gain(tree_num=j, tree=dumped_model["tree_info"][j]):
            tree_info.append(list(i))
    tree_info_df = pd.DataFrame(
        tree_info,
        columns=[
            "TreeNo",
            "Type",
            "ParentFeature",
            "SplitOnfeature",
            "Level",
            "TreePos",
            "Gain",
        ],
    )
    lgbm_feat_dict = dict(enumerate(dumped_model["feature_names"]))
    lgbm_feat_dict[-1] = "base"
    tree_info_df["ParentFeature"].replace(lgbm_feat_dict, inplace=True)
    tree_info_df["SplitOnfeature"].replace(lgbm_feat_dict, inplace=True)
    tree_info_df["Interactions"] = (
        tree_info_df["ParentFeature"].map(str)
        + " - "
        + tree_info_df["SplitOnfeature"].map(str)
    )
    tree_info_df = round(tree_info_df, 2)
    lgb_inter_calc = (
        tree_info_df.groupby("Interactions")["Gain"]
        .agg(["count", "sum", "min", "max", "mean", "std"])
        .sort_values(by="sum", ascending=False)
        .reset_index("Interactions")
        .fillna(0)
    )
    lgb_inter_calc = round(lgb_inter_calc, 2)
    lgb_inter_calc_nobase = lgb_inter_calc[
        lgb_inter_calc["Interactions"].str.contains("base") is False
    ]
    data = (
        lgb_inter_calc_nobase.sort_values("sum", ascending=False)
        .iloc[0:75]
        .reset_index(drop=True)
    )
    plt.figure(figsize=(20, 14))
    ax = plt.subplot(121)
    sns.barplot(
        x="sum", y="Interactions", data=data.sort_values("sum", ascending=False), ax=ax
    )
    ax.set_title("Total Gain for Feature Interaction", fontweight="bold", fontsize=14)
    ax = plt.subplot(122)
    sns.barplot(
        x="count",
        y="Interactions",
        data=data.sort_values("sum", ascending=False),
        ax=ax,
    )
    ax.set_title("No. of times Feature interacted", fontweight="bold", fontsize=14)
    plt.tight_layout()
