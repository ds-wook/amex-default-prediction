import gc
import time
from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from shap import TreeExplainer

from evaluation.evaluate import amex_metric, lgb_amex_metric


def get_shap_importances(
    train: pd.DataFrame,
    label: pd.Series,
    test: pd.DataFrame,
    cat_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    # train LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(train, label, free_raw_data=False, silent=True)
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting": "rf",
        "seed": 42,
        "num_leaves": 100,
        "learning_rate": 0.01,
        "feature_fraction": 0.20,
        "bagging_freq": 10,
        "bagging_fraction": 0.50,
        "n_jobs": -1,
        "lambda_l2": 2,
        "min_data_in_leaf": 40,
    }

    # train the model
    model = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=200,
        categorical_feature=cat_features,
    )

    print("Explain Start!!")
    explainer = TreeExplainer(model)
    del train, label, model
    gc.collect()

    shap_values = explainer.shap_values(test)
    shap_sum = np.abs(shap_values).mean(axis=1).sum(axis=0)

    importance_df = pd.DataFrame([test.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ["column_name", "shap_importance"]

    importance_total_df = importance_df.sort_values("shap_importance", ascending=False)

    return importance_total_df


def get_feature_importances(
    data: pd.DataFrame, label: pd.Series, config: DictConfig, shuffle: bool = True
) -> pd.DataFrame:
    """
    Get feature importances
    Args:
        data: dataframe
        label: label
        config: config
        shuffle: shuffle
    Returns:
        imp_df: importance dataframe
    """
    # Go over fold and keep track of CV score (train and valid) and feature importances
    y = label.copy().sample(frac=1.0) if shuffle else label

    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data, y, free_raw_data=False, silent=True)
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting": "rf",
        "seed": config.model.params.seed,
        "num_leaves": 100,
        "learning_rate": 0.01,
        "feature_fraction": 0.20,
        "bagging_freq": 10,
        "bagging_fraction": 0.50,
        "n_jobs": -1,
        "lambda_l2": 2,
        "min_data_in_leaf": 40,
    }

    # train the model
    model = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=200,
        categorical_feature=config.dataset.cat_features,
    )

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(data.columns)
    imp_df["importance_gain"] = model.feature_importance(importance_type="gain")
    imp_df["importance_split"] = model.feature_importance(importance_type="split")
    imp_df["trn_score"] = amex_metric(y, model.predict(data))

    return imp_df


def get_null_importances(
    data: pd.DataFrame, label: pd.Series, config: DictConfig
) -> pd.DataFrame:
    """
    Null importance
    Args:
        data: dataframe
        label: label
        config: config
    Returns:
        imp_df: importance dataframe
    """
    null_imp_df = pd.DataFrame()
    nb_runs = config.features.nb_runs
    start = time.time()
    dsp = ""

    for i in range(nb_runs):
        # Get current run importances
        imp_df = get_feature_importances(
            data=data, label=label, config=config, shuffle=True
        )
        imp_df["run"] = i + 1
        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        # Erase previous message
        for _ in range(len(dsp)):
            print("\b", end="", flush=True)
        # Display current run and time used
        spent = (time.time() - start) / 60
        dsp = "Done with %4d of %4d (Spent %5.1f min)" % (i + 1, nb_runs, spent)
        print(dsp, end="", flush=True)

    return null_imp_df


def get_score_correlation(
    actual_imp_df: pd.DataFrame, null_imp_df: pd.Series
) -> Tuple[pd.DataFrame, List[Tuple[str, float, float]]]:
    correlation_scores = []
    for _f in actual_imp_df["feature"].unique():
        f_null_imps = null_imp_df.loc[
            null_imp_df["feature"] == _f, "importance_gain"
        ].values
        f_act_imps = actual_imp_df.loc[
            actual_imp_df["feature"] == _f, "importance_gain"
        ].values
        gain_score = (
            100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        )
        f_null_imps = null_imp_df.loc[
            null_imp_df["feature"] == _f, "importance_split"
        ].values
        f_act_imps = actual_imp_df.loc[
            actual_imp_df["feature"] == _f, "importance_split"
        ].values
        split_score = (
            100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        )
        correlation_scores.append((_f, split_score, gain_score))

    corr_scores_df = pd.DataFrame(
        correlation_scores, columns=["feature", "split_score", "gain_score"]
    )

    return corr_scores_df, correlation_scores


def score_feature_selection(
    df: Optional[pd.DataFrame] = None,
    train_features: Optional[List[str]] = None,
    cat_features: Optional[List[str]] = None,
    target: Optional[pd.Series] = None,
):
    # Fit LightGBM
    dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, silent=True)
    lgb_params = {
        "objective": "binary",
        "boosting": "gbdt",
        "seed": 42,
        "num_leaves": 100,
        "learning_rate": 0.1,
        "feature_fraction": 0.20,
        "bagging_freq": 10,
        "bagging_fraction": 0.50,
        "n_jobs": -1,
        "lambda_l2": 2,
        "min_data_in_leaf": 40,
        "verbose": -1,
    }

    # Fit the model
    hist = lgb.cv(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=2000,
        categorical_feature=cat_features,
        nfold=5,
        feval=lgb_amex_metric,
        stratified=True,
        shuffle=True,
        early_stopping_rounds=50,
        verbose_eval=0,
        seed=42,
    )
    # Return the last mean / std values
    return hist["amex-mean"][-1], hist["amex-stdv"][-1]
