import gc
import logging
import time
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from shap import TreeExplainer

from evaluation.evaluate import amex_metric


def select_features(
    train: pd.DataFrame,
    label: pd.Series,
    test: pd.DataFrame,
    seed: int,
    cat_features: List[str],
) -> Tuple[List[str], List[str]]:
    # train LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(train, label, free_raw_data=False, silent=True)
    print("Train Start!")
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
    importance_df = importance_total_df.query("shap_importance != 0")
    boosting_shap_col = importance_df.column_name.values.tolist()
    logging.info(f"Select {len(boosting_shap_col)} features")

    selected_cat_features = [col for col in boosting_shap_col if col in cat_features]
    del test
    gc.collect()

    return boosting_shap_col, selected_cat_features, importance_total_df


def get_feature_importances(
    data: pd.DataFrame, label: pd.Series, config: DictConfig, shuffle: bool = True
) -> pd.DataFrame:
    # Go over fold and keep track of CV score (train and valid) and feature importances

    if shuffle:
        # Here you could as well use a binomial distribution
        y = label.copy().sample(frac=1.0)

    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data, label, free_raw_data=False, silent=True)
    lgb_params = {
        "objective": "binary",
        "boosting_type": "rf",
        "subsample": 0.623,
        "colsample_bytree": 0.7,
        "num_leaves": 127,
        "max_depth": 8,
        "seed": 42,
        "bagging_freq": 1,
        "n_jobs": 4,
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


def importance_null(
    data: pd.DataFrame, label: pd.Series, config: DictConfig
) -> pd.DataFrame:
    null_imp_df = pd.DataFrame()
    nb_runs = 80
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
