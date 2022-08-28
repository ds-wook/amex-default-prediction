import gc
import warnings
from pathlib import Path

import hydra
import lightgbm as lgb
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


def make_meta_features(
    train_x: pd.DataFrame, train_y: pd.Series, config: DictConfig
) -> pd.DataFrame:
    """
    Create meta features
    Args:
        df: dataframe
    Returns:
        dataframe
    """
    models = dict()
    folds = config.model.fold
    seed = config.dataset.seed

    str_kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = str_kf.split(train_x, train_y)
    oof_preds = np.zeros(len(train_x))

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        # split train and validation data
        X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
        X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]
        train_set = lgb.Dataset(data=X_train, label=y_train)
        valid_set = lgb.Dataset(data=X_valid, label=y_valid)

        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
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
        # model
        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=lgb_params,
            verbose_eval=config.model.verbose,
            num_boost_round=config.model.num_boost_round,
            early_stopping_rounds=config.model.early_stopping_rounds,
        )
        models[f"fold_{fold}"] = model
        path = Path(get_original_cwd()) / config.model.path / config.model.working
        model_name = f"{config.model.name}_fold{fold}.lgb"
        model_path = path / model_name

        model.save_model(model_path)
        # validation
        oof_preds[valid_idx] = model.predict(X_valid)

        del X_train, X_valid, y_train, y_valid, model
        gc.collect()

    return oof_preds


def load_model(config: DictConfig, fold: int) -> lgb.Booster:
    """
    load model
    Args:
        model_path: path to model
    Returns:
        model
    """
    path = Path(get_original_cwd()) / config.model.path / config.model.working
    model_path = path / f"{config.model.name}_fold{fold}.lgb"
    return lgb.Booster(model_file=model_path)


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.dataset.path
    # create dataset
    train = pd.read_parquet(path / "train.parquet")
    target = pd.read_csv(path / "train_labels.csv")

    train = pd.merge(train, target, on="customer_ID")
    train_x = train.drop(
        columns=[*cfg.dataset.drop_features] + [cfg.dataset.target], axis=1
    )
    train_y = train[cfg.dataset.target]

    # make meta features
    oof_preds = make_meta_features(train_x, train_y, cfg)
    train["preds"] = oof_preds
    train.to_parquet(path / "train_meta.parquet")
    test = pd.read_parquet(path / "test.parquet")
    test_x = test.drop(columns=[*cfg.dataset.drop_features], axis=1)
    preds = np.zeros(len(test_x))
    models = [load_model(cfg, fold) for fold in range(1, 5 + 1)]

    for model in models:
        preds += model.predict(test_x) / len(models)

    test["preds"] = preds
    test.to_parquet(path / "test_meta.parquet")


if __name__ == "__main__":
    _main()
