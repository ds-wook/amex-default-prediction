import gc
import warnings
from pathlib import Path
from typing import Tuple

import hydra
import lightgbm as lgb
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

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

        # model
        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=dict(config.model.params),
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


def split_dataset(a: np.ndarray, n: int) -> Tuple[np.ndarray]:
    """
    Split array into n parts
    Args:
        a: array
        n: number of parts
    Returns:
        array of tuple
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def predict_meta_features(config: DictConfig, test_x: pd.DataFrame) -> np.ndarray:
    """
    load model
    Args:
        config: config
        test_x: test dataframe
    Returns:
        predictions
    """
    path = Path(get_original_cwd()) / config.model.path / config.model.working
    models = [
        lgb.Booster(model_file=path / f"{config.model.name}_fold{fold}.lgb")
        for fold in range(1, 5 + 1)
    ]
    preds = np.zeros(len(test_x))

    for model in tqdm(models, total=len(models)):
        preds += model.predict(test_x) / len(models)

    return preds


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.dataset.path
    # create dataset
    train = pd.read_parquet(path / "train.parquet")
    target = pd.read_csv(path / "train_labels.csv")
    test = pd.read_parquet(path / "test.parquet")
    train = pd.merge(train, target, on="customer_ID")
    train_x = train.drop(
        columns=[*cfg.dataset.drop_features] + [cfg.dataset.target], axis=1
    )
    train_y = train[cfg.dataset.target]

    # make meta features
    oof_preds = make_meta_features(train_x, train_y, cfg)
    train["preds"] = oof_preds
    train = train.drop(cfg.dataset.target, axis=1)
    train.to_parquet(path / "train_meta.parquet")

    del train, train_x, train_y, oof_preds

    # build test features
    split_ids = split_dataset(test.customer_ID.unique(), 10)
    # infer test
    preds_proba = []

    for (i, ids) in enumerate(split_ids):
        print(f"Inferring test fold {i}")
        test_sample = test[test.customer_ID.isin(ids)]
        test_x = test_sample.drop(columns=[*cfg.dataset.drop_features], axis=1)
        preds = predict_meta_features(cfg, test_x)
        preds_proba.extend(preds)

    test["preds"] = preds_proba
    test.to_parquet(path / "test_meta.parquet")


if __name__ == "__main__":
    _main()
