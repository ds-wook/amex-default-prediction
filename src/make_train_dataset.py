import gc
from pathlib import Path
from typing import NoReturn

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.dataset import split_dataset
from features.build import build_features


@hydra.main(config_path="../config/", config_name="data")
def _main(cfg: DictConfig) -> NoReturn:
    path = Path(get_original_cwd())
    train = pd.read_parquet(path / "input/amex-data-parquet/train.parquet")
    label = pd.read_csv(path / "input/amex-default-prediction/train_labels.csv")
    split_ids = split_dataset(train.customer_ID.unique(), cfg.dataset.num_train)

    path = Path(get_original_cwd()) / cfg.dataset.path

    for (i, ids) in enumerate(split_ids):
        train_sample = train[train.customer_ID.isin(ids)]
        train_agg = build_features(train_sample)

        print(i, train_agg.shape)
        train_agg.to_parquet(path / f"{cfg.dataset.train}_{i}.parquet")

        del train_agg
        gc.collect()

    train = pd.read_parquet(path / f"{cfg.dataset.train}_0.parquet")

    for num in range(1, cfg.dataset.num):
        train_sample = pd.read_parquet(path / f"{cfg.dataset.train}_{num}.parquet")
        train = pd.concat([train, train_sample], axis=0)
        del train_sample
        gc.collect()

    train = pd.merge(train, label, on="customer_ID")

    print(train.shape)

    train.to_parquet(path / f"{cfg.dataset.train}.parquet")


if __name__ == "__main__":
    _main()
