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
    test = pd.read_parquet(path / "input/amex-data-parquet/test.parquet")
    split_ids = split_dataset(test.customer_ID.unique(), cfg.dataset.num_test)

    path = Path(get_original_cwd()) / cfg.dataset.path

    for (i, ids) in enumerate(split_ids):
        test_sample = test[test.customer_ID.isin(ids)]
        test_agg = build_features(test_sample)
        print(i, test_agg.shape)

        test_agg.to_parquet(path / f"{cfg.dataset.test}_{i}.parquet")

        del test_agg
        gc.collect()

    del test
    gc.collect()


if __name__ == "__main__":
    _main()
