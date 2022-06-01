from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.dataset import load_cat_test_dataset
from models.infer import load_model, predict
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/", config_name="predict.yaml")
def _main(cfg: DictConfig):
    test_x = load_cat_test_dataset(cfg)
    test_x = reduce_mem_usage(test_x)
    path = Path(get_original_cwd()) / cfg.output.path

    # model load
    lgb_results = load_model(cfg.model.lightgbm)

    # infer test
    pred = predict(lgb_results, test_x)
    submit = pd.DataFrame({"customer_ID": test_x.index, "prediction": pred})
    submit.to_csv(path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()