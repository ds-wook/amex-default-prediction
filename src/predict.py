import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.dataset import load_test_dataset
from features.build import create_categorical_test
from models.infer import load_model, predict
from utils import seed_everything


@hydra.main(config_path="../config/", config_name="predict")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.output.path

    # model load
    results = load_model(cfg, cfg.models.name)

    # infer test
    preds_proba = []

    for num in range(10):
        seed_everything(cfg.models.params.seed)
        test_sample = load_test_dataset(cfg, num)
        test_sample = create_categorical_test(test_sample, cfg)

        logging.info(f"Test dataset {num} predicting...")
        preds = predict(results, test_sample)
        preds_proba.extend(preds.tolist())

    submit = pd.read_csv(path / cfg.output.submit)
    submit["prediction"] = preds_proba
    submit.to_csv(path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
