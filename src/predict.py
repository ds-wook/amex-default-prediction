import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.dataset import load_test_dataset
from features.build import (
    add_diff_features,
    add_trick_features,
    create_categorical_test,
)
from models.infer import inference, load_model
from utils import seed_everything


@hydra.main(config_path="../config/", config_name="predict")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.output.path

    # model load
    results = load_model(cfg, cfg.model.name)

    # infer test
    preds_proba = []

    for num in range(10):
        seed_everything(cfg.model.params.seed)
        test_sample = load_test_dataset(cfg, num)
        test_sample = create_categorical_test(test_sample, cfg)
        # test_sample = test_sample[cfg.features.selected_features]
        test_sample = add_trick_features(test_sample)
        test_sample = add_diff_features(test_sample)

        logging.info(f"Test dataset {num} predicting...")
        preds = inference(results, test_sample)
        preds_proba.extend(preds.tolist())

    submit = pd.read_csv(path / cfg.output.submit)
    submit["prediction"] = preds_proba
    submit.to_csv(path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
