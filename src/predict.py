from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from models.infer import load_model, predict


@hydra.main(config_path="../config/", config_name="predict.yaml")
def _main(cfg: DictConfig):
    path = Path(get_original_cwd()) / cfg.output.path
    submit_path = Path(get_original_cwd()) / cfg.dataset.submit_path
    # model load
    result = load_model(cfg.model)

    # infer test
    preds = predict(result, cfg)
    submit = pd.read_csv(submit_path / cfg.dataset.submit)
    submit["prediction"] = preds
    submit.to_csv(path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
